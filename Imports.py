import os
import math
import numpy as np
import dill as pickle
import pymatreader as pmt
from tqdm.auto import tqdm

from joblib import Parallel, delayed

from dipy.core.gradients import gradient_table
from dipy.data import get_fnames
from dipy.io.gradients import read_bvals_bvecs
from dipy.sims.voxel import single_tensor

from dipy.io.image import load_nifti
from dipy.segment.mask import median_otsu
from dipy.align.reslice import reslice

from dipy.align.imaffine import (transform_centers_of_mass,
                                 AffineMap,
                                 MutualInformationMetric,
                                 AffineRegistration)
from dipy.align.transforms import (TranslationTransform3D,
                                   RigidTransform3D,
                                   AffineTransform3D)

import scipy as sp
import scipy.stats as stats
from scipy.spatial.distance import pdist, squareform
from scipy.ndimage import gaussian_filter,binary_dilation

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.colors import TwoSlopeNorm

import torch
from torch import Tensor

from skimage.metrics import structural_similarity as ssim

from sbi.inference import SNPE, DirectPosterior
from sbi.utils import BoxUniform



WLSFit   = 'sandybrown'
SBIFit   = np.array([64,176,166])/255

MSDir = './MS_data/'
HCPDir = './HCP_data/'
NetworkDir = './Networks/'
RetestDir = './Retest/'

RTNames = ['070618','131217','180518','210516','231017','310118']


ChunkSize = 128

font = {
    'family': 'sans-serif',  # Use sans-serif family
    'sans-serif': ['Helvetica'],  # Specify Helvetica as the sans-serif font
    'size': 14  # Set the default font size
}
plt.rc('font', **font)

# Set tick label sizes
plt.rc('ytick', labelsize=24)
plt.rc('xtick', labelsize=24)

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "Helvetica"
})
# Customize axes spines and legend appearance
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['legend.frameon'] = False


def masked_local_ssim(img1, img2, mask, win_size=15,dat_range=None):
    half_win = win_size // 2
    padded_img1 = np.pad(img1, half_win, mode='reflect')
    padded_img2 = np.pad(img2, half_win, mode='reflect')
    padded_mask = np.pad(mask, half_win, mode='constant', constant_values=0)

    ssim_values = []

    rows, cols = img1.shape
    for i in range(rows):
        for j in range(cols):
            mask_patch = padded_mask[i:i+win_size, j:j+win_size]
            if mask_patch.all():  # Only if fully valid
                img1_patch = padded_img1[i:i+win_size, j:j+win_size]
                img2_patch = padded_img2[i:i+win_size, j:j+win_size]
                if(not (np.isnan(img1_patch).any() or np.isnan(img2_patch).any())):
                    if(dat_range is None):
                        val = ssim(img1_patch, img2_patch,
                                   data_range=max(img1.max() - img1.min(),img2.max()-img2.min()), full=False)
                    else:
                        val = ssim(img1_patch, img2_patch,
                                   data_range=dat_range, full=False)
                    ssim_values.append(val)

    return np.nanmean(ssim_values)


# plotting function
def BoxPlots(y_data, positions, colors, colors2, ax,hatch = False,scatter=False,scatter_alpha=0.5,jitter=0.02, **kwargs):

    GREY_DARK = "#747473"
    # Clean data to remove NaNs column-wise
    if(np.ndim(y_data) == 1):
        cleaned_data = y_data[~np.isnan(y_data)]
    else:
        cleaned_data = [d[~np.isnan(d)] for d in y_data]
    
    # Define properties for the boxes (patch objects)
    boxprops = dict(
        linewidth=2, 
        facecolor='none',       # use facecolor for filling (set to 'none' if you want no fill)
        edgecolor='turquoise'   # edgecolor for the outline
    )

    # Define properties for the medians (Line2D objects)
    # Ensure GREY_DARK is defined (or replace it with a color string)
    medianprops = dict(
        linewidth=2, 
        color=GREY_DARK,
        solid_capstyle="butt"
    )

    # For whiskers, since they are Line2D objects, use 'color'
    whiskerprops = dict(
        linewidth=2, 
        color='turquoise'
    )

    bplot = ax.boxplot(
        cleaned_data,
        positions=positions, 
        showfliers=False,
        showcaps = False,
        medianprops=medianprops,
        whiskerprops=whiskerprops,
        boxprops=boxprops,
        patch_artist=True,
        **kwargs
    )

    # Update the color of each box (these are patch objects)
    for i, box in enumerate(bplot['boxes']):
        box.set_edgecolor(colors[i])
        if(hatch):
            box.set_hatch('/')
    
    
    # Update the color of the whiskers (each box has 2 whiskers)
    for i in range(len(positions)):
        bplot['whiskers'][2*i].set_color(colors[i])
        bplot['whiskers'][2*i+1].set_color(colors[i])
    
    # If caps are enabled, update their color (Line2D objects)
    if 'caps' in bplot:
        for i, cap in enumerate(bplot['caps']):
            cap.set_color(colors[i//2])  # two caps per box

    if(scatter):
        if(np.ndim(cleaned_data) == 1):
            x_data = np.array([positions] * len(cleaned_data)).squeeze()
            x_jittered = x_data + stats.t(df=6, scale=jitter).rvs(len(x_data))
            ax.scatter(x_jittered, cleaned_data, s=100, color=colors2, alpha=scatter_alpha)
        else:
            x_data = [np.array([positions[i]] * len(d)) for i, d in enumerate(cleaned_data)]
            x_jittered = [x + stats.t(df=6, scale=jitter).rvs(len(x)) for x in x_data]
            # Plot the scatter points with jitter (using colors2)
            for x, y, c in zip(x_jittered, cleaned_data, colors2):
                ax.scatter(x, y, s=100, color=c, alpha=scatter_alpha)

def vals_to_mat(dt):
    DTI = np.zeros((3,3))
    DTI[0,0] = dt[0]
    DTI[0,1],DTI[1,0] =  dt[1],dt[1]
    DTI[1,1] =  dt[2]
    DTI[0,2],DTI[2,0] =  dt[3],dt[3]
    DTI[1,2],DTI[2,1] =  dt[4],dt[4]
    DTI[2,2] =  dt[5]
    return DTI

def mat_to_vals(DTI):
    dt = np.zeros(6)
    dt[0] = DTI[0,0]
    dt[1] = DTI[0,1]
    dt[2] = DTI[1,1]
    dt[3] = DTI[0,2]
    dt[4] = DTI[1,2]
    dt[5] = DTI[2,2]
    return dt

def GenRicciNoise(signal,S0,snr):

    size = signal.shape
    sigma = S0 / snr
    noise1 = np.random.normal(0, sigma, size=size)
    noise2 = np.random.normal(0, sigma, size=size)

    return np.sqrt((signal+noise1) ** 2 + noise2 ** 2)

def AddNoise(signal,S0,snr):
    
    return GenRicciNoise(signal,S0,snr)

def FracAni(evals,MD):
    numerator = np.sqrt(3 * np.sum((evals - MD) ** 2))
    denominator = np.sqrt(2) * np.sqrt(np.sum(evals ** 2))
    
    return numerator / denominator

def clip_negative_eigenvalues(matrix):
    # Perform eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    
    # Clip negative eigenvalues to 0
    clipped_eigenvalues = np.maximum(eigenvalues, 1e-5)
    
    # Reconstruct the matrix with the clipped eigenvalues
    clipped_matrix = eigenvectors @ np.diag(clipped_eigenvalues) @ np.linalg.inv(eigenvectors)
    
    return clipped_matrix

def rigid_register(static, moving, static_aff, moving_aff,
                   nbins=32, level_iters=(1000, 200, 50),
                   sigmas=(3.0, 1.0, 0.0), factors=(4, 2, 1)):

    static = static.astype(np.float32)
    moving = moving.astype(np.float32)

    metric = MutualInformationMetric(nbins=nbins, sampling_proportion=None)

    affreg = AffineRegistration(metric=metric,
                                level_iters=list(level_iters),
                                sigmas=list(sigmas),
                                factors=list(factors))

    # 1) translation init
    trans = affreg.optimize(
        static=static, moving=moving,
        transform=TranslationTransform3D(),
        params0=None,
        static_grid2world=static_aff,
        moving_grid2world=moving_aff,
    )

    # 2) rigid refinement
    rigid_map = affreg.optimize(
        static=static, moving=moving,
        transform=RigidTransform3D(),
        params0=None,
        static_grid2world=static_aff,
        moving_grid2world=moving_aff,
        starting_affine=trans.affine,
    )

    return rigid_map