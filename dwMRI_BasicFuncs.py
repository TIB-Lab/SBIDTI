import numpy as np
import dill as pickle
import os
import math
import scipy as sp



import scipy.stats as stats
from scipy.spatial.distance import pdist, squareform
from scipy.stats import lognorm,norm

from skimage.metrics import structural_similarity as ssim


from sbi import analysis as analysis
from sbi import utils as utils
from sbi.inference import SNPE, simulate_for_sbi
from sbi.inference.potentials.posterior_based_potential import posterior_estimator_based_potential
from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
)
from sbi.utils import process_prior,BoxUniform

import torch
from torch.distributions import Categorical,Normal
from torch import Tensor


from dipy.sims.voxel import single_tensor
from dipy.data import get_fnames
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.reconst.dti import (decompose_tensor, from_lower_triangular)
from dipy.io.image import load_nifti
from dipy.segment.mask import median_otsu
import dipy.reconst.dti as dti
import dipy.reconst.dki as dki
from dipy.align.reslice import reslice
from dipy.core.sphere import disperse_charges, Sphere, HemiSphere

import pymatreader as pmt
from scipy.special import j0, jv
from scipy.optimize import bisect

### DTI Functions ##################### ##################### ##################### ##################### ##################### #####################

def j1_derivative(x):
    """Derivative of J1(x) using the identity: J1'(x) = 0.5 * (J0(x) - J2(x))."""
    return 0.5 * (j0(x) - j2(x))

def j2(x):
    """Bessel function J_2(x)."""
    return jv(2, x)

def j1prime_zeros(n, x_max=100, step=0.1):
    """
    Find the first n positive roots of J1'(x) by scanning from x=0 to x_max.
    
    Parameters
    ----------
    n     : int
        Number of roots to find
    x_max : float
        Maximum x to search
    step  : float
        Step size for scanning sign changes
    
    Returns
    -------
    zeros : list of float
        List of the first n roots (x > 0) of J1'(x).
    """
    zeros = []
    x_vals = np.arange(0.0, x_max, step)
    
    f_prev = j1_derivative(x_vals[0])
    for i in range(1, len(x_vals)):
        f_curr = j1_derivative(x_vals[i])
        # Check for a sign change in [x_vals[i-1], x_vals[i]]
        if f_prev * f_curr < 0:
            root = bisect(j1_derivative, x_vals[i-1], x_vals[i])
            zeros.append(root)
            if len(zeros) == n:
                break
        f_prev = f_curr
    
    return zeros

n_roots = 100
Bessel_roots = np.array(j1prime_zeros(n_roots, x_max=10e6, step=0.01))
Bessel = True

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

def fill_lower_diag(a):
    b = [a[0],a[3],a[1],a[4],a[5],a[2]]
    n = 3
    mask = np.tri(n,dtype=bool) 
    out = np.zeros((n,n),dtype=float)
    out[mask] = b
    return out

def ComputeDTI(params):
    L = fill_lower_diag(params)
    
    np.fill_diagonal(L, np.abs(np.diagonal(L)))

    A = L @ L.T
    return A

def ForceLowFA(dt):
    # Modify the matrix to ensure low FA (more isotropic)
    eigenvalues, eigenvectors = np.linalg.eigh(dt)
    
    # Make the eigenvalues more similar to enforce low FA
    mean_eigenvalue = np.mean(eigenvalues)

    adjusted_eigenvalues = np.clip(eigenvalues, mean_eigenvalue * np.random.rand(), mean_eigenvalue * 1.0)
    
    # Reconstruct the matrix with the adjusted eigenvalues
    dt_low_fa = eigenvectors @ np.diag(adjusted_eigenvalues) @ eigenvectors.T
    
    return dt_low_fa
    
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


### DKI Functions ##################### ##################### ##################### ##################### ##################### #####################

def FitDT(Dat,seed=1):

    np.random.seed(seed)
    # DT_abc
    data = Dat[:,0]
    shape,loc,scale = lognorm.fit(data)
    
    dti1_fitted = stats.lognorm(shape, loc=loc, scale=scale)

    #DT_rest
    data = Dat[:,1]
    loc,scale = norm.fit(data)
    
    # Compute the fitted PDF
    dti2_fitted = stats.norm(loc=loc, scale=scale)

    return dti1_fitted,dti2_fitted

def FitKT(Dat,seed=1):
    np.random.seed(seed)    
    # Fitting x4
    data = Dat[:,0]
    shape,loc,scale = lognorm.fit(data)
    x4_fitted = stats.lognorm(shape, loc=loc, scale=scale)
    
    # Fitting R1
    data = Dat[:,3]
    loc,scale = norm.fit(data)
    R1_fitted = norm(loc,scale)
    
    # Fitting x2
    data = Dat[:,9]
    shape,loc,scale = lognorm.fit(data)
    x2_fitted = stats.lognorm(shape, loc=loc, scale=scale)

    # Fitting R2
    data = Dat[:,12]
    loc,scale = norm.fit(data)
    R2_fitted = norm(loc,scale)


    return x4_fitted,R1_fitted,x2_fitted,R2_fitted

def GenDTKT(DT_Fits,KT_Fits,seed,size):

    np.random.seed(seed)
    DT = np.zeros([size,6])
    KT = np.zeros([size,15])

    DT[:,0] = DT_Fits[0].rvs(size)
    DT[:,2] = DT_Fits[0].rvs(size)
    DT[:,5] = DT_Fits[0].rvs(size)

    DT[:,1] = DT_Fits[1].rvs(size)
    DT[:,3] = DT_Fits[1].rvs(size)
    DT[:,4] = DT_Fits[1].rvs(size)

    for k in range(3):
        KT[:,k] = KT_Fits[0].rvs(size)
    for k in range(3,9):
        KT[:,k] = KT_Fits[1].rvs(size)
    for k in range(9,12):
        KT[:,k] = KT_Fits[2].rvs(size)
    for k in range(12,15):
        KT[:,k] = KT_Fits[3].rvs(size)

    return DT,KT
    
def DKIMetrics(dt,kt,analytical=True):
    if(dt.ndim == 1):
        dt = vals_to_mat(dt)
    evals,evecs = np.linalg.eigh(dt)
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]
    
    params = np.concatenate([evals,np.hstack(evecs),kt])
    params2 = np.concatenate([evals,np.hstack(evecs),-kt])

    mk = dki.mean_kurtosis(params,analytical=analytical,min_kurtosis=-3.0 / 7, max_kurtosis=np.inf)

    ak = dki.axial_kurtosis(params,analytical=analytical,min_kurtosis=-3.0 / 7, max_kurtosis=np.inf)

    rk = dki.radial_kurtosis(params,analytical=analytical,min_kurtosis=-3.0 / 7, max_kurtosis=np.inf)

    mkt = dki.mean_kurtosis_tensor(params, min_kurtosis=-3.0 / 7, max_kurtosis=np.inf)

    kfa = kurtosis_fractional_anisotropy_test(params)

    return mk,ak,rk,mkt,kfa

def kurtosis_fractional_anisotropy_test(dki_params):
    r"""Compute the anisotropy of the kurtosis tensor (KFA).

    See :footcite:p:`Glenn2015` and :footcite:p:`NetoHenriques2021` for further
    details about the method.

    Parameters
    ----------
    dki_params : ndarray (x, y, z, 27) or (n, 27)
        All parameters estimated from the diffusion kurtosis model.
        Parameters are ordered as follows:
            1) Three diffusion tensor's eigenvalues
            2) Three lines of the eigenvector matrix each containing the first,
                second and third coordinates of the eigenvector
            3) Fifteen elements of the kurtosis tensor

    Returns
    -------
    kfa : array
        Calculated mean kurtosis tensor.

    Notes
    -----
    The KFA is defined as :footcite:p:`Glenn2015`:

    .. math::

         KFA \equiv
         \frac{||\mathbf{W} - MKT \mathbf{I}^{(4)}||_F}{||\mathbf{W}||_F}

    where $W$ is the kurtosis tensor, MKT the kurtosis tensor mean, $I^{(4)}$ is
    the fully symmetric rank 2 isotropic tensor and $||...||_F$ is the tensor's
    Frobenius norm :footcite:p:`Glenn2015`.

    References
    ----------
    .. footbibliography::

    """
    Wxxxx = dki_params[..., 12]
    Wyyyy = dki_params[..., 13]
    Wzzzz = dki_params[..., 14]
    Wxxxy = dki_params[..., 15]
    Wxxxz = dki_params[..., 16]
    Wxyyy = dki_params[..., 17]
    Wyyyz = dki_params[..., 18]
    Wxzzz = dki_params[..., 19]
    Wyzzz = dki_params[..., 20]
    Wxxyy = dki_params[..., 21]
    Wxxzz = dki_params[..., 22]
    Wyyzz = dki_params[..., 23]
    Wxxyz = dki_params[..., 24]
    Wxyyz = dki_params[..., 25]
    Wxyzz = dki_params[..., 26]


    W = 1.0 / 5.0 * (Wxxxx + Wyyyy + Wzzzz + 2 * Wxxyy + 2 * Wxxzz + 2 * Wyyzz)
    # Compute's equation numerator
    A = (
        (Wxxxx - W) ** 2
        + (Wyyyy - W) ** 2
        + (Wzzzz - W) ** 2
        + 4 * (Wxxxy**2 + Wxxxz**2 + Wxyyy**2 + Wyyyz**2 + Wxzzz**2 + Wyzzz**2)
        + 6 * ((Wxxyy - W / 3) ** 2 + (Wxxzz - W / 3) ** 2 + (Wyyzz - W / 3) ** 2)
        + 12 * (Wxxyz**2 + Wxyyz**2 + Wxyzz**2)
    )
    # Compute's equation denominator
    B = (
        Wxxxx**2
        + Wyyyy**2
        + Wzzzz**2
        + 4 * (Wxxxy**2 + Wxxxz**2 + Wxyyy**2 + Wyyyz**2 + Wxzzz**2 + Wyzzz**2)
        + 6 * (Wxxyy**2 + Wxxzz**2 + Wyyzz**2)
        + 12 * (Wxxyz**2 + Wxyyz**2 + Wxyzz**2)
    )

    # Compute KFA
    KFA = np.zeros(A.shape)
    KFA = np.sqrt(A / B)

    return KFA


### Simulators ##################### ##################### ##################### ##################### ##################### #####################

def CustomSimulator(Mat,gtab,S0,snr=None):
    evals,evecs = np.linalg.eigh(Mat)
    signal = single_tensor(gtab, S0=S0, evals=evals, evecs=evecs)
    if(snr is None):
        return signal
    else:
        return AddNoise(signal,S0,snr)

def Simulator(bvals,bvecs,S0,params,SNR):

    dt = ComputeDTI(params)
    signal_dti = CustomSimulator(dt,gradient_table(bvals, bvecs),S0,SNR)
    
    return signal_dti


def GenRicciNoise(signal,S0,snr):

    size = signal.shape
    sigma = S0 / snr
    noise1 = np.random.normal(0, sigma, size=size)
    noise2 = np.random.normal(0, sigma, size=size)

    return np.sqrt((signal+noise1) ** 2 + noise2 ** 2)


def AddNoise(signal,S0,snr):
    
    return GenRicciNoise(signal,S0,snr)

def CustomDKISimulator(dt,kt,gtab,S0,snr=None):
    if(dt.ndim == 1):
        dt = vals_to_mat(dt)
    evals,evecs = np.linalg.eigh(dt)
    params = np.concatenate([evals,np.hstack(evecs),kt])
    signal = dki.dki_prediction(params,gtab,S0)
    if(snr is None):
        return signal
    else:
        return AddNoise(signal,S0,snr)

### SBI Priors ##################### ##################### ##################### ##################### ##################### #####################

class DTIPrior:
    def __init__(self, lower_abs : Tensor, upper_abs : Tensor, 
                       lower_rest: Tensor, upper_rest: Tensor,
                        return_numpy: bool = False):

        self.dist_abs = BoxUniform(low= lower_abs* torch.ones(3), high=upper_abs * torch.ones(3))
        self.dist_rest = BoxUniform(low=lower_rest * torch.ones(3), high=upper_rest *torch.ones(3))
        self.return_numpy = return_numpy
        
    def sample(self, sample_shape=torch.Size([])):
        
        abc  = self.dist_abs.sample(sample_shape)
        rest = self.dist_rest.sample(sample_shape)
        
        if self.return_numpy:   
            params = np.hstack([abc,rest]) 
        else:
            params = torch.hstack([abc,rest])

        return params
        
    def log_prob(self, values):
        if self.return_numpy:
            values = torch.as_tensor(values)
        
        abc  = values[:,:3]
        rest = values[:,3:]

        log_prob_abc  = self.dist_abs.log_prob(abc)
        log_prob_rest = self.dist_rest.log_prob(rest)
        return log_prob_abc+log_prob_rest

class DTIPriorS0:
    def __init__(self, lower_abs : Tensor, upper_abs : Tensor, 
                       lower_rest: Tensor, upper_rest: Tensor,
                       lower_S0: Tensor, upper_S0: Tensor,
                        return_numpy: bool = False):

        self.dist_abs = BoxUniform(low= lower_abs* torch.ones(3), high=upper_abs * torch.ones(3))
        self.dist_rest = BoxUniform(low=lower_rest * torch.ones(3), high=upper_rest *torch.ones(3))
        self.dist_S0 = BoxUniform(low=torch.tensor([lower_S0]), high=torch.tensor([upper_S0]))
        self.return_numpy = return_numpy
        
    def sample(self, sample_shape=torch.Size([])):
        
        abc  = self.dist_abs.sample(sample_shape)
        rest = self.dist_rest.sample(sample_shape)
        S0   = self.dist_S0.sample(sample_shape)
        
        if self.return_numpy:   
            params = np.hstack([abc,rest,S0]) 
        else:
            params = torch.hstack([abc,rest,S0])

        return params
        
    def log_prob(self, values):
        if self.return_numpy:
            values = torch.as_tensor(values)
        
        abc  = values[:,:3]
        rest = values[:,3:-1]
        S0   = values[:,-1]

        log_prob_abc  = self.dist_abs.log_prob(abc)
        log_prob_rest = self.dist_rest.log_prob(rest)
        log_prob_S0 = self.dist_S0.log_prob(S0)
        return log_prob_abc+log_prob_rest+log_prob_S0

class DTIPriorS0Direc:
    def __init__(self, lower_abs : Tensor, upper_abs : Tensor, 
                       lower_rest: Tensor, upper_rest: Tensor,
                       lower_S0: Tensor, upper_S0: Tensor,
                        return_numpy: bool = False):

        self.dist_abs = BoxUniform(low= lower_abs* torch.ones(3), high=upper_abs * torch.ones(3))
        self.dist_rest = BoxUniform(low=lower_rest * torch.ones(3), high=upper_rest *torch.ones(3))
        self.dist_S0 = BoxUniform(low=torch.tensor([lower_S0]), high=torch.tensor([upper_S0]))
        self.direction_choice = Categorical(probs=torch.ones(1, 5))
        self.return_numpy = return_numpy
        
    def sample(self, sample_shape=torch.Size([])):
        
        abc  = self.dist_abs.sample(sample_shape)
        rest = self.dist_rest.sample(sample_shape)
        S0   = self.dist_S0.sample(sample_shape)
        direc = self.direction_choice.sample(sample_shape)       
        
        if self.return_numpy:   
            params = np.hstack([abc,rest,S0,direc]) 
        else:
            params = torch.hstack([abc,rest,S0,direc])

        return params
        
    def log_prob(self, values):
        if self.return_numpy:
            values = torch.as_tensor(values)
        
        abc  = values[:,:3]
        rest = values[:,3:-2]
        S0   = values[:,-2]
        direc   = values[:,-1]

        log_prob_abc   = self.dist_abs.log_prob(abc)
        log_prob_rest  = self.dist_rest.log_prob(rest)
        log_prob_S0    = self.dist_S0.log_prob(S0)
        log_prob_direc =  self.direction_choice.log_prob(direc)
        return log_prob_abc+log_prob_rest+log_prob_S0+log_prob_direc

class DTIPriorS0Noise:
    def __init__(self, lower_abs : Tensor, upper_abs : Tensor, 
                       lower_rest: Tensor, upper_rest: Tensor,
                       lower_S0: Tensor, upper_S0: Tensor,
                       lower_noise: Tensor, upper_noise: Tensor,
                        return_numpy: bool = False):

        self.dist_abs = BoxUniform(low= lower_abs* torch.ones(3), high=upper_abs * torch.ones(3))
        self.dist_rest = BoxUniform(low=lower_rest * torch.ones(3), high=upper_rest *torch.ones(3))
        self.dist_S0 = BoxUniform(low=torch.tensor([lower_S0]), high=torch.tensor([upper_S0]))
        self.dist_noise = BoxUniform(low=torch.tensor([lower_noise]), high=torch.tensor([upper_noise]))
        self.return_numpy = return_numpy
        
    def sample(self, sample_shape=torch.Size([])):
        
        abc     = self.dist_abs.sample(sample_shape)
        rest    = self.dist_rest.sample(sample_shape)
        S0      = self.dist_S0.sample(sample_shape)
        noise   = self.dist_noise.sample(sample_shape)
        
        if self.return_numpy:   
            params = np.hstack([abc,rest,S0,noise]) 
        else:
            params = torch.hstack([abc,rest,S0,noise])

        return params
        
    def log_prob(self, values):
        if self.return_numpy:
            values = torch.as_tensor(values)
        
        abc     = values[:,:3]
        rest    = values[:,3:-2]
        S0      = values[:,-2]
        noise   = values[:,-1]

        log_prob_abc  = self.dist_abs.log_prob(abc)
        log_prob_rest = self.dist_rest.log_prob(rest)
        log_prob_S0 = self.dist_S0.log_prob(S0)
        log_prob_noise = self.dist_noise.log_prob(noise)
        return log_prob_abc+log_prob_rest+log_prob_S0+log_prob_noise

def histogram_mode(data, bins=50):
    # Calculate the histogram
    counts, bin_edges = np.histogram(data, bins=bins)
    
    # Find the bin with the maximum count (highest frequency)
    max_bin_index = np.argmax(counts)
    
    # Calculate the mode as the midpoint of the bin with the highest count
    mode = (bin_edges[max_bin_index] + bin_edges[max_bin_index + 1]) / 2
    
    return mode


class DTIPriorDirec:
    def __init__(self, lower_abs : Tensor, upper_abs : Tensor, 
                       lower_rest: Tensor, upper_rest: Tensor,
                        return_numpy: bool = False):

        self.dist_abs = BoxUniform(low= lower_abs* torch.ones(3), high=upper_abs * torch.ones(3))
        self.dist_rest = BoxUniform(low=lower_rest * torch.ones(3), high=upper_rest *torch.ones(3))
        self.direction_choice = Categorical(probs=torch.ones(1, 5))
        self.return_numpy = return_numpy
        
    def sample(self, sample_shape=torch.Size([])):
        
        abc  = self.dist_abs.sample(sample_shape)
        rest = self.dist_rest.sample(sample_shape)
        direc = self.direction_choice.sample(sample_shape)
        
        if self.return_numpy:   
            params = np.hstack([abc,rest,direc]) 
        else:
            params = torch.hstack([abc,rest,direc])

        return params
        
    def log_prob(self, values):
        if self.return_numpy:
            values = torch.as_tensor(values)
        
        abc   = values[:,:3]
        rest  = values[:,3:-1]
        direc = values[:,-1]

        log_prob_abc  = self.dist_abs.log_prob(abc)
        log_prob_rest = self.dist_rest.log_prob(rest)
        log_prob_direc =  self.direction_choice.log_prob(direc)
        return log_prob_abc+log_prob_rest+log_prob_direc

### Errors ##################### ##################### ##################### ##################### ##################### #####################

def Errors(Guess,Truth,gtab,signal_true,signal_provided,S0Guess=200):
    # Eigenvalue error
    evals_guess_raw,evecs_guess = np.linalg.eigh(Guess)
    evals_guess = np.sort(evals_guess_raw)
    evals_true_raw,evecs_true = np.linalg.eigh(Truth)
    evals_true = np.sort(evals_true_raw)
    
    EigError = np.linalg.norm(evals_guess-evals_true)

    # Mean diffusivitiy
    mean_true = np.mean(evals_true)
    mean_guess = np.mean(evals_guess)
    MD = abs(mean_true-mean_guess)

    # Fractional Anisotropy
    FA_true  = FracAni(evals_true,mean_true)
    FA_guess = FracAni(evals_guess,mean_guess)
    FA = abs(FA_true-FA_guess)                                        

    # Frobenius error
    Frob =  np.linalg.norm(Guess-Truth, 'fro')

    # Signal error
    signal_guess = single_tensor(gtab, S0=S0Guess, evals=evals_guess_raw, evecs=evecs_guess)
    Err  = np.linalg.norm(signal_true-signal_guess)/len(signal_true)
    Corr = np.corrcoef(signal_true,signal_guess)[0,1]
    
    Err2  = np.linalg.norm(signal_provided-signal_guess[:len(signal_provided)])/len(signal_provided)
    Corr2 = np.corrcoef(signal_provided,signal_guess[:len(signal_provided)])[0,1]
    
    return MD,FA,EigError,Frob,Err,Corr,Err2,Corr2

def ErrorsMDFA(Guess,Truth):
    # Eigenvalue error
    evals_guess_raw,evecs_guess = np.linalg.eigh(Guess)
    evals_guess = np.sort(evals_guess_raw)
    evals_true_raw,evecs_true = np.linalg.eigh(Truth)
    evals_true = np.sort(evals_true_raw)

    # Mean diffusivitiy
    mean_true = np.mean(evals_true)
    mean_guess = np.mean(evals_guess)
    if(not mean_true == 0):
        MD = abs(mean_true-mean_guess)
    else:
        MD = abs(mean_true-mean_guess)

    # Fractional Anisotropy
    FA_true  = FracAni(evals_true,mean_true)
    FA_guess = FracAni(evals_guess,mean_guess)
    if(not FA_true == 0):
        FA = abs(FA_true-FA_guess)
    else:
        FA = abs(FA_true-FA_guess)
                                    
    
    return MD,FA
    
def PercsMDFA(Guess,Truth):
    # Eigenvalue error
    evals_guess_raw,evecs_guess = np.linalg.eigh(Guess)
    evals_guess = np.sort(evals_guess_raw)
    evals_true_raw,evecs_true = np.linalg.eigh(Truth)
    evals_true = np.sort(evals_true_raw)

    # Mean diffusivitiy
    mean_true = np.mean(evals_true)
    mean_guess = np.mean(evals_guess)
    if(not mean_true == 0):
        MD = abs(mean_true-mean_guess)/mean_true
    else:
        MD = abs(mean_true-mean_guess)/mean_true

    # Fractional Anisotropy
    FA_true  = FracAni(evals_true,mean_true)
    FA_guess = FracAni(evals_guess,mean_guess)
    if(not FA_true == 0):
        FA = abs(FA_true-FA_guess)/FA_true
    else:
        FA = abs(FA_true-FA_guess)/FA_true
                                    
    
    return MD,FA


def DKIErrors(GuessDT,GuessKT,TruthDT,TruthKT):
    guess = DKIMetrics(GuessDT,GuessKT,False)
    truth = DKIMetrics(TruthDT,TruthKT,False)

    #mk diff
    mk = abs(guess[0]-truth[0])
    ak = abs(guess[1]-truth[1])
    rk = abs(guess[2]-truth[2])
    mkt = abs(guess[3]-truth[3])
    kfa = abs(guess[4]-truth[4])

    return mk,ak,rk,mkt,kfa

def Percs(GuessDT,GuessKT,TruthDT,TruthKT):
    guess = DKIMetrics(GuessDT,GuessKT,False)
    truth = DKIMetrics(TruthDT,TruthKT,False)
    
    #mk diff
    mk = abs(guess[0]-truth[0])/abs(truth[0])
    ak = abs(guess[1]-truth[1])/abs(truth[1])
    rk = abs(guess[2]-truth[2])/abs(truth[2])
    mkt = abs(guess[3]-truth[3])/abs(truth[3])
    kfa = abs(guess[4]-truth[4])/abs(truth[4])
    
    return mk,ak,rk,mkt,kfa

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
                                   data_range=img1.max() - img1.min(), full=False)
                    else:
                        val = ssim(img1_patch, img2_patch,
                                   data_range=dat_range, full=False)
                    ssim_values.append(val)

    return np.nanmean(ssim_values)


def CombSignal_poisson(bvecs, bvals, Delta, delta, params):
    """
    Compute the combined diffusion signal in a fast, vectorized way.
    
    Parameters:
      bvecs  : (M,3) array of b-vectors.
      bvals  : (M,) array of b-values.
      Delta, delta : acquisition parameters (scalars)
      params : list/tuple of parameters:
          params[0] : fiber directions as an (N,2) array of spherical angles (theta, phi)
          params[1] : Dpar (scalar)
          params[2] : Dperp (scalar)
          params[3] : D (for hindered compartment; passed to vals_to_mat)
          params[4] : fiber fractions as an (N+1,) array 
                      (first element for hindered compartment, then one per fiber)
          params[5] : mean (scalar, for gamma distribution)
          params[6] : sig2 (scalar, for gamma distribution)
          params[7] : S0 (scalar)
    
    Returns:
      Signal : (M,) array of simulated signal values.
    """
    # Unpack parameters
    V_angles, Dpar, Dperp, D, fracs, mean, S0 = params

    # --- 1. Compute fiber unit vectors from spherical angles ---
    # Assume V_angles is an (N,2) array: each row is (theta, phi).
    theta_fibers = V_angles[:, 0]
    phi_fibers   = V_angles[:, 1]
    V_unit = np.column_stack((np.sin(theta_fibers) * np.cos(phi_fibers),
                              np.sin(theta_fibers) * np.sin(phi_fibers),
                              np.cos(theta_fibers)))  # shape: (N, 3)

    # --- 2. Compute angles between each fiber and each b-vector ---
    # Make sure bvecs is an array.
    bvecs = np.asarray(bvecs)  # shape: (M,3)
    M = bvecs.shape[0]
    N = V_unit.shape[0]

    # Precompute norms of bvecs (we assume fibers are unit length so no extra norm is needed)
    bvec_norms = np.linalg.norm(bvecs, axis=1)
    # Avoid division by zero:
    safe_bvec_norms = np.where(bvec_norms == 0, 1, bvec_norms)

    # Compute the dot products for each fiber with all bvecs:
    # This gives a (N, M) array where the (i,j) element = v_i dot bvec_j.
    dots = V_unit @ bvecs.T  # shape: (N, M)

    # Divide each column j by the norm of bvec j (broadcasting over fibers)
    cos_angles = dots / safe_bvec_norms  # shape: (N, M)
    cos_angles = np.clip(cos_angles, -1, 1)
    # Get the angles in [0,pi]
    Angs = np.arccos(cos_angles)
    # For bvecs that are zero (norm==0), force the angle to zero.
    if np.any(bvec_norms == 0):
        Angs[:, bvec_norms == 0] = 0
    # If an angle is greater than pi/2, use pi - angle.
    Angs = np.where(Angs > np.pi/2, np.pi - Angs, Angs)
    # In the original code the first measurement was forced to zero (presumably b = 0)
    Angs[:, 0] = 0

    # --- 3. Precompute the gamma-distributed weights for the integration over R ---
    # Gamma distribution parameters:
    lam = mean*10000
    # Define R values (50 points between 0.0001 and 0.005)
    R_vals = np.arange(0.0001, 0.01, 0.0001)  # 
    transR = (R_vals * 10000).astype(int)

    weights = (lam**transR) * np.exp(-lam) / np.array([math.factorial(r) for r in transR.astype(int)]).astype(np.double)
    weights /= np.sum(weights)

    # --- 4. Precompute the "sumterm" that appears in the restricted compartment ---
    # Here we use m=10 terms and assume that a global array Bessel_roots is available.
    m = 10
    br = Bessel_roots[:m]  # shape: (m,)
    br2 = br**2
    br6 = br**6
    # For each R in R_vals, compute the sumterm.
    # We need to broadcast over R and over the m terms.
    R2 = R_vals**2  # shape: (50,)
    # numerator: shape (50, m)
    num = (2 * Dperp * br2 * delta / R2[:, None] - 2 +
           2 * np.exp(-Dperp * br2 * delta / R2[:, None]) +
           2 * np.exp(-Dperp * br2 * Delta / R2[:, None]) -
           np.exp(-Dperp * br2 * (Delta - delta) / R2[:, None]) -
           np.exp(-Dperp * br2 * (Delta + delta) / R2[:, None]))
    # denominator: shape (50, m)
    den = (Dperp**2) * br6 * (br2 - 1) / (R_vals[:, None]**6)
    sumterm_R = np.sum(num / den, axis=1)  # shape: (50,)

    # --- 5. Compute the restricted compartment signal ---
    # For each fiber orientation i (i = 0...N-1) and for each measurement j (j = 0...M-1)
    # we need to compute:
    #   Restricted(b, theta, R) = exp(-b * (cos(theta)**2) * Dpar) *
    #                             exp(-2 * b * (sin(theta)**2) / ((Delta-delta/3)*delta**2) * sumterm)
    #
    # Notice that only the second exponential depends on R (via sumterm_R) and we need to integrate
    # over R with weights.
    #
    # Compute the part independent of R (base) and the factor x that multiplies sumterm_R.
    #
    # Angs has shape (N, M) (one row per fiber) and bvals is (M,).
    # (We assume that bvals is a 1D array; if not, cast it with np.asarray(bvals).)
    bvals = np.asarray(bvals)  # shape: (M,)
    base = np.exp(-bvals * (np.cos(Angs)**2) * Dpar)  # shape: (N, M)
    # Factor multiplying sumterm_R inside the second exponential.
    x = -2 * bvals * (np.sin(Angs)**2) / ((Delta - delta/3) * delta**2)  # shape: (N, M)
    # For each fiber orientation and measurement, we want to compute:
    #    f(i,j) = sum_{r=0}^{49} weights[r] * exp( x(i,j) * sumterm_R[r] )
    # We can compute the 3D array exp(x * sumterm_R) with shape (N, M, 50) and then contract out the last axis.
    exp_term = np.exp(x[..., None] * sumterm_R)  # shape: (N, M, 50)
    # Now take the weighted sum over the last axis (the R axis):
    restricted_integral = np.tensordot(exp_term, weights, axes=([2], [0]))  # shape: (N, M)
    # The restricted compartment signal for each fiber and measurement is then:
    Res = base * restricted_integral  # shape: (N, M)
    #
    # Finally, combine the fibers by weighting each fiber's contribution by its fraction.
    # The original code did: np.sum([f * R for f,R in zip(fracs[1:],Res)], axis=0)
    # That is equivalent to a dot product: (fracs[1:]) dot (each row of Res).
    restricted_signal = np.dot(fracs[1:], Res)  # shape: (M,)

    # --- 6. Compute the hindered compartment signal ---
    # Compute the diffusion tensor from D (using your vals_to_mat function).
    dh = vals_to_mat(D)
    # The hindered signal is given by:
    #    Hi = exp(-b * s)
    # where s = sum((bvec @ dh)*bvec, axis=1). Here bvecs is (M,3).
    s = np.sum((bvecs @ dh) * bvecs, axis=1)  # shape: (M,)
    hindered_signal = np.exp(-bvals * s)  # shape: (M,)

    # --- 7. Combine compartments and scale by S0 ---
    Signal = fracs[0] * hindered_signal + restricted_signal
    return S0 * Signal

def SpherAng(v_in):

    if v_in[2] < 0:
        v_in = -v_in  # Flip the vector to the top hemisphere

    x, y, z = v_in
    r = np.linalg.norm(v_in)
    if r == 0:
        # Degenerate vector, define angles however you like:
        return 0.0, 0.0
    
    # Polar angle in [0, pi]
    theta = np.arccos(z / r)
    
    # Azimuthal angle in (-pi, pi]
    phi = np.arctan2(y, x)
        
    return theta,phi