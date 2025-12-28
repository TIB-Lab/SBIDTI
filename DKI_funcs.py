from Imports import *
import dipy.reconst.dki as dki
from scipy.stats import lognorm,norm


lower_abs,upper_abs = -0.07,0.07
lower_rest,upper_rest = -0.015,0.015
lower_S0 = 25
upper_S0 = 2000

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

    kfa = kurtosis_fractional_anisotropy2(params)

    return mk,ak,rk,mkt,kfa

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
    
def kurtosis_fractional_anisotropy2(dki_params):
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


def DKIFeatures(bvecs, bvals, Signal):
    """
    Compress diffusion–kurtosis data into sufficient statistics for
    linear least-squares fitting of the diffusion and kurtosis tensors.

    Model:
        y = -ln(S / S0) = A * theta
    where:
        theta = [Dxx, Dxy, Dyy, Dxz, Dyz, Dzz,
                 W1111, W2222, W3333,
                 W1122, W1133, W2233,
                 W1112, W1113, W1222, W2223, W1333, W2333,
                 W1123, W1223, W1233]  (21 parameters)

    The returned feature vector is:
        [ S0_hat,
          (X @ y / N),        # 21 elements: "signal features"
          (X @ X.T / N).ravel() ]   # 21×21 acquisition features
    """

    bvecs = np.asarray(bvecs, dtype=float)
    bvals = np.asarray(bvals, dtype=float)
    Signal = np.asarray(Signal, dtype=float)

    # Separate b=0 and diffusion-weighted
    S0_mask = (bvals == 0)
    dw_mask = (bvals > 0)

    if not np.any(S0_mask):
        raise ValueError("At least one b=0 measurement is required to estimate S0.")
    if np.sum(dw_mask) < 21:
        raise ValueError("At least 21 DW measurements are recommended for DKI (6 D + 15 W parameters).")

    S0_hat = np.mean(Signal[S0_mask])

    S_dw   = Signal[dw_mask]
    b_dw   = bvals[dw_mask]
    g_dw   = bvecs[dw_mask, :]

    # Response variable
    y = -np.log(S_dw / S0_hat)  # shape [N]

    g1, g2, g3 = g_dw[:, 0], g_dw[:, 1], g_dw[:, 2]
    b  = b_dw
    b2 = b_dw**2

    # -------- Diffusion design (6 terms) --------
    # Using the same convention as your DTI code:
    #   [b*g1**2, 2b*g1*g2, b*g2**2, 2b*g1*g3, 2b*g2*g3, b*g3**2]
    D1 = b * g1 * g1
    D2 = 2 * b * g1 * g2
    D3 = b * g2 * g2
    D4 = 2 * b * g1 * g3
    D5 = 2 * b * g2 * g3
    D6 = b * g3 * g3

    # -------- Kurtosis design (15 terms) --------
    # Standard 4th-order kurtosis tensor basis (Tabesh-like ordering):
    #
    #   1:  W1111 ~ g1^4
    #   2:  W2222 ~ g2^4
    #   3:  W3333 ~ g3^4
    #   4:  W1122 ~ g1^2 g2^2
    #   5:  W1133 ~ g1^2 g3^2
    #   6:  W2233 ~ g2^2 g3^2
    #   7:  W1112 ~ g1^3 g2
    #   8:  W1113 ~ g1^3 g3
    #   9:  W1222 ~ g1 g2^3
    #   10: W2223 ~ g2^3 g3
    #   11: W1333 ~ g1 g3^3
    #   12: W2333 ~ g2 g3^3
    #   13: W1123 ~ g1^2 g2 g3
    #   14: W1223 ~ g1 g2^2 g3
    #   15: W1233 ~ g1 g2 g3^2
    #
    # The multiplicity factors (4, 6, 12) account for symmetric permutations
    # of indices in the 4th-order tensor, and we include the 1/6 factor from
    # the DKI signal model.
    c = 1.0 / 6.0

    W1  = c * b2 * (g1**4)                  # W1111
    W2  = c * b2 * (g2**4)                  # W2222
    W3  = c * b2 * (g3**4)                  # W3333

    W10  = c * b2 * (6 * g1**2 * g2**2)      # W1122
    W11  = c * b2 * (6 * g1**2 * g3**2)      # W1133
    W12  = c * b2 * (6 * g2**2 * g3**2)      # W2233

    W4  = c * b2 * (4 * g1**3 * g2)         # W1112
    W5  = c * b2 * (4 * g1**3 * g3)         # W1113
    W6  = c * b2 * (4 * g1 * g2**3)         # W1222
    W7 = c * b2 * (4 * g2**3 * g3)         # W2223
    W8 = c * b2 * (4 * g1 * g3**3)         # W1333
    W9 = c * b2 * (4 * g2 * g3**3)         # W2333

    W13 = c * b2 * (12 * g1**2 * g2 * g3)   # W1123
    W14 = c * b2 * (12 * g1 * g2**2 * g3)   # W1223
    W15 = c * b2 * (12 * g1 * g2 * g3**2)   # W1233

    # Stack all regressors into X: shape [21, N]
    X = np.vstack([
        D1, D2, D3, D4, D5, D6,
        W1, W2, W3, W4, W5, W6,
        W7, W8, W9, W10, W11, W12,
        W13, W14, W15
    ])

    N = X.shape[1]

    # Acquisition features (design self-product)
    XtX = X @ X.T       # [21, 21]

    # Signal features (design–response product)
    SigFeat = X @ y     # [21]

    # Normalize by number of DW measurements, as in your DTI code
    AcqFeat = (XtX / N).ravel()
    SigFeat = SigFeat / N

    return np.hstack([S0_hat, SigFeat, AcqFeat])

def CustomDKISimulator(dt,kt,gtab,S0,snr=None):
    if(dt.ndim == 1):
        dt = vals_to_mat(dt)
    evals,evecs = np.linalg.eigh(dt)
    params = np.concatenate([evals,np.hstack(evecs),kt])
    signal = dki.dki_prediction(params,gtab,S0=S0)
    if(snr is None):
        return signal
    else:
        return AddNoise(signal,S0,snr)