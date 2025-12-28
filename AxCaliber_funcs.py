from DTI_funcs import *

from scipy.special import j0, jv
from scipy.optimize import bisect

Delta = [0.017, 0.035, 0.061]             # ms
delta = 0.007           # ms

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

def AxCaliber(bvecs, bvals, Delta, delta, params):
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

def AxCal_Errors(TrueSig,TrueParams,GuessParams,Delta,bvecs,bvals):

    Res = np.linalg.norm(residuals(GuessParams,TrueSig,bvecs,bvals,Delta))
    alpha_err = np.abs(GuessParams[11]-TrueParams[11])

    angle_err1 =  np.abs(GuessParams[0]-TrueParams[0])
    angle_err2 =  np.abs(GuessParams[1]-TrueParams[1])

    Dpar_err  = np.abs(TrueParams[2]-GuessParams[2])
    Dperp_err  = np.abs(TrueParams[3]-GuessParams[3])

    MD_guess = np.linalg.eigh(vals_to_mat(GuessParams[4:10]))[0].mean()
    MD_true = np.linalg.eigh(vals_to_mat(TrueParams[4:10]))[0].mean()

    FA_guess = FracAni(np.linalg.eigh(vals_to_mat(GuessParams[4:10]))[0],MD_guess)
    FA_true  = FracAni(np.linalg.eigh(vals_to_mat(TrueParams[4:10]))[0],MD_true)

    MD_err = np.abs(MD_guess-MD_true)
    FA_err = np.abs(FA_guess-FA_true)

    Frac_err  = np.abs(TrueParams[10]-GuessParams[10])

    return Res, alpha_err,angle_err1,angle_err2,Dpar_err,Dperp_err,MD_err,FA_err,Frac_err

def residuals(params,TrueSig,bvecs,bvals,Delta,EstS0 = True):
    if(EstS0):
        Signal = AxC_Simulator(params,bvecs,bvals,Delta,S0=params[-1])
    else:
        Signal = AxC_Simulator(params,bvecs,bvals,Delta)
    return TrueSig - Signal

def AxC_Simulator(params,bvecs,bvals,Delta,S0=200):
    new_params = [np.array([params[:2]]),params[2],params[3],params[4:10],[params[10],1-params[10]],params[11],S0]
    Sig = []
    for bve,bva,d in zip(bvecs,bvals,Delta):
        Sig.append(AxCaliber(bve,bva,d,delta,new_params))
    return np.hstack(Sig) 


def shell_time_stats(b_dw, Δ_dw, y_dw):
    """
    Compute mean and variance of y for each (b, Δ) combination
    using DW-only arrays.

    Parameters
    ----------
    b_dw  : (N,) array of DW b-values
    Δ_dw  : (N,) array of DW diffusion times
    y_dw  : (N,) array of DW log-attenuations

    Returns
    -------
    feats : 1D array of length 2 * (#unique b) * (#unique Δ)
            [mean_y(b1,Δ1), var_y(b1,Δ1), mean_y(b1,Δ2), var_y(b1,Δ2), ...]
    """
    b_dw = np.asarray(b_dw)
    Δ_dw = np.asarray(Δ_dw)
    y_dw = np.asarray(y_dw)

    shells = np.unique(b_dw)
    times  = np.unique(Δ_dw)

    feats = []
    for b in shells:
        for Δ in times:
            m = (b_dw == b) & (Δ_dw == Δ)
            if np.any(m):
                y_sub = y_dw[m]
                feats.append(y_sub.mean())
                feats.append(y_sub.var())
            else:
                feats.append(0.0)
                feats.append(0.0)
    return np.array(feats, dtype=float)
    
def AxcaliberFeatures(bvecs, bvals, Deltas, Signal):
    """
    Feature construction for AxCaliber-like data, analogous to DTIFeatures/DKIFeatures.

    Parameters
    ----------
    bvecs  : (M, 3) array
        Gradient directions.
    bvals  : (M,) array
        b-values corresponding to each gradient.
    Deltas : (M,) array
        Diffusion times (Delta) for each measurement.
    Signal : (M,) array
        Measured signal.

    Returns
    -------
    feats : (1 + 18 + 18*18,) array
        Concatenated feature vector:
          [ S0_hat,
            SigFeat (18,),
            AcqFeat (18*18,) ]
    """
    bvecs  = np.asarray(bvecs, dtype=float)
    bvals  = np.asarray(bvals, dtype=float)/1000
    Deltas = np.asarray(Deltas, dtype=float)
    Signal = np.asarray(Signal, dtype=float)

    if bvecs.shape[0] != bvals.shape[0] or bvals.shape[0] != Deltas.shape[0] \
       or Signal.shape[0] != bvals.shape[0]:
        raise ValueError("bvecs, bvals, Deltas, and Signal must all have the same length (M).")

    # ---- 1. Separate b=0 and DW ----
    S0_mask = (bvals == 0)
    dw_mask = (bvals > 0)

    if not np.any(S0_mask):
        raise ValueError("At least one b=0 measurement is required to estimate S0.")
    if np.sum(dw_mask) < 6:
        raise ValueError("At least 6 DW measurements are recommended for this feature mapping.")

    S0_hat = np.mean(Signal[S0_mask])

    S_dw   = Signal[dw_mask]
    b_dw   = bvals[dw_mask]
    g_dw   = bvecs[dw_mask, :]
    Δ_dw   = Deltas[dw_mask]

    # Response variable y = -ln(S / S0)
    y = -np.log(S_dw / S0_hat)       # shape [N]
    N = y.size

    g1, g2, g3 = g_dw[:, 0], g_dw[:, 1], g_dw[:, 2]
    sqrtb = np.sqrt(b_dw)

    # ---- 2. Base diffusion-like design (6 terms, as in DTI) ----
    # Each term is effectively ~ b * g_i g_j
    # We construct it using the same pattern as your DTIFeatures.
    X_base = np.vstack([
        (sqrtb * g1) * (sqrtb * g1),         # b * g1^2
        2 * (sqrtb * g1) * (sqrtb * g2),     # 2 b * g1 g2
        (sqrtb * g2) * (sqrtb * g2),         # b * g2^2
        2 * (sqrtb * g1) * (sqrtb * g3),     # 2 b * g1 g3
        2 * (sqrtb * g2) * (sqrtb * g3),     # 2 b * g2 g3
        (sqrtb * g3) * (sqrtb * g3)          # b * g3^2
    ])  # shape [6, N]

    # ---- 3. Time basis functions for Delta ----
    # Use a simple polynomial basis in Delta: [1, Δ, Δ^2]
    # Then modulate the 6 diffusion-like terms with each time basis.
    T0 = np.ones_like(Δ_dw)       # Δ^0
    T1 = Δ_dw                     # Δ^1
    T2 = Δ_dw**2                  # Δ^2

    # Each gives a 6×N block when multiplied with X_base.
    X_Δ0 = X_base                 # 6 × N
    X_Δ1 = X_base * T1            # 6 × N (broadcast over columns)
    X_Δ2 = X_base * T2            # 6 × N

    # Stack into a single design matrix: 18 × N
    X = np.vstack([X_Δ0, X_Δ1, X_Δ2])  # shape [18, N]

    # ---- 4. Acquisition and signal features ----
    # Acquisition features: design self-product
    XtX = X_base @ X_base.T                 # [18, 18]
    AcqFeat = (XtX / N).ravel()   # length 18*18

    # Signal features: design–response product
    SigFeat = (X @ y) / N         # length 18
    

    ShellTimeFeat = shell_time_stats(b_dw, Δ_dw, y)

    # ---- 5. Concatenate ----
    return np.hstack([S0_hat, SigFeat, AcqFeat,ShellTimeFeat])


def Par_frac(i,j,Mat):
    MD = np.linalg.eigh(vals_to_mat(Mat[i,j]))[0].mean()

    FA = FracAni(np.linalg.eigh(vals_to_mat(Mat[i,j]))[0],MD)
    return i, j, [FA,MD]