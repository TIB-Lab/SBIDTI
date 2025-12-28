from Imports import *
import dipy.reconst.dti as dti

def random_diffusion_tensor(MD, FA):
    # Step 1: compute variance term
    X = (3 * (FA**2) * (MD**2)) / (1.5 - FA**2)

    # Step 2: sample shape angle
    theta = np.random.uniform(0, 2*np.pi)
    a = np.sqrt(X/6) * np.cos(theta)
    b = np.sqrt(X/2) * np.sin(theta)

    # Step 3: eigenvalues
    l1 = MD + a + b
    l2 = MD + a - b
    l3 = MD - 2*a
    lambdas = np.array([l1, l2, l3])

    # Ensure positivity
    if np.any(lambdas <= 0):
        return random_diffusion_tensor(MD, FA)

    # Step 4: random orientation (QR decomposition of random Gaussian matrix)
    Q = random_SO3()
    
    # Step 5: construct tensor
    D = Q @ np.diag(lambdas) @ Q.T
    return D

# Generating diffusion tensors
def random_SO3():
    M = np.random.normal(size=(3,3))
    Q, _ = np.linalg.qr(M)
    if np.linalg.det(Q) < 0:
        Q[:,0] *= -1
    return Q


# Compressing signal into featuers
def DTIFeatures(bvecs,bvals,Signal):
    
    S_0hat = np.mean(Signal[bvals==0])
    S_dw = Signal[bvals>0]
    LogSig = -np.log(S_dw/S_0hat)
    N = len(S_dw)
    b_dw = bvecs[bvals>0]
    bvals_dw = bvals[bvals>0]
    g1,g2,g3 = b_dw[:,0],b_dw[:,1],b_dw[:,2]
    
    sqrtb = np.sqrt(bvals_dw)
    X = np.vstack([
        (sqrtb * g1) * (sqrtb * g1),
        2 * (sqrtb * g1) * (sqrtb * g2),
        (sqrtb * g2) * (sqrtb * g2),
        2 * (sqrtb * g1) * (sqrtb * g3),
        2 * (sqrtb * g2) * (sqrtb * g3),
        (sqrtb * g3) * (sqrtb * g3)
    ])  # shape [6, N]

    # now form X^T X (sum over directions)
    XtX = X @ X.T     # shape [6, 6]

    # flatten (or take upper triangle)
    AcqFeat = XtX.flatten() / N
    SigFeat = X @ LogSig / N 
    
    return np.hstack([S_0hat,SigFeat,AcqFeat])

# MD & FA
def MD_FA(mat):
    mat = clip_negative_eigenvalues(mat)
    Eigs = np.linalg.eigh(mat)[0]
    MD = np.mean(Eigs)
    FA = FracAni(Eigs,MD)
    return MD,FA

def CustomSimulator(Mat,gtab,S0,snr=None):
    evals,evecs = np.linalg.eigh(Mat)
    signal = single_tensor(gtab, S0=S0, evals=evals, evecs=evecs)
    if(snr is None):
        return signal
    else:
        return AddNoise(signal,S0,snr)