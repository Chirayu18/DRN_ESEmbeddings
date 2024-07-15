import torch
import numpy as np
from torch.nn.functional import softplus

sqrtPiOn2 = 1.25331413732
sqrt2 = 1.41421356237

minval = 1e-9
maxval = 1e9

def dscb_single(x, mu, sigma, alphaL, nL, alphaR, nR):
    t = (x-mu)/sigma
    
    fact1L = alphaL/nL
    fact2L = nL/alphaL - alphaL - t

    fact1R = alphaR/nR
    fact2R = nR/alphaR - alphaR + t

    if -alphaL <= t and alphaR >= t:
        result = torch.exp(-0.5*t*t)
    elif t < -alphaL:
        result = torch.exp(-0.5*alphaL*alphaL) * torch.pow(fact1L * fact2L, -nL)
    elif t > alphaR:
        result = torch.exp(-0.5*alphaR*alphaR) * torch.pow(fact1R * fact2R, -nR)
    else:
        print("UH OH")
        result = torch.zeros_like(x)

    return result

def naiive_vectorized(x, mu, sigma, alphaL, nL, alphaR, nR):
    result = torch.zeros_like(x)
    for i in range(len(x)):
        result[i] = dscb_single(x[i], mu[i], sigma[i], alphaL[i], nL[i], alphaR[i], nR[i])

    norm = double_crystalball_norm(mu, sigma, alphaL, nL, alphaR, nR)

    return result/norm

def smarter(x, mu, sigma, alphaL, nL, alphaR, nR):
    t = (x-mu)/sigma

    result = torch.empty_like(x)

    middle = torch.logical_and(-alphaL <= t, t <= alphaR)
    left = t < - alphaL
    right = alphaR < t

    tM = t[middle]
    result[middle] = torch.exp(-0.5*tM*tM)

    nLL = nL[left]
    tL = t[left]
    alphaLL = alphaL[left]
    fact1L = alphaLL/nLL
    fact2L = nLL/alphaLL - alphaLL - tL
    result[left] = torch.exp(-0.5*alphaLL*alphaLL) * torch.pow(fact1L * fact2L, -nLL)

    nRR = nR[right]
    tR = t[right]
    alphaRR = alphaR[right]
    fact1R = alphaRR/nRR
    fact2R = nRR/alphaRR - alphaRR + tR
    result[right] = torch.exp(-0.5*alphaRR*alphaRR) * torch.pow(fact1R * fact2R, -nRR)

    norm = double_crystalball_norm(mu, sigma, alphaL, nL, alphaR, nR)
    result = result/norm

    small = result < minval
    result[small] = minval

    return result

def double_crystalball_norm(mu, sigma, alphaL, nL, alphaR, nR):
    LN_top = torch.exp(-0.5*torch.square(alphaL))*nL
    LN_bottom = alphaL*(nL-1)

    RN_top = torch.exp(-0.5*torch.square(alphaR))*nR
    RN_bottom = alphaR*(nR-1)

    CN = sqrtPiOn2 * (torch.erf(alphaL/sqrt2) + torch.erf(alphaR/sqrt2))

    return (LN_top/LN_bottom + RN_top/RN_bottom + CN) * sigma

def dscb_semiparam(pred):
    mu = pred[:,0]
    sigma = softplus(pred[:,1])+1e-3
    alphaL = softplus(pred[:,2])+1e-3
    nL = softplus(pred[:,3]) + 1+1e-3
    alphaR = softplus(pred[:,4]) + 1e-3
    nR = softplus(pred[:,5]) + 1 + 1e-3

    return mu, sigma, alphaL, nL, alphaR, nR

def dscb_semiparam_sigmoid(pred, threshold, epsilon):
    mu = (torch.sigmoid(pred[:,0])  - 0.5)* 2 * threshold

    sigma = softplus(pred[:,1])+ epsilon
    alphaL = softplus(pred[:,2])+ epsilon
    nL = softplus(pred[:,3]) + 1+ epsilon
    alphaR = softplus(pred[:,4]) + epsilon
    nR = softplus(pred[:,5]) + 1 + epsilon

    return mu, sigma, alphaL, nL, alphaR, nR

def dscb_semiparam_sigmoid_minalpha(pred, threshold, epsilon, minalpha):
    mu = (torch.sigmoid(pred[:,0])  - 0.5)* 2 * threshold

    sigma = softplus(pred[:,1])+ epsilon
    alphaL = softplus(pred[:,2])+ epsilon + minalpha
    nL = softplus(pred[:,3]) + 1+ epsilon
    alphaR = softplus(pred[:,4]) + epsilon + minalpha
    nR = softplus(pred[:,5]) + 1 + epsilon

    return mu, sigma, alphaL, nL, alphaR, nR

def get_dscb_loss_l2(threshold, reg):
    logthresh = np.log(threshold)
    return lambda pred, target, weight=None : _dscb_loss_l2(pred, target, weight, logthresh, reg)

def _dscb_loss_l2(pred, target, weight= None, threshold = 2, reg = 1):
    batch_size = pred.size()[0]

    param = dscb_semiparam(pred)

    prob = smarter(target, *param)
    logprob = torch.log(prob)
    loss1 = -torch.sum(logprob)/batch_size

    correction = torch.abs(param[0])
    small = correction<threshold
    correction[small]=0
    loss2 = reg*torch.sum(correction*correction)

    return loss1+loss2

def dscb_loss(pred, target, weight=None):
    batch_size = pred.size()[0]

    param = dscb_semiparam(pred)

    prob = smarter(target, *param)
    
    logprob = torch.log(prob)
    loss = -torch.sum(logprob)/batch_size

    return loss

def get_dscb_loss_sigmoid(threshold, epsilon):
    logthresh = np.log(threshold)
    return lambda pred, target, weight=None : _dscb_loss_sigmoid(pred, target, weight, logthresh, epsilon)

def _dscb_loss_sigmoid(pred, target, weight=None, threshold = 2, epsilon = 1e-3):
    batch_size = pred.size()[0]

    param = dscb_semiparam_sigmoid(pred, threshold, epsilon)

    prob = smarter(target, *param)
    
    logprob = torch.log(prob)
    loss = -torch.sum(logprob)/batch_size

    return loss

def get_dscb_loss_sigmoid_minalpha(threshold, epsilon, minalpha):
    logthresh = np.log(threshold)
    return lambda pred, target, weight=None : _dscb_loss_sigmoid_minalpha(pred, target, weight, logthresh, epsilon, minalpha)

def _dscb_loss_sigmoid_minalpha(pred, target, weight=None, threshold = 2, epsilon = 1e-3, minalpha=1):
    batch_size = pred.size()[0]

    param = dscb_semiparam_sigmoid_minalpha(pred, threshold, epsilon, minalpha)

    prob = smarter(target, *param)
    
    logprob = torch.log(prob)
    loss = -torch.sum(logprob)/batch_size

    return loss

def ExpGaussExpNorm(x, mu, sigma, kL, kR):
    NormL = torch.exp(-0.5*kL*kL)/kL
    NormR = torch.exp(-0.5*kR*kR)/kR
    
    NormC = sqrtPiOn2 * (torch.erf(kL/sqrt2) + torch.erf(kR/sqrt2))

    return (NormL + NormR + NormC) * sigma

def ExpGaussExp(x, mu, sigma, kL, kR):
    t = (x-mu)/sigma

    result = torch.empty_like(x)

    middle = torch.logical_and(-kL <= t, t <= kR)
    left = t < -kL
    right = kR < t

    tM = t[middle]
    result[middle] = torch.exp(-0.5*tM*tM)

    tL = t[left]
    kLL = kL[left]
    result[left] = torch.exp( kLL * (kLL*0.5 + tL))

    tR = t[right]
    kRR = kR[right]
    result[right] = torch.exp(kRR * (0.5*kRR + tR))

    norm = ExpGaussExpNorm(x, mu, sigma, kL, kR)
    result = (result+minval)/(norm+minval)

    small = result < minval
    result[small] = minval

    large = result > maxval
    result[large] = maxval

    return result

def ExpGaussExp_semiparam(pred):
    mu = pred[:,0]
    sigma = softplus(pred[:,1]) + minval
    kL = softplus(pred[:,2]) + minval
    kR = softplus(pred[:,3]) + minval

    return mu, sigma, kL, kR

def ExpGaussExp_loss(pred, target, weight=None):
    batch_size = pred.size()[0]

    param = ExpGaussExp_semiparam(pred)

    prob = ExpGaussExp(target, *param)

    logprob = torch.log(prob)
    
    return -torch.sum(logprob)/batch_size
