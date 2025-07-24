#utils.py
import numpy as np
from config import *

def ssim_func(S):
    return ALPHA * S ** 2 + BETA * S + GAMMA

def satisfaction_func(S, delta):
    return delta * ssim_func(S)

def gen_deltas(num_vmu):
    np.random.seed(SEED)
    return np.random.normal(DELTA_MEAN, DELTA_STD, num_vmu)

def clamp(x, xmin, xmax):
    return np.clip(x, xmin, xmax)
