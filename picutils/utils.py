import math
import torch
import numpy as np
from typing import Callable

from picutils.RecursiveWarper import make_multi_return_recursive_func, make_recursive_func

def decohints(decorator: Callable) -> Callable:
    return decorator

def angle2Rad(angle):
    return angle / 180 * math.pi
def rad2Angle(rad):
    return rad / math.pi * 180

def addOnesToVect(X: torch.Tensor):
    tmp1 = torch.ones(X.shape[0] + 1, X.shape[1], dtype=X.dtype, device=X.device)
    tmp1[:X.shape[0]] = X
    return tmp1

def addZerosToVect(X: torch.Tensor):
    tmp1 = torch.zeros(X.shape[0] + 1, X.shape[1], dtype=X.dtype, device=X.device)
    tmp1[:X.shape[0]] = X
    return tmp1

@make_multi_return_recursive_func
def batchSqueeze(X: torch.Tensor):
    assert len(X.shape) >= 3
    batch, h, w, *r = X.shape
    X = X.transpose(0, 1)
    X = X.reshape(h, batch * w, *r)
    return X, batch

@make_recursive_func
def batchUnsqueeze(X: torch.Tensor, batch):
    h, batch_x_w, *r = X.shape
    w = batch_x_w // batch
    X = X.reshape(h, batch, w, *r)
    return X.transpose(0, 1)

def picDefaultDeviceAndDtype(device=None, dtype=None, defaultDevice=None, defaultDtype=None):
    if defaultDevice is None:
        defaultDevice = 'cpu'
    if defaultDtype is None:
        defaultDtype = torch.float64

    if device is None:
        device = defaultDevice
    if dtype is None:
        dtype = defaultDtype
    return device, dtype

def quaternion2Mat(x, y, z, w, device=None, dtype=None):
    device, dtype = picDefaultDeviceAndDtype(device=device, dtype=dtype)

    return torch.tensor([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w], 
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w], 
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
    ], dtype=dtype, device=device)

def make_nograd_func(func: Callable) -> Callable:
    """Utilities to make function no gradient

    Args:
        func: input function

    Returns:
        no gradient function wrapper for input function
    """

    def wrapper(*f_args, **f_kwargs):
        with torch.no_grad():
            ret = func(*f_args, **f_kwargs)
        return ret

    return wrapper

@make_recursive_func
def tensor_to_device(X, device) -> torch.Tensor:
    if isinstance(X, torch.Tensor):
        return X.to(device)
    return X

@make_recursive_func
def tensor_to_cuda(X):
    if isinstance(X, torch.Tensor):
        return X.cuda()
    return X

@make_recursive_func
def tensor2item(vars) -> float:
    """Convert tensor to float"""
    if isinstance(vars, torch.Tensor):
        return vars.data.item()
    return vars
    
@make_recursive_func
def tensor2numpy(vars, copy=True) -> np.ndarray:
    """Convert tensor to numpy array"""
    if isinstance(vars, torch.Tensor):
        res = vars.detach().cpu().numpy()
        if copy:
            return res.copy()
        else:
            return res
    return vars