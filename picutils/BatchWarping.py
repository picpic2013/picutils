from typing import List, Tuple

import torch
from picutils.MyPerspectiveCamera import MyPerspectiveCamera
from picutils.MyGridSample import grid_sample as myEnhancedGridSample

def getWarppingLine(refCam: List[MyPerspectiveCamera], srcCams: List[List[MyPerspectiveCamera]], normalize=False) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    @return basePoint_src, direction_src [B x V x 3 x H x W]
    '''
    B = len(refCam)
    V = len(srcCams[0])
    H = refCam[0].imgH
    W = refCam[0].imgW
    device = refCam[0].posture.device
    dtype = refCam[0].posture.dtype

    # construct posture and K
    refPosture_inv = torch.stack([cam.posture_inv for cam in refCam]).unsqueeze(1).repeat(1, V, 1, 1)  # B x V x 4 x 4
    refK_inv = torch.eye(4, device=device, dtype=dtype).view(1, 4, 4).repeat(B, 1, 1)                  # B x 4 x 4
    refK_inv[:, :3, :3] = torch.stack([cam.k_inv for cam in refCam])
    refK_inv = refK_inv.unsqueeze(1).repeat(1, V, 1, 1)                                                # B x V x 4 x 4

    srcPosture = torch.stack([torch.stack([cam.posture for cam in camList]) for camList in srcCams])   # B x V x 4 x 4
    srcK = torch.eye(4, device=device, dtype=dtype).view(1, 1, 4, 4).repeat(B, V, 1, 1)                # B x V x 4 x 4
    srcK[:, :, :3, :3] = torch.stack([torch.stack([cam.k for cam in camList]) for camList in srcCams])

    if refPosture_inv.dtype != dtype:
        refPosture_inv = refPosture_inv.type(dtype)
    if refK_inv.dtype != dtype:
        refK_inv = refK_inv.type(dtype)
    if srcPosture.dtype != dtype:
        srcPosture = srcPosture.type(dtype)
    if srcK.dtype != dtype:
        srcK = srcK.type(dtype)

    KRRK = torch.bmm(refPosture_inv.view(B * V, 4, 4), refK_inv.view(B * V, 4, 4))
    KRRK = torch.bmm(srcPosture.view(B * V, 4, 4), KRRK)
    KRRK = torch.bmm(srcK.view(B * V, 4, 4), KRRK) # (B * V) x 4 x 4

    refGrid = refCam[0].uv_grid.unsqueeze(0).repeat(B, 1, 1, 1).view(B, 2, H * W) # B x 2 x H x W => B x 2 x (H * W)
    # (B * V) x 4 x (H * W) ,,  [ u, v, 1, 0 ]
    direction_ref = torch.zeros(B, 1, 4, H * W, device=device, dtype=dtype)
    direction_ref[:,:,:2,:] = refGrid
    direction_ref[:,:,2,:] = 1
    direction_ref = direction_ref.repeat(1, V, 1, 1).view(B * V, 4, H * W)

    # (B * V) x 4 x (H * W) ,,  [ 0, 0, 0, 1 ]
    basePoint_ref = torch.zeros(B, 1, 4, H * W, device=device, dtype=dtype)
    basePoint_ref[:,:,3,:] = 1
    basePoint_ref = basePoint_ref.repeat(1, V, 1, 1).view(B * V, 4, H * W)

    basePoint_src = torch.bmm(KRRK, basePoint_ref).view(B, V, 4, H, W)
    direction_src = torch.bmm(KRRK, direction_ref).view(B, V, 4, H, W)

    if normalize:
        HS = srcCams[0][0].imgH
        WS = srcCams[0][0].imgW
        normalize_base = torch.tensor([WS, HS], dtype=dtype, device=device) * 0.5
        normalize_base = normalize_base.view(1, 1, 2, 1, 1)
        basePoint_src[:,:,:2,:,:] = basePoint_src[:,:,:2,:,:] / normalize_base
        direction_src[:,:,:2,:,:] = direction_src[:,:,:2,:,:] / normalize_base

    return basePoint_src[:,:,:3,:,:], direction_src[:,:,:3,:,:]

def getWarppingGrid(refCam: List[MyPerspectiveCamera],  # len(refCam) : B
    srcCams: List[List[MyPerspectiveCamera]],           # len(srcCams) : B && len(srcCams[0]) : N
    refDep: torch.Tensor,                               # [ B x n_plane x H x W ] | [ B x n_plane ]
    srcImgs: torch.Tensor,                              # [ B x n_view x channel x Hsrc x Wsrc]
    eps:float=1e-8):
    '''
    @param refCam:  len(refCam) : B
    @param refCam:  len(srcCams) : B && len(srcCams[0]) : N
    @param refDep:  [ B x n_plane x H x W ] | [ B x n_plane ]
    @param srcImgs: [ B x n_view x channel x Hsrc x Wsrc]

    @returns warpped src grid (ref_hat) [ B, N, NP, H, W, 2 ]
    '''
    if len(refDep.shape) == 2:
        refDep = refDep.unsqueeze(2).unsqueeze(2)
        refDep = refDep.repeat(1, 1, refCam[0].imgH, refCam[0].imgW)

    basePoint_src, direction_src = getWarppingLine(refCam, srcCams, normalize=True)
    B, V, _, H, W = basePoint_src.shape
    D = refDep.size(1)

    grid = basePoint_src.view(B, V, 1, 3, H, W) + direction_src.view(B, V, 1, 3, H, W) * refDep.view(B, 1, D, 1, H, W) # BVD3HW
    grid = grid[:,:,:,:2,:,:] / (grid[:,:,:,2:,:,:] + eps)
    grid = grid.permute(0, 1, 2, 4, 5, 3)

    return grid.view(B, V, D, H, W, 2) - 1.


def batchWarping(
    refCam: List[MyPerspectiveCamera],        # len(refCam) : B
    srcCams: List[List[MyPerspectiveCamera]], # len(srcCams) : B && len(srcCams[0]) : N
    refDep: torch.Tensor,                     # [ B x n_plane x H x W ] | [ B x n_plane ]
    srcImgs: torch.Tensor,                    # [ B x n_view x channel x Hsrc x Wsrc]
    eps:float=1e-8, 
    mode:str='bilinear', 
    padding_mode:str='zeros', 
    align_corners:bool=False):
    '''
    @param refCam:  len(refCam) : B
    @param refCam:  len(srcCams) : B && len(srcCams[0]) : N
    @param refDep:  [ B x n_plane x H x W ] | [ B x n_plane ]
    @param srcImgs: [ B x n_view x channel x Hsrc x Wsrc]

    @returns warpped src img (ref_hat) [ B x N x n_plane x C x H x W ]
    '''

    grid = getWarppingGrid(refCam, srcCams, refDep, srcImgs, eps)
    B, N, NP, H, W, _ = grid.shape
    _, _, C, HS, WS = srcImgs.shape
    dtype = refDep.dtype

    grid = grid.view(B * N * NP, H, W, 2)
    if grid.dtype != dtype:
        grid = grid.type(dtype)

    # apply gird_sample
    warppedImg = torch.nn.functional.grid_sample(srcImgs.unsqueeze(2).expand(B, N, NP, C, HS, WS).reshape(B * N * NP, C, HS, WS), grid, mode, padding_mode, align_corners)
    
    return warppedImg.view(B, N, NP, C, H, W)

def enhancedBatchWarping(refCam: List[MyPerspectiveCamera], # len(refCam) : B
    srcCams: List[List[MyPerspectiveCamera]],               # len(srcCams) : B && len(srcCams[0]) : N
    refDep: torch.Tensor,                                   # [ B x n_plane x H x W ] | [ B x n_plane ]
    srcImgs: torch.Tensor,                                  # [ B x n_view x channel x Hsrc x Wsrc]
    eps:float=1e-8):
    '''
    @param refCam:  len(refCam) : B
    @param refCam:  len(srcCams) : B && len(srcCams[0]) : N
    @param refDep:  [ B x n_plane x H x W ] | [ B x n_plane ]
    @param srcImgs: [ B x n_view x channel x Hsrc x Wsrc]

    @returns warpped src img (ref_hat) [ B x N x n_plane x C x H x W ]
    '''

    grid = getWarppingGrid(refCam, srcCams, refDep, srcImgs, eps)
    B, N, NP, H, W, _ = grid.shape
    _, _, C, HS, WS = srcImgs.shape

    grid = grid.view(B * N * NP, H, W, 2)

    warppedImg = myEnhancedGridSample(srcImgs.unsqueeze(2).expand(B, N, NP, C, HS, WS).view(B * N * NP, C, HS, WS), grid)

    return warppedImg.view(B, N, NP, C, H, W)