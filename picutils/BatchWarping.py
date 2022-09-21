from typing import List, Tuple

import torch
from picutils.MyPerspectiveCamera import MyPerspectiveCamera
from picutils.MyGridSample import grid_sample as myEnhancedGridSample

def getWarppingLine_raw(ref_R_inv: torch.Tensor, ref_K_inv: torch.Tensor, src_R: torch.Tensor, src_K: torch.Tensor, grid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    @param ref_R: [B x 4 x 4]
    @param ref_K: [B x 3 x 3]
    @param src_R: [B x V x 4 x 4]
    @param src_K: [B x V x 3 x 3]
    @param grid:  [B x 2 x ?] | [2 x ?] (u, v)

    @return basePoint_src, direction_src [B x V x 3 x ?]
    '''

    B, V, _, _ = src_R.shape
    if len(grid.shape) == 2:
        grid = grid.unsqueeze(0).repeat(B, 1, 1)
    HW = grid.size(2)
    device = src_R.device
    dtype = src_R.dtype

    if grid.dtype != dtype:
        grid = grid.type(dtype)
    
    refPosture_inv = ref_R_inv.unsqueeze(1).repeat(1, V, 1, 1)                          # B x V x 4 x 4
    refK_inv = torch.eye(4, device=device, dtype=dtype).view(1, 4, 4).repeat(B, 1, 1)
    refK_inv[:, :3, :3] = ref_K_inv
    refK_inv = refK_inv.unsqueeze(1).repeat(1, V, 1, 1)                                 # B x V x 4 x 4

    srcPosture = src_R
    srcK = torch.eye(4, device=device, dtype=dtype).view(1, 1, 4, 4).repeat(B, V, 1, 1)
    srcK[:, :, :3, :3] = src_K                                                          # B x V x 4 x 4

    KRRK = torch.bmm(refPosture_inv.view(B * V, 4, 4), refK_inv.view(B * V, 4, 4))
    KRRK = torch.bmm(srcPosture.view(B * V, 4, 4), KRRK)
    KRRK = torch.bmm(srcK.view(B * V, 4, 4), KRRK) # (B * V) x 4 x 4

    refGrid = grid                                                                      # B x 2 x (H * W)
    # (B * V) x 4 x (H * W) ,,  [ u, v, 1, 0 ]
    direction_ref = torch.zeros(B, 1, 4, HW, device=device, dtype=dtype)
    direction_ref[:,0,:2,:] = refGrid
    direction_ref[:,:,2,:] = 1
    direction_ref = direction_ref.repeat(1, V, 1, 1).view(B * V, 4, HW)

    # (B * V) x 4 x (H * W) ,,  [ 0, 0, 0, 1 ]
    basePoint_ref = torch.zeros(B, 1, 4, HW, device=device, dtype=dtype)
    basePoint_ref[:,:,3,:] = 1
    basePoint_ref = basePoint_ref.repeat(1, V, 1, 1).view(B * V, 4, HW)

    basePoint_src = torch.bmm(KRRK, basePoint_ref).view(B, V, 4, HW)
    direction_src = torch.bmm(KRRK, direction_ref).view(B, V, 4, HW)

    return basePoint_src[:,:,:3,:], direction_src[:,:,:3,:]

def getWarppingLine_multi_view_raw(R_inv: torch.Tensor, K_inv: torch.Tensor, R: torch.Tensor, K: torch.Tensor, grid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    @param ref_R: [B x V x 4 x 4]
    @param ref_K: [B x V x 3 x 3]
    @param src_R: [B x V x 4 x 4]
    @param src_K: [B x V x 3 x 3]
    @param grid:  [B x 2 x ?] | [2 x ?] (u, v)

    @return basePoint_src, direction_src [B x V x V x 3 x ?]
    '''
    device = R_inv.device
    dtype = R_inv.dtype
    B, V, _, _ = R_inv.shape
    if len(grid.shape) == 2:
        grid = grid.unsqueeze(0).repeat(B, 1, 1)
    if grid.dtype != dtype:
        grid = grid.type(dtype)
    HW = grid.size(2)

    posture = R               # B x V x 4 x 4
    posture_inv = R_inv       # B x V x 4 x 4
    k = torch.eye(4, device=device, dtype=dtype).view(1, 1, 4, 4).repeat(B, V, 1, 1)     # B x V x 4 x 4
    k_inv = torch.eye(4, device=device, dtype=dtype).view(1, 1, 4, 4).repeat(B, V, 1, 1) # B x V x 4 x 4
    k[:,:,:3,:3] = K
    k_inv[:,:,:3,:3] = K_inv

    posture = posture.unsqueeze(1).repeat(1, V, 1, 1, 1)         # B x V x V x 4 x 4
    posture_inv = posture_inv.unsqueeze(1).repeat(1, V, 1, 1, 1) # B x V x V x 4 x 4
    k = k.unsqueeze(1).repeat(1, V, 1, 1, 1)                     # B x V x V x 4 x 4
    k_inv = k_inv.unsqueeze(1).repeat(1, V, 1, 1, 1)             # B x V x V x 4 x 4

    # construct line cords
    # (BVV) x 4 x 4
    KRRK = torch.bmm(
        posture_inv.permute(0, 2, 1, 3, 4).reshape(B * V * V, 4, 4), 
        k_inv.permute(0, 2, 1, 3, 4).reshape(B * V * V, 4, 4)
    )
    KRRK = torch.bmm(posture.view(B * V * V, 4, 4), KRRK)
    KRRK = torch.bmm(k.view(B * V * V, 4, 4), KRRK)

    refGrid = grid.unsqueeze(0).repeat(V * V, 1, 1, 1).view(B * V * V, 2, HW) # BVV x 2 x H x W => BVV x 2 x (H * W)
    # (BVV) x 4 x (HW) ,,  [ u, v, 1, 0 ]
    direction_ref = torch.zeros(B * V * V, 4, HW, device=device, dtype=dtype)
    direction_ref[:,:2,:] = refGrid
    direction_ref[:,2,:] = 1

    # (BVV) x 4 x (HW) ,,  [ 0, 0, 0, 1 ]
    basePoint_ref = torch.zeros(B * V * V, 4, HW, device=device, dtype=dtype)
    basePoint_ref[:,3,:] = 1

    basePoint_src = torch.bmm(KRRK, basePoint_ref).view(B, V, V, 4, HW)
    direction_src = torch.bmm(KRRK, direction_ref).view(B, V, V, 4, HW)

    return basePoint_src[:,:,:,:3,:], direction_src[:,:,:,:3,:]


def getWarppingLine(refCam: List[MyPerspectiveCamera], srcCams: List[List[MyPerspectiveCamera]], normalize=True) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    @param normalize: if true, return value /= 0.5([WS, HS])
    @return basePoint_src, direction_src [B x V x 3 x H x W] 
    (if normalize, basePoint_src and direction_src are * / ([HS-1 HW-1]*0.5))
    '''
    B = len(refCam)
    V = len(srcCams[0])
    H = refCam[0].imgH
    W = refCam[0].imgW
    device = refCam[0].posture.device
    dtype = refCam[0].posture.dtype

    basePoint_src, direction_src = getWarppingLine_raw(
        torch.stack([cam.posture_inv for cam in refCam]), 
        torch.stack([cam.k_inv for cam in refCam]), 
        torch.stack([torch.stack([cam.posture for cam in camList]) for camList in srcCams]), 
        torch.stack([torch.stack([cam.k for cam in camList]) for camList in srcCams]), 
        refCam[0].uv_grid.view(2, H * W)
    )

    basePoint_src = basePoint_src.view(B, V, 3, H, W)
    direction_src = direction_src.view(B, V, 3, H, W)

    if normalize:
        HS = srcCams[0][0].imgH
        WS = srcCams[0][0].imgW

        basePoint_src[:,:,0,:,:] = basePoint_src[:,:,0,:,:] / ((WS - 1) * 0.5)
        basePoint_src[:,:,1,:,:] = basePoint_src[:,:,1,:,:] / ((HS - 1) * 0.5)
        direction_src[:,:,0,:,:] = direction_src[:,:,0,:,:] / ((WS - 1) * 0.5)
        direction_src[:,:,1,:,:] = direction_src[:,:,1,:,:] / ((HS - 1) * 0.5)

    return basePoint_src, direction_src

def getWarppingLine_multi_view(cams: List[List[MyPerspectiveCamera]], normalize=True) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    @param normalize: if true, return value /= 0.5([WS, HS])
    @return basePoint_src, direction_src [B x V x V x 3 x H x W]
    (if normalize, basePoint_src and direction_src are * / ([HS-1 HW-1]*0.5))
    '''
    B = len(cams)
    V = len(cams[0])
    H = cams[0][0].imgH
    W = cams[0][0].imgW
    device = cams[0][0].posture.device
    dtype = cams[0][0].posture.dtype

    basePoint_src, direction_src = getWarppingLine_multi_view_raw(
        torch.stack([torch.stack([cam.posture_inv for cam in camList]) for camList in cams]), 
        torch.stack([torch.stack([cam.k_inv for cam in camList]) for camList in cams]), 
        torch.stack([torch.stack([cam.posture for cam in camList]) for camList in cams]), 
        torch.stack([torch.stack([cam.k for cam in camList]) for camList in cams]), 
        cams[0][0].uv_grid.view(2, H * W)
    )

    basePoint_src = basePoint_src.view(B, V, V, 3, H, W)
    direction_src = direction_src.view(B, V, V, 3, H, W)

    if normalize:
        HS = cams[0][0].imgH
        WS = cams[0][0].imgW

        basePoint_src[:,:,:,0,:,:] = basePoint_src[:,:,:,0,:,:] / ((WS - 1) * 0.5)
        basePoint_src[:,:,:,1,:,:] = basePoint_src[:,:,:,1,:,:] / ((HS - 1) * 0.5)
        direction_src[:,:,:,0,:,:] = direction_src[:,:,:,0,:,:] / ((WS - 1) * 0.5)
        direction_src[:,:,:,1,:,:] = direction_src[:,:,:,1,:,:] / ((HS - 1) * 0.5)

    return basePoint_src, direction_src


def getWarppingGrid(refCam: List[MyPerspectiveCamera],  # len(refCam) : B
    srcCams: List[List[MyPerspectiveCamera]],           # len(srcCams) : B && len(srcCams[0]) : N
    refDep: torch.Tensor,                               # [ B x n_plane x H x W ] | [ B x n_plane ]
    eps:float=1e-8, 
    lineParam=None):
    '''
    @param refCam:  len(refCam) : B
    @param srcCams: len(srcCams) : B && len(srcCams[0]) : N
    @param refDep:  [ B x n_plane x H x W ] | [ B x n_plane ]
    @param srcImgs: [ B x n_view x channel x Hsrc x Wsrc]

    @returns warpped src grid (ref_hat) [ B, N, NP, H, W, 2 ]
    '''
    if len(refDep.shape) == 2:
        refDep = refDep.unsqueeze(2).unsqueeze(2)
        refDep = refDep.repeat(1, 1, refCam[0].imgH, refCam[0].imgW)

    if lineParam is None:
        basePoint_src, direction_src = getWarppingLine(refCam, srcCams, normalize=True)
    else:
        basePoint_src, direction_src = lineParam

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
    align_corners:bool=False, 
    lineParam=None):
    '''
    @param refCam:  len(refCam) : B
    @param refCam:  len(srcCams) : B && len(srcCams[0]) : N
    @param refDep:  [ B x n_plane x H x W ] | [ B x n_plane ]
    @param srcImgs: [ B x n_view x channel x Hsrc x Wsrc]

    @returns warpped src img (ref_hat) [ B x N x n_plane x C x H x W ]
    '''

    grid = getWarppingGrid(refCam, srcCams, refDep, eps, lineParam)
    B, N, NP, H, W, _ = grid.shape
    _, _, C, HS, WS = srcImgs.shape
    dtype = refDep.dtype

    grid = grid.view(B * N * NP, H, W, 2)
    if grid.dtype != dtype:
        grid = grid.type(dtype)

    # apply gird_sample
    warppedImg = torch.nn.functional.grid_sample(srcImgs.unsqueeze(2).expand(-1, -1, NP, -1, -1, -1).reshape(B * N * NP, C, HS, WS), grid, mode, padding_mode, align_corners)
    
    return warppedImg.view(B, N, NP, C, H, W)

def enhancedBatchWarping(refCam: List[MyPerspectiveCamera], # len(refCam) : B
    srcCams: List[List[MyPerspectiveCamera]],               # len(srcCams) : B && len(srcCams[0]) : N
    refDep: torch.Tensor,                                   # [ B x n_plane x H x W ] | [ B x n_plane ]
    srcImgs: torch.Tensor,                                  # [ B x n_view x channel x Hsrc x Wsrc]
    eps:float=1e-8, 
    lineParam=None):
    '''
    @param refCam:  len(refCam) : B
    @param refCam:  len(srcCams) : B && len(srcCams[0]) : N
    @param refDep:  [ B x n_plane x H x W ] | [ B x n_plane ]
    @param srcImgs: [ B x n_view x channel x Hsrc x Wsrc]

    @returns warpped src img (ref_hat) [ B x N x n_plane x C x H x W ]
    '''

    grid = getWarppingGrid(refCam, srcCams, refDep, eps, lineParam)
    B, N, NP, H, W, _ = grid.shape
    _, _, C, HS, WS = srcImgs.shape

    grid = grid.view(B * N * NP, H, W, 2)

    warppedImg = myEnhancedGridSample(srcImgs.unsqueeze(2).expand(-1, -1, NP, -1, -1, -1).reshape(B * N * NP, C, HS, WS), grid)

    return warppedImg.view(B, N, NP, C, H, W)