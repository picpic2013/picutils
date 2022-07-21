from typing import List

import torch
from picutils.MyPerspectiveCamera import MyPerspectiveCamera
from picutils.MyGridSample import grid_sample as myEnhancedGridSample

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

    # read info
    B, N, C, HS, WS = srcImgs.shape
    _, NP, H, W = refDep.shape
    device = refDep.device
    dtype = refDep.dtype
    
    # construct posture and K
    refPosture_inv = torch.stack([cam.posture_inv for cam in refCam])  # B x 4 x 4
    refK_inv = torch.stack([cam.k_inv for cam in refCam])              # B x 3 x 3

    srcPosture = torch.stack([torch.stack([cam.posture for cam in camList]) for camList in srcCams]) # B x N x 4 x 4
    srcK = torch.stack([torch.stack([cam.k for cam in camList]) for camList in srcCams])             # B x N x 3 x 3

    srcPosture = srcPosture.reshape(B, N, 1, 4, 4) # B x N x 1 x 4 x 4
    srcK = srcK.reshape(B, N, 1, 3, 3)             # B x N x 1 x 3 x 3

    if refPosture_inv.dtype != dtype:
        refPosture_inv = refPosture_inv.type(dtype)
    if refK_inv.dtype != dtype:
        refK_inv = refK_inv.type(dtype)
    if srcPosture.dtype != dtype:
        srcPosture = srcPosture.type(dtype)
    if srcK.dtype != dtype:
        srcK = srcK.type(dtype)

    # build torch one tensor
    oneTensor = torch.ones(B, 1, H * W, dtype=dtype, device=device)             # B x 1 x (W * H)
    zeroTensor = torch.zeros(B, 1, H * W, dtype=dtype, device=device)           # B x 1 x (W * H)
    tensor0001 = torch.tensor([0., 0., 0., 1.], dtype=dtype, device=device)     # 4
    normalize_base = torch.tensor([WS, HS], dtype=dtype, device=device) * 0.5   # 2

    # calculate world line
    refGrid = refCam[0].uv_grid.unsqueeze(0).repeat(B, 1, 1, 1).view(B, 2, H * W)        # B x 2 x H x W => B x 2 x (H * W)
    ref_Dir_XYZ_Picture = torch.cat([refGrid, oneTensor], dim=1)            # B x 3 x (H * W)
    ref_Dir_XYZ_Camera = torch.bmm(refK_inv, ref_Dir_XYZ_Picture)           # B x 3 x (H * W)

    dstCameraAnchorPoint4 = torch.bmm(refPosture_inv, tensor0001.unsqueeze(0).repeat(B, 1).unsqueeze(2)) # B x 4 x 1
    directionVectorU2Camera4 = torch.cat([ref_Dir_XYZ_Camera, zeroTensor], dim=1)           # B x 4 x 1
    directionVectorU2World4 = torch.bmm(refPosture_inv, directionVectorU2Camera4)           # B x 4 x (H * W)

    # calculate world points
    #               [ B x 1 x 4 x 1 ]                    [ B x 1 x 4 x (H * W) ]                [ B, NP, 1, (H * W) ]
    ref_XYZ_World = dstCameraAnchorPoint4.unsqueeze(1) + directionVectorU2World4.unsqueeze(1) * refDep.view(B, NP, 1, H * W) # B x NP x 4 x (H * W)
    
    # project to src cameras
    ref_XYZ_World = ref_XYZ_World.unsqueeze(1).repeat(1, N, 1, 1, 1)                 # B x N x NP x 4 x (H * W)
    ref_XYZ_World = ref_XYZ_World.view(B * N * NP, 4, H * W)                         # (B * N * NP) x 4 x (H * W)
    src_XYZ_camera = torch.bmm(srcPosture.repeat(1, 1, NP, 1, 1).view(B * N * NP, 4, 4), ref_XYZ_World)           # (B * N * NP) x 4 x (H * W)
    src_XYZ_camera = src_XYZ_camera[:, :3, :]                                        # (B * N * NP) x 3 x (H * W)
    src_XYZ_picture = torch.bmm(srcK.repeat(1, 1, NP, 1, 1).view(B * N * NP, 3, 3), src_XYZ_camera)               # (B * N * NP) x 3 x (H * W)
    src_XYZ_picture = src_XYZ_picture[:, :2, :] / (src_XYZ_picture[:, 2:3, :] + eps) # (B * N * NP) x 2 x (H * W)
    src_XYZ_picture = src_XYZ_picture.view(B * N * NP, 2, H, W)                      # (B * N * NP) x 2 x H x W
    src_XYZ_picture = src_XYZ_picture.permute(0, 2, 3, 1)                            # (B * N * NP) x H x W x 2

    # calculate grid
    grid = src_XYZ_picture / normalize_base.expand(B * N * NP, H, W, 2) - 1.

    return grid.view(B, N, NP, H, W, 2)


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

    grid = grid.view(B * N * NP, H, W, 2)

    # apply gird_sample
    warppedImg = torch.nn.functional.grid_sample(srcImgs.unsqueeze(2).repeat(1, 1, NP, 1, 1, 1).view(B * N * NP, C, HS, WS), grid, mode, padding_mode, align_corners)
    
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

    warppedImg = myEnhancedGridSample(srcImgs.unsqueeze(2).repeat(1, 1, NP, 1, 1, 1).view(B * N * NP, C, HS, WS), grid)

    return warppedImg.view(B, N, NP, C, H, W)