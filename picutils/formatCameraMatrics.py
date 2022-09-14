import torch
from typing import Tuple, List
from picutils.MyPerspectiveCamera import MyPerspectiveCamera

def formatMyPerspectiveCameraMatrics(
        refCam: List[MyPerspectiveCamera],        # len(refCam) : B
        srcCams: List[List[MyPerspectiveCamera]], # len(srcCams) : B && len(srcCams[0]) : V
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    '''
    format camera params
    @param refCam: len(refCam) : B
    @param srcCams: len(srcCams) : B && len(srcCams[0]) : V
    
    @returns:
        refExt_inv:  B x 4 x 4
        refIntr_inv: B x 3 x 3
        srcExt:      B x V x 4 x 4
        srcIntr:     B x V x 3 x 3
        grid:        2 x H x W (u, v)
    '''
    
    refExt_inv = torch.stack([cam.posture_inv for cam in refCam])
    refIntr_inv = torch.stack([cam.k_inv for cam in refCam])
    srcExt = torch.stack([torch.stack([cam.posture for cam in camList]) for camList in srcCams])
    srcIntr = torch.stack([torch.stack([cam.k for cam in camList]) for camList in srcCams])
    grid = refCam[0].uv_grid

    return refExt_inv, refIntr_inv, srcExt, srcIntr, grid