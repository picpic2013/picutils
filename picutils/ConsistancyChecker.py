import torch
import numpy as np
from typing import Tuple, List

from picutils.MyPerspectiveCamera import MyPerspectiveCamera

class ConsistancyChecker:
    
    def checkConsistancy(refCam: MyPerspectiveCamera, refDMap: torch.Tensor, srcCam: MyPerspectiveCamera, srcDMap: torch.Tensor, 
        pix_thre: float, dep_thre: float, absoluteDepth: bool=False, grid_sample_mode='bilinear', grid_sample_padding_mode='zeros', 
        grid_sample_align_corners=False) -> Tuple[torch.Tensor]:
        
        device = refDMap.device
        dtype = refDMap.dtype

        height, width = refDMap.shape[:2]
        u, v = refCam.uv_grid
        valid_points = torch.ones_like(refDMap).to(device).bool()
        u, v, depth_ref = u[valid_points], v[valid_points], refDMap[valid_points]
        uvs = torch.stack([u, v])

        _, _, base, direction = refCam.uv2WorldLine(uvs)
        xyz_world_ref = base + depth_ref * direction

        uv_src = srcCam.world2uv(xyz_world_ref)
        uv_src_grid_input = uv_src.permute(1, 0).unsqueeze(0).unsqueeze(0)
        normalize_base = torch.tensor([width * 0.5, height * 0.5], dtype=dtype, device=device).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        uv_src_grid_input = (uv_src_grid_input - normalize_base) / normalize_base
        srcDMap_grid_input = srcDMap.unsqueeze(0).unsqueeze(0)
        src_dep = torch.nn.functional.grid_sample(srcDMap_grid_input, uv_src_grid_input, grid_sample_mode, grid_sample_padding_mode, grid_sample_align_corners)

        src_dep = src_dep.squeeze(0).squeeze(0).squeeze(0)

        _, _, base, direction = srcCam.uv2WorldLine(uv_src)
        xyz_world_src = base + src_dep * direction

        uv_ref = refCam.world2uv(xyz_world_src)
        depth_reprojected = refCam.getDistToPlane(xyz_world_src)

        dist = ((uvs - uv_ref)**2).sum(dim=0)**0.5
        depth_diff = (depth_reprojected - depth_ref).abs()
        if absoluteDepth:
            relative_depth_diff = depth_diff
        else:
            relative_depth_diff = depth_diff / depth_ref

        mask = (dist < pix_thre) & (relative_depth_diff < dep_thre)
        depth_reprojected[~mask] = 0

        return mask.reshape(height, width), depth_reprojected.reshape(height, width)

    def getMeanCorrectDepth(cams: List[MyPerspectiveCamera], dMaps: List[torch.Tensor], 
        pix_thre: float, dep_thre: float, view_thre: float, absoluteDepth: bool=False, grid_sample_mode='bilinear', 
        grid_sample_padding_mode='zeros', grid_sample_align_corners=False) -> Tuple[torch.Tensor]:
        '''
        @param cams:      len(cams) = view_num
        @param dMaps:     dMaps[i].shape = [ H x W ]
        @param pix_thre:  mask[reprojected pixel diff > pix_thre] = 0
        @param dep_thre:  mask[reprojected depth diff > dep_thre] = 0
        @param view_thre: mask[number of cams that seen the point < view_thre] = 0
        @param absoluteDepth: if false, [reprojected depth diff] /= raw_depth
        '''
        for refIdx, (refCam, refDMap) in enumerate(zip(cams, dMaps)):
            geo_mask_sum = 0
            all_srcview_depth_ests = []
            for srcIdx, (srcCam, srcDMap) in enumerate(zip(cams, dMaps)):
                if refIdx == srcIdx:
                    continue

                geo_mask, depth_reprojected = ConsistancyChecker.checkConsistancy(
                    refCam, refDMap, srcCam, srcDMap, pix_thre, dep_thre, absoluteDepth, grid_sample_mode, 
                    grid_sample_padding_mode, grid_sample_align_corners)

                geo_mask_sum = geo_mask_sum + geo_mask.int()
                all_srcview_depth_ests.append(depth_reprojected)

            depth_est_averaged = (torch.stack(all_srcview_depth_ests).sum(dim=0) + refDMap) / (geo_mask_sum + 1)
            geo_mask = geo_mask_sum >= view_thre

            yield depth_est_averaged, geo_mask

