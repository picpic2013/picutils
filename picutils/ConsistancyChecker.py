from requests import delete
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

    def getMeanCorrectDepth_single_batch(cams: List[MyPerspectiveCamera], dMaps: List[torch.Tensor], 
        pix_thre: float, dep_thre: float, view_thre: float, absoluteDepth: bool=False, grid_sample_mode='bilinear', 
        grid_sample_padding_mode='zeros', grid_sample_align_corners=False) -> Tuple[torch.Tensor]:
        '''
        @param cams:      len(cams) = view_num
        @param dMaps:     dMaps[i].shape = [ H x W ]
        @param pix_thre:  mask[reprojected pixel diff > pix_thre] = 0
        @param dep_thre:  mask[reprojected depth diff > dep_thre] = 0
        @param view_thre: mask[number of cams that seen the point < view_thre] = 0
        @param absoluteDepth: if false, [reprojected depth diff] /= raw_depth

        @return [ (correct_dep, mask), ... ]
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

    def getMeanCorrectDepth(cams: List[List[MyPerspectiveCamera]], dMaps: torch.Tensor, 
        pix_thre: float, dep_thre: float, view_thre: float, absoluteDepth: bool=False, eps=1e-8, 
        grid_sample_mode='bilinear', grid_sample_padding_mode='zeros', grid_sample_align_corners=False) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        @param cams:      len(cams) = BatchSize && len(cams[i]) = view_num
        @param dMaps:     dMaps.shape = [ B x V x H x W ]
        @param pix_thre:  mask[reprojected pixel diff > pix_thre] = 0
        @param dep_thre:  mask[reprojected depth diff > dep_thre] = 0
        @param view_thre: mask[number of cams that seen the point < view_thre] = 0
        @param absoluteDepth: if false, [reprojected depth diff] /= raw_depth

        @return Tuple [ reproject_dMaps, mask ]
            reproject_dMaps: [ B x V x H x W ]
            mask:            [ B x V x H x W ]
        '''

        # construct info
        B, V, H, W = dMaps.shape
        device = dMaps.device
        dtype = dMaps.dtype

        # construct util vectors
        oneTensor = torch.ones(B * V, 1, H * W, dtype=dtype, device=device)       # (B * V) x 1 x (H * W)
        zeroTensor = torch.zeros(B * V, 1, H * W, dtype=dtype, device=device)     # (B * V) x 1 x (H * W)
        tensor0001 = torch.tensor([0., 0., 0., 1.], dtype=dtype, device=device)   # 4
        normalize_base = torch.tensor([W, H], dtype=dtype, device=device) * 0.5   # 2

        # construct posture and K
        posture = torch.stack([torch.stack([cam.posture for cam in camList]) for camList in cams])         # B x V x 4 x 4
        posture_inv = torch.stack([torch.stack([cam.posture_inv for cam in camList]) for camList in cams]) # B x V x 4 x 4
        k = torch.stack([torch.stack([cam.k for cam in camList]) for camList in cams])                     # B x V x 3 x 3
        k_inv = torch.stack([torch.stack([cam.k_inv for cam in camList]) for camList in cams])             # B x V x 3 x 3

        posture = posture.view(B * V, 4, 4).float()         # (B * V) x 4 x 4
        posture_inv = posture_inv.view(B * V, 4, 4).float() # (B * V) x 4 x 4
        k = k.view(B * V, 3, 3).float()                     # (B * V) x 3 x 3
        k_inv = k_inv.view(B * V, 3, 3).float()             # (B * V) x 3 x 3

        # project points
        grid = cams[0][0].uv_grid.view(1, 1, 2, H, W).repeat(B, V, 1, 1 ,1) # B x V x 2 x H x W
        grid = grid.view(B * V, 2, H * W)                                   # (B * V) x 2 x (H * W)

        v_posture = posture.view(B, V, 1, 4, 4).repeat(1, 1, V, 1, 1).permute(1, 0, 2, 3, 4).reshape(V * B * V, 4, 4)         # (V_src * B * V_ref) x 4 x 4
        v_posture_inv = posture_inv.view(B, V, 1, 4, 4).repeat(1, 1, V, 1, 1).permute(1, 0, 2, 3, 4).reshape(V * B * V, 4, 4) # (V_src * B * V_ref) x 4 x 4
        v_k = k.view(B, V, 1, 3, 3).repeat(1, 1, V, 1, 1).permute(1, 0, 2, 3, 4).reshape(V * B * V, 3, 3)                     # (V_src * B * V_ref) x 3 x 3
        v_k_inv = k_inv.view(B, V, 1, 3, 3).repeat(1, 1, V, 1, 1).permute(1, 0, 2, 3, 4).reshape(V * B * V, 3, 3)             # (V_src * B * V_ref) x 3 x 3
        v_oneTensor = oneTensor.repeat(V, 1, 1)     # (V_src * B * V_ref) x 1 x (H * W)
        v_zeroTensor = zeroTensor.repeat(V, 1, 1)   # (V_src * B * V_ref) x 1 x (H * W)
        
        ref_Dir_XYZ_Picture = torch.cat([grid, oneTensor], dim=1)  # (B * V) x 3 x (H * W)
        ref_Dir_XYZ_Camera = torch.bmm(k_inv, ref_Dir_XYZ_Picture) # (B * V) x 3 x (H * W)

        dstCameraAnchorPoint4 = torch.bmm(posture_inv, tensor0001.view(1, 4, 1).repeat(B * V, 1, 1)) # (B * V) x 4 x 1
        directionVectorU2Camera4 = torch.cat([ref_Dir_XYZ_Camera, zeroTensor], dim=1)                # (B * V) x 4 x (H * W)
        directionVectorU2World4 = torch.bmm(posture_inv, directionVectorU2Camera4)                   # (B * V) x 4 x (H * W)

        # calculate world points
        #               [ (B * V) x 4 x 1 ]     [ (B * V) x 4 x (H * W) ] [ (B * V), 1, (H * W) ]
        ref_XYZ_World = dstCameraAnchorPoint4 + directionVectorU2World4 * dMaps.view(B * V, 1, H * W) # (B * V) x 4 x (H * W)
        ref_XYZ_World = ref_XYZ_World.repeat(V, 1, 1) # (V_src * B * V_ref) x 4 x (H * W)

        # project to src cameras
        src_XYZ_camera = torch.bmm(v_posture, ref_XYZ_World)           # (V_src * B * V_ref) x 4 x (H * W)
        src_XYZ_camera = src_XYZ_camera[:, :3, :]                      # (V_src * B * V_ref) x 3 x (H * W)
        src_XYZ_picture = torch.bmm(v_k, src_XYZ_camera)               # (V_src * B * V_ref) x 3 x (H * W)
        src_XYZ_picture = src_XYZ_picture[:, :2, :] / (src_XYZ_picture[:, 2:3, :] + eps) # (V_src * B * V_ref) x 2 x (H * W)
        src_XYZ_picture = src_XYZ_picture.view(V * B * V, 2, H, W)                       # (V_src * B * V_ref) x 2 x H x W
        src_XYZ_picture = src_XYZ_picture.permute(0, 2, 3, 1)                            # (V_src * B * V_ref) x H x W x 2

        # calculate grid
        grid_norm = src_XYZ_picture / normalize_base.expand(V * B * V, H, W, 2) - 1.          # (V_src * B * V_ref) x H x W x 2

        v_dMaps = dMaps.view(B * V, 1, H, W).repeat(V, 1, 1, 1) # (V_src * B * V_ref) x 1 x H x W
        src_dMaps = torch.nn.functional.grid_sample(v_dMaps, grid_norm, grid_sample_mode, grid_sample_padding_mode, grid_sample_align_corners) # (V_src * B * V_ref) x 1 x H x W

        # project points
        grid = grid.repeat(V, 1, 1) # (V_src * B * V_ref) x 2 x (H * W)

        del oneTensor
        del zeroTensor
        del posture
        del posture_inv
        del k
        del k_inv
        del ref_Dir_XYZ_Picture
        del ref_Dir_XYZ_Camera
        del dstCameraAnchorPoint4
        del directionVectorU2Camera4
        del directionVectorU2World4
        del ref_XYZ_World
        del src_XYZ_camera
        del src_XYZ_picture
        del grid_norm
        
        ref_Dir_XYZ_Picture = torch.cat([grid, v_oneTensor], dim=1)  # (V_src * B * V_ref) x 3 x (H * W)
        ref_Dir_XYZ_Camera = torch.bmm(v_k_inv, ref_Dir_XYZ_Picture) # (V_src * B * V_ref) x 3 x (H * W)

        dstCameraAnchorPoint4 = torch.bmm(v_posture_inv, tensor0001.view(1, 4, 1).repeat(V * B * V, 1, 1)) # (V_src * B * V_ref) x 4 x 1
        directionVectorU2Camera4 = torch.cat([ref_Dir_XYZ_Camera, v_zeroTensor], dim=1)                    # (V_src * B * V_ref) x 4 x (H * W)
        directionVectorU2World4 = torch.bmm(v_posture_inv, directionVectorU2Camera4)                       # (V_src * B * V_ref) x 4 x (H * W)

        # calculate world points
        #               [ (VBV) x 4 x 1 ]       [ (VBV) x 4 x (H * W) ]   [ (VBV), 1, (H * W) ]
        ref_XYZ_World = dstCameraAnchorPoint4 + directionVectorU2World4 * src_dMaps.view(V * B * V, 1, H * W) # (V_src * B * V_ref) x 4 x (H * W)

        # project to src cameras
        src_XYZ_camera = torch.bmm(v_posture, ref_XYZ_World)           # (V_src * B * V_ref) x 4 x (H * W)
        src_XYZ_camera = src_XYZ_camera[:, :3, :]                      # (V_src * B * V_ref) x 3 x (H * W)
        src_XYZ_picture = torch.bmm(v_k, src_XYZ_camera)               # (V_src * B * V_ref) x 3 x (H * W)

        reproject_dMaps =  src_XYZ_picture[:, 2:3, :]                  # (V_src * B * V_ref) x 1 x (H * W)
        src_XYZ_picture = src_XYZ_picture[:, :2, :] / (src_XYZ_picture[:, 2:3, :] + eps) # (V_src * B * V_ref) x 2 x (H * W)

        # build mask
        dist = ((src_XYZ_picture - grid)**2).sum(dim=1).unsqueeze(1)**0.5        # (V_src * B * V_ref) x 1 x (H * W)
        depth_diff = (reproject_dMaps - v_dMaps.view(V * B * V, 1, H * W)).abs() # (V_src * B * V_ref) x 1 x (H * W)
        if absoluteDepth:
            relative_depth_diff = depth_diff
        else:
            relative_depth_diff = depth_diff / (v_dMaps.view(V * B * V, 1, H * W).abs() + eps)

        mask = (dist < pix_thre) & (relative_depth_diff < dep_thre) # (V_src * B * V_ref) x 1 x (H * W)
        reproject_dMaps[~mask] = 0

        # build better depth
        mask = mask.view(V, B, V, H, W).permute(1, 0, 2, 3, 4)                        # B x V_src x V_ref x H x W
        reproject_dMaps = reproject_dMaps.view(V, B, V, H, W).permute(1, 0, 2, 3, 4)  # B x V_src x V_ref x H x W

        geo_mask_sum = mask.int().sum(dim=1)         # B x V_ref x H x W
        reproject_dMaps = reproject_dMaps.sum(dim=1) # B x V_ref x H x W
        reproject_dMaps = reproject_dMaps / geo_mask_sum
        
        return reproject_dMaps, mask.sum(dim=1) >= view_thre + 1