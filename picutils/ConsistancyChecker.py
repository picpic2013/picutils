import torch
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
        dtype = cams[0][0].posture.dtype

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

        posture = posture.view(B * V, 4, 4).type(dtype)         # (B * V) x 4 x 4
        posture_inv = posture_inv.view(B * V, 4, 4).type(dtype) # (B * V) x 4 x 4
        k = k.view(B * V, 3, 3).type(dtype)                     # (B * V) x 3 x 3
        k_inv = k_inv.view(B * V, 3, 3).type(dtype)             # (B * V) x 3 x 3

        # project points
        grid = cams[0][0].uv_grid.view(1, 1, 2, H, W).repeat(B, V, 1, 1 ,1) # B x V x 2 x H x W
        grid = grid.view(B * V, 2, H * W)                                   # (B * V) x 2 x (H * W)
        v_grid = grid.repeat(V, 1, 1)

        v_posture = posture.view(B, V, 1, 4, 4).repeat(1, 1, V, 1, 1)         # B x V_src x V_ref x 4 x 4
        v_posture_inv = posture_inv.view(B, V, 1, 4, 4).repeat(1, 1, V, 1, 1) # B x V_src x V_ref x 4 x 4
        v_k = k.view(B, V, 1, 3, 3).repeat(1, 1, V, 1, 1)                     # B x V_src x V_ref x 3 x 3
        v_k_inv = k_inv.view(B, V, 1, 3, 3).repeat(1, 1, V, 1, 1)             # B x V_src x V_ref x 3 x 3
        v_oneTensor = oneTensor.repeat(V, 1, 1)     # (B * V_src * V_ref) x 1 x (H * W)
        v_zeroTensor = zeroTensor.repeat(V, 1, 1)   # (B * V_src * V_ref) x 1 x (H * W)
        
        ref_Dir_XYZ_Picture = torch.cat([grid, oneTensor], dim=1)  # (B * V) x 3 x (H * W)
        ref_Dir_XYZ_Camera = torch.bmm(k_inv, ref_Dir_XYZ_Picture) # (B * V) x 3 x (H * W)

        dstCameraAnchorPoint4 = torch.bmm(posture_inv, tensor0001.view(1, 4, 1).repeat(B * V, 1, 1)) # (B * V) x 4 x 1
        directionVectorU2Camera4 = torch.cat([ref_Dir_XYZ_Camera, zeroTensor], dim=1)                # (B * V) x 4 x (H * W)
        directionVectorU2World4 = torch.bmm(posture_inv, directionVectorU2Camera4)                   # (B * V) x 4 x (H * W)

        # calculate world points
        #               [ (B * V) x 4 x 1 ]     [ (B * V) x 4 x (H * W) ] [ (B * V), 1, (H * W) ]
        ref_XYZ_World = dstCameraAnchorPoint4 + directionVectorU2World4 * dMaps.view(B * V, 1, H * W) # (B * V) x 4 x (H * W)
        ref_XYZ_World = ref_XYZ_World.view(B, V, 1, 4, H * W).repeat(1, 1, V, 1, 1)       # B x V_ref x V_src x 4 x (H * W)
        ref_XYZ_World = ref_XYZ_World.reshape(B * V * V, 4, H * W) # (B * V_src * V_ref) x 4 x (H * W)

        # project to src cameras
        src_XYZ_camera = torch.bmm(v_posture.permute(0, 2, 1, 3, 4).reshape(B * V * V, 4, 4), ref_XYZ_World)           # (B * V_src * V_ref) x 4 x (H * W)
        src_XYZ_camera = src_XYZ_camera[:, :3, :]                      # (B * V_src * V_ref) x 3 x (H * W)
        src_XYZ_picture = torch.bmm(v_k.permute(0, 2, 1, 3, 4).reshape(B * V * V, 3, 3), src_XYZ_camera)               # (B * V_src * V_ref) x 3 x (H * W)
        src_XYZ_picture = src_XYZ_picture[:, :2, :] / (src_XYZ_picture[:, 2:3, :] + eps) # (B * V_src * V_ref) x 2 x (H * W)
        grid = src_XYZ_picture
        src_XYZ_picture = src_XYZ_picture.view(B * V * V, 2, H, W)                       # (B * V_src * V_ref) x 2 x H x W
        src_XYZ_picture = src_XYZ_picture.permute(0, 2, 3, 1)                            # (B * V_src * V_ref) x H x W x 2

        # calculate grid
        grid_norm = src_XYZ_picture / normalize_base.view(1, 1, 1, 2).expand(B * V * V, H, W, 2) - 1.     # (B * V_src * V_ref) x H x W x 2
        grid_norm = grid_norm.float()

        src_dMaps = torch.nn.functional.grid_sample(
            dMaps.view(B, 1, V, 1, H, W).repeat(1, V, 1, 1, 1, 1).view(B * V * V, 1, H, W), 
            grid_norm, grid_sample_mode, grid_sample_padding_mode, grid_sample_align_corners
        ) # (B * V_src * V_ref) x 1 x H x W

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
        
        # project points
        ref_Dir_XYZ_Picture = torch.cat([grid, v_oneTensor], dim=1)  # (B * V_src * V_ref) x 3 x (H * W)
        ref_Dir_XYZ_Camera = torch.bmm(v_k_inv.permute(0, 2, 1, 3, 4).reshape(B * V * V, 3, 3), ref_Dir_XYZ_Picture) # (B * V_src * V_ref) x 3 x (H * W)

        dstCameraAnchorPoint4 = torch.bmm(v_posture_inv.permute(0, 2, 1, 3, 4).reshape(B * V * V, 4, 4), tensor0001.view(1, 4, 1).repeat(B * V * V, 1, 1)) # (B * V_src * V_ref) x 4 x 1
        directionVectorU2Camera4 = torch.cat([ref_Dir_XYZ_Camera, v_zeroTensor], dim=1)                    # (V_src * B * V_ref) x 4 x (H * W)
        directionVectorU2World4 = torch.bmm(v_posture_inv.permute(0, 2, 1, 3, 4).reshape(B * V * V, 4, 4), directionVectorU2Camera4)                       # (B * V_src * V_ref) x 4 x (H * W)

        # calculate world points
        #               [ (BVV) x 4 x 1 ]       [ (BVV) x 4 x (H * W) ]   [ (BVV) x 1 x (H * W) ]
        ref_XYZ_World = dstCameraAnchorPoint4 + directionVectorU2World4 * src_dMaps.view(B * V * V, 1, H * W) # (B * V_src * V_ref) x 4 x (H * W)

        # project to src cameras
        src_XYZ_camera = torch.bmm(v_posture.view(B * V * V, 4, 4), ref_XYZ_World)       # (B * V_src * V_ref) x 4 x (H * W)
        src_XYZ_camera = src_XYZ_camera[:, :3, :]                                        # (B * V_src * V_ref) x 3 x (H * W)
        src_XYZ_picture = torch.bmm(v_k.view(B * V * V, 3, 3), src_XYZ_camera)           # (B * V_src * V_ref) x 3 x (H * W)

        reproject_dMaps =  src_XYZ_picture[:, 2:3, :]                                    # (B * V_src * V_ref) x 1 x (H * W)
        src_XYZ_picture = src_XYZ_picture[:, :2, :] / (src_XYZ_picture[:, 2:3, :] + eps) # (B * V_src * V_ref) x 2 x (H * W)

        # build mask
        dist = ((src_XYZ_picture - v_grid)**2).sum(dim=1).unsqueeze(1)**0.5        # (B * V_src * V_ref) x 1 x (H * W)
        v_dMaps = dMaps.view(B, V, 1, H * W).repeat(1, 1, V, 1).permute(1, 0, 2, 3).reshape(B * V * V, 1, H * W)
        depth_diff = (reproject_dMaps - v_dMaps).abs() # (B * V_src * V_ref) x 1 x (H * W)
        if absoluteDepth:
            relative_depth_diff = depth_diff
        else:
            relative_depth_diff = depth_diff / (v_dMaps.abs() + eps)

        mask = (dist < pix_thre) & (relative_depth_diff < dep_thre) # (B * V_src * V_ref) x 1 x (H * W)
        reproject_dMaps[~mask] = 0

        # build better depth
        mask = mask.view(B, V, V, H, W)                        # B x V_src x V_ref x H x W
        reproject_dMaps = reproject_dMaps.view(B, V, V, H, W)  # B x V_src x V_ref x H x W

        geo_mask_sum = mask.sum(dim=2)               # B x V_ref x H x W
        reproject_dMaps = reproject_dMaps.sum(dim=2) # B x V_ref x H x W
        reproject_dMaps = reproject_dMaps / (geo_mask_sum + eps)

        return reproject_dMaps, geo_mask_sum > view_thre

class FastConsistancyChecker:
    def __init__(self, cams: List[List[MyPerspectiveCamera]], 
        pix_thre: float, dep_thre: float, view_thre: float, absoluteDepth: bool=False, eps=1e-8, 
        grid_sample_mode='bilinear', grid_sample_padding_mode='zeros', grid_sample_align_corners=False) -> None:
        '''
        @param cams:      len(cams) = BatchSize && len(cams[i]) = view_num
        @param pix_thre:  mask[reprojected pixel diff > pix_thre] = 0
        @param dep_thre:  mask[reprojected depth diff > dep_thre] = 0
        @param view_thre: mask[number of cams that seen the point < view_thre] = 0
        @param absoluteDepth: if false, [reprojected depth diff] /= raw_depth
        '''
        self.cams = cams
        self.pix_thre = pix_thre
        self.dep_thre = dep_thre
        self.view_thre = view_thre
        self.absoluteDepth = absoluteDepth
        self.eps = eps
        self.grid_sample_mode = grid_sample_mode
        self.grid_sample_padding_mode = grid_sample_padding_mode
        self.grid_sample_align_corners = grid_sample_align_corners

        # read info
        self.device = cams[0][0].posture.device
        self.dtype = cams[0][0].posture.dtype
        self.B = len(cams)
        self.V = len(cams[0])
        self.H = cams[0][0].imgH
        self.W = cams[0][0].imgW

        # construct posture and K
        posture = torch.stack([torch.stack([cam.posture for cam in camList]) for camList in cams])               # B x V x 4 x 4
        posture_inv = torch.stack([torch.stack([cam.posture_inv for cam in camList]) for camList in cams])       # B x V x 4 x 4
        k = torch.eye(4, device=self.device, dtype=self.dtype).view(1, 1, 4, 4).repeat(self.B, self.V, 1, 1)     # B x V x 4 x 4
        k_inv = torch.eye(4, device=self.device, dtype=self.dtype).view(1, 1, 4, 4).repeat(self.B, self.V, 1, 1) # B x V x 4 x 4
        k[:,:,:3,:3] = torch.stack([torch.stack([cam.k for cam in camList]) for camList in cams])
        k_inv[:,:,:3,:3] = torch.stack([torch.stack([cam.k_inv for cam in camList]) for camList in cams])

        posture = posture.unsqueeze(1).repeat(1, self.V, 1, 1, 1)         # B x V x V x 4 x 4
        posture_inv = posture_inv.unsqueeze(1).repeat(1, self.V, 1, 1, 1) # B x V x V x 4 x 4
        k = k.unsqueeze(1).repeat(1, self.V, 1, 1, 1)                     # B x V x V x 4 x 4
        k_inv = k_inv.unsqueeze(1).repeat(1, self.V, 1, 1, 1)             # B x V x V x 4 x 4

        # construct line cords
        # (BVV) x 4 x 4
        KRRK = torch.bmm(
            posture_inv.permute(0, 2, 1, 3, 4).reshape(self.B * self.V * self.V, 4, 4), 
            k_inv.permute(0, 2, 1, 3, 4).reshape(self.B * self.V * self.V, 4, 4)
        )
        KRRK = torch.bmm(posture.view(self.B * self.V * self.V, 4, 4), KRRK)
        KRRK = torch.bmm(k.view(self.B * self.V * self.V, 4, 4), KRRK)
        self.KRRK = KRRK.view(self.B, self.V, self.V, 4, 4) # BVV44
        self.KRRK_inv = self.KRRK.permute(0, 2, 1, 3, 4).contiguous()

        refGrid = cams[0][0].uv_grid.unsqueeze(0).repeat(self.B * self.V * self.V, 1, 1, 1).view(self.B * self.V * self.V, 2, self.H * self.W) # BVV x 2 x H x W => BVV x 2 x (H * W)
        self.refGrid = refGrid
        # (BVV) x 4 x (HW) ,,  [ u, v, 1, 0 ]
        direction_ref = torch.zeros(self.B * self.V * self.V, 4, self.H * self.W, device=self.device, dtype=self.dtype)
        direction_ref[:,:2,:] = refGrid
        direction_ref[:,2,:] = 1
        self.direction_ref = direction_ref

        # (BVV) x 4 x (HW) ,,  [ 0, 0, 0, 1 ]
        basePoint_ref = torch.zeros(self.B * self.V * self.V, 4, self.H * self.W, device=self.device, dtype=self.dtype)
        basePoint_ref[:,3,:] = 1
        self.basePoint_ref = basePoint_ref

        basePoint_src = torch.bmm(KRRK, basePoint_ref).view(self.B, self.V, self.V, 4, self.H, self.W)[:,:,:,:3,:,:]
        direction_src = torch.bmm(KRRK, direction_ref).view(self.B, self.V, self.V, 4, self.H, self.W)[:,:,:,:3,:,:]

        HS = cams[0][0].imgH
        WS = cams[0][0].imgW
        normalize_base = torch.tensor([WS, HS], dtype=self.dtype, device=self.device) * 0.5
        normalize_base = normalize_base.view(1, 1, 1, 2, 1, 1)
        self.normalize_base = normalize_base
        # basePoint_src[:,:,:,:2,:,:] = basePoint_src[:,:,:,:2,:,:] / normalize_base
        # direction_src[:,:,:,:2,:,:] = direction_src[:,:,:,:2,:,:] / normalize_base

        self.basePoint_src = basePoint_src
        self.direction_src = direction_src

    def getMeanCorrectDepth(self, dMaps: torch.Tensor):
        '''
        @param dMaps:     dMaps.shape = [ B x V x H x W ]

        @return Tuple [ reproject_dMaps, mask ]
            reproject_dMaps: [ B x V x H x W ]
            mask:            [ B x V x H x W ]
        '''

        srcGrid = self.basePoint_src + self.direction_src * dMaps.view(self.B, self.V, 1, 1, self.H, self.W).repeat(1, 1, self.V, 1, 1, 1) # BVV3HW
        srcGrid = srcGrid[:,:,:,:2,:,:] / (srcGrid[:,:,:,2:,:,:] + self.eps) # BVV2HW

        srcGrids = srcGrid / self.normalize_base - 1.
        srcGrids = srcGrids.permute(0, 1, 2, 4, 5, 3).reshape(self.B * self.V * self.V, self.H, self.W, 2)

        if srcGrids.dtype != dMaps.dtype:
            srcGrids = srcGrids.type(dMaps.dtype)
        
        srcDep = torch.nn.functional.grid_sample(
            dMaps.view(self.B, 1, self.V, 1, self.H, self.W).repeat(1, self.V, 1, 1, 1, 1).view(self.B * self.V * self.V, 1, self.H, self.W), 
            srcGrids, self.grid_sample_mode, self.grid_sample_padding_mode, self.grid_sample_align_corners
        ) # (BVV)1HW

        direction_ref = self.direction_ref.clone()
        direction_ref[:,:2,:] = srcGrid.view(self.B * self.V * self.V, 2, self.H * self.W)
        basePoint_src = torch.bmm(self.KRRK_inv.view(self.B * self.V * self.V, 4, 4), self.basePoint_ref)[:,:3,:] # (BVV)3(HW)
        direction_src = torch.bmm(self.KRRK_inv.view(self.B * self.V * self.V, 4, 4), direction_ref)[:,:3,:]      # (BVV)3(HW)

        refGrid = basePoint_src + direction_src * srcDep.view(self.B * self.V * self.V, 1, self.H * self.W)
        reprojectDep = refGrid[:,2:,:] # (BVV)1(HW)
        refGrids = refGrid[:,:2,:] / (reprojectDep + self.eps) # (BVV)2(HW)

        # build mask
        dist = ((refGrids - self.refGrid)**2).sum(dim=1).unsqueeze(1)**0.5        # (B * V_src * V_ref) x 1 x (H * W)
        v_dMaps = dMaps.view(self.B, self.V, 1, self.H * self.W).repeat(1, 1, self.V, 1).view(self.B * self.V * self.V, 1, self.H * self.W)
        depth_diff = (reprojectDep - v_dMaps).abs() # (B * V_src * V_ref) x 1 x (H * W)
        if self.absoluteDepth:
            relative_depth_diff = depth_diff
        else:
            relative_depth_diff = depth_diff / (v_dMaps.abs() + self.eps)

        mask = (dist < self.pix_thre) & (relative_depth_diff < self.dep_thre) & (relative_depth_diff > 0) # (B * V_src * V_ref) x 1 x (H * W)
        reprojectDep[~mask] = 0

        # build better depth
        mask = mask.view(self.B, self.V, self.V, self.H, self.W)                        # B x V_src x V_ref x H x W
        reprojectDep = reprojectDep.view(self.B, self.V, self.V, self.H, self.W)  # B x V_src x V_ref x H x W

        geo_mask_sum = mask.sum(dim=2)               # B x V_ref x H x W
        reprojectDep = reprojectDep.sum(dim=2) # B x V_ref x H x W
        reprojectDep = reprojectDep / (geo_mask_sum + self.eps)

        return reprojectDep, geo_mask_sum > self.view_thre