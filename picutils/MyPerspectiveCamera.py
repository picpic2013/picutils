from __future__ import annotations
from typing import Tuple, Dict

import torch
import numpy as np
import cv2

from picutils.MyImageTranslator import MyImageTranslator
from picutils.MyPosture import MyPosture
from picutils.utils import picDefaultDeviceAndDtype, addOnesToVect, batchSqueeze, batchUnsqueeze

class MyPerspectiveCamera:
    
    _uvDict: Dict[Tuple, torch.Tensor] = {}

    @property
    def uvDict(self) -> Dict[Tuple, torch.Tensor]:
        return MyPerspectiveCamera._uvDict
    @uvDict.setter
    def uvDict(self, dic):
        MyPerspectiveCamera._uvDict = dic

    def getUVGrid(self, imgH, imgW, device, dtype):
        if (int(imgH), int(imgW), device, dtype) not in MyPerspectiveCamera._uvDict.keys():
            u, v = np.meshgrid(np.arange(0, imgW), np.arange(0, imgH))
            u, v = torch.from_numpy(u).to(device).type(dtype), torch.from_numpy(v).to(device).type(dtype)
            uvs = torch.stack([u, v])
            uvs.requires_grad = False
            MyPerspectiveCamera._uvDict.setdefault((int(imgH), int(imgW), device, dtype), uvs)
        uvs = MyPerspectiveCamera._uvDict[(int(imgH), int(imgW), device, dtype)]
        return uvs

    def __init__(self, k, posture_world2cam, imgH, imgW, resize=None, dtype=None, device=None, requires_grad=False):
        '''
        k: [[fx,  s, x0], 
            [ 0, fy, y0], 
            [ 0,  0,  1]]
        
        posture_world2cam: [[r, r, r, t], 
                           [r, r, r, t], 
                           [r, r, r, t], 
                           [0, 0, 0, 1]]
                           Xc = R @ Xw

        resize: None | Tuple (imgH, imgW)
        '''
        device, dtype = picDefaultDeviceAndDtype(device=device, dtype=dtype)
        if resize is None:
            resize = [imgH, imgW]
        if type(resize) != torch.Tensor:
            resize = torch.tensor(resize)
        self.imgH, self.imgW = resize
        resize = resize / torch.tensor([imgH, imgW])
        resize = addOnesToVect(resize.unsqueeze(1))
        
        if type(k) != torch.Tensor:
            k = torch.tensor(k, dtype=torch.float64)
        
        if type(posture_world2cam) == MyPosture:
            posture_world2cam = posture_world2cam.getPosture()
        
        if type(posture_world2cam) != torch.Tensor:
            posture_world2cam = torch.tensor(posture_world2cam, dtype=torch.float64)
        
        posture_world2cam = posture_world2cam.cpu().double()

        self.k = resize.cpu() * k.cpu().double()
        self.posture = posture_world2cam
        
        assert self.k.shape == torch.Size([3, 3])
        assert self.posture.shape == torch.Size([4, 4])
        
        try:
            self.k_inv = torch.tensor(np.linalg.inv(self.k), dtype=torch.float64)
        except:
            print('k is Singular matrix')
        
        try:
            self.posture_inv = torch.tensor(np.linalg.inv(self.posture), dtype=torch.float64)
        except:
            print('posture is Singular matrix')

        # u, v = np.meshgrid(np.arange(0, self.imgW), np.arange(0, self.imgH))
        # u, v = torch.from_numpy(u).to(device).type(torch.float32), torch.from_numpy(v).to(device).type(torch.float32)
        # uvs = torch.stack([u, v])
        uvs = self.getUVGrid(self.imgH, self.imgW, device, dtype)
        
        self.uv_grid = uvs
        self.k = self.k.type(dtype=dtype).to(device)
        self.k_inv = self.k_inv.type(dtype=dtype).to(device)
        self.posture = self.posture.type(dtype=dtype).to(device)
        self.posture_inv = self.posture_inv.type(dtype=dtype).to(device)

        # self.uv_grid.requires_grad = False
        self.k.requires_grad = requires_grad
        self.k_inv.requires_grad = requires_grad
        self.posture.requires_grad = requires_grad
        self.posture_inv.requires_grad = requires_grad

    @staticmethod
    def buildFromCamera(cam, resize=None, dtype=None, device=None, requires_grad=False):
        return MyPerspectiveCamera(cam.k, cam.posture, cam.imgH, cam.imgW, 
            resize=resize, dtype=dtype, device=device, requires_grad=requires_grad)

    def world2uv(self, Xw, eps=1e-8):
        '''
        Xw: [4 * ?] or [3 * ?]  or  [b * 4 * ?] or [b * 3 * ?]
          (x,y,z,1)    (x,y,z)

        returns: U [2 * ?] (u, v)
        '''
        needReshape = False
        batchSize = None
        if len(Xw.shape) == 3:
            Xw, batchSize = batchSqueeze(Xw)
            needReshape = True

        if Xw.shape[0] == 3:
            Xw = addOnesToVect(Xw)
        assert Xw.shape[0] == 4

        posture = self.posture.type(Xw.dtype).to(Xw.device)
        k = self.k.type(Xw.dtype).to(Xw.device)

        xYCameraSrc4 = posture @ Xw
        xYCameraSrc3 = xYCameraSrc4[:3]
        u_src3 = k @ xYCameraSrc3

        d_src = u_src3[2,:]
        u_src3 = u_src3 / (d_src.unsqueeze(0) + eps)

        u_src2 = u_src3[:2]

        if needReshape:
            u_src2 = batchUnsqueeze(u_src2, batchSize)

        return u_src2

    def world2Cam(self, Xw):
        '''
        Xw: [4 * ?] or [3 * ?]  or  [b * 4 * ?] or [b * 3 * ?]
          (x,y,z,1)    (x,y,z)

        returns: Xc4,     Xc3
           (x,y,z,1)  (x,y,z)
        '''
        needReshape = False
        batchSize = None
        if len(Xw.shape) == 3:
            Xw, batchSize = batchSqueeze(Xw)
            needReshape = True

        if Xw.shape[0] == 3:
            Xw = addOnesToVect(Xw)
        posture = self.posture.type(Xw.dtype).to(Xw.device)
        Xc4 = posture @ Xw

        if needReshape:
            Xc4 = batchUnsqueeze(Xc4, batchSize)
            return Xc4, Xc4[:, :3, :]

        return Xc4, Xc4[:3, :]

    def getDistToPlane(self, Xw):
        Xc4, _ = self.world2Cam(Xw)
        if len(Xw.shape) == 3:
            return Xc4[:, 2, :]
        return Xc4[2, :]

    def uv2WorldLine(self, U):
        '''
        Line: X = X0 + t * D, t = depth(z)
        Note that you should normalize D if t means depth to the anchor point. 
        On default, otherwise, t means z depth to the camera plane. 

        params: 
            U: [2 * ?] or   [3 * ?]  or  [b * 2 * ?] or [b * 3 * ?]
                (u, v) or (u, v, 1)

        returns: 
            dstCameraAnchorPoint4:   X0 [4 * 1] (x, y, z, 1)^T
            directionVectorU2World4:  D [4 * ?] (x, y, z, 0)^T
            dstCameraAnchorPoint3:   X0 [3 * 1] (x, y, z)^T
            directionVectorU2World3:  D [3 * ?] (x, y, z)^T
        '''
        needReshape = False
        batchSize = None
        if len(U.shape) == 3:
            U, batchSize = batchSqueeze(U)
            needReshape = True

        if U.shape[0] == 2:
            U = addOnesToVect(U)
        assert U.shape[0] == 3

        dtype = U.dtype
        device = U.device

        U = U.type(dtype)

        # posture = self.posture.type(dtype).to(device)
        posture_inv = self.posture_inv.type(dtype).to(device)
        # k = self.k.type(dtype).to(device)
        k_inv = self.k_inv.type(dtype).to(device)

        dstCameraAnchorPoint4 = posture_inv @ torch.tensor([0., 0., 0., 1.], dtype=dtype, device=device).unsqueeze(1)
        dstCameraAnchorPoint3 = dstCameraAnchorPoint4[:3]

        directionVectorU2Camera = k_inv @ U
        directionVectorU2Camera4 = torch.zeros(4, directionVectorU2Camera.shape[1], dtype=dtype, device=device)
        directionVectorU2Camera4[:3] = directionVectorU2Camera
        directionVectorU2World4 = posture_inv @ directionVectorU2Camera4
        directionVectorU2World3 = directionVectorU2World4[:3]

        if needReshape:
            dstCameraAnchorPoint4 = batchUnsqueeze(dstCameraAnchorPoint4, batchSize)
            directionVectorU2World4 = batchUnsqueeze(directionVectorU2World4, batchSize)
            dstCameraAnchorPoint3 = batchUnsqueeze(dstCameraAnchorPoint3, batchSize)
            directionVectorU2World3 = batchUnsqueeze(directionVectorU2World3, batchSize)
        
        return dstCameraAnchorPoint4, directionVectorU2World4, dstCameraAnchorPoint3, directionVectorU2World3

    def points2pic(self, Xw: torch.Tensor, colors: torch.Tensor=None, mask=None, eps=1e-8) -> torch.Tensor:
        '''
        Xw: [4 * ?] or [3 * ?]  or  [b * 4 * ?] or [b * 3 * ?]
          (x,y,z,1)    (x,y,z)
        colors: [3 * ?]  or  [b * 3 * ?] in [RGB]
        mask: [?] or [b * ?] valid points to project

        returns: image [imgH * imgW * 3]
        '''
        
        if colors is None:
            if len(Xw.shape) == 2:
                colors = torch.ones_like(Xw)[:3, :] * 255
            if len(Xw.shape) == 3:
                colors = torch.ones_like(Xw)[:, :3, :] * 255
                raise NotImplementedError

        uvs = self.world2uv(Xw, eps=eps)
        us, vs = uvs.round().long()

        us = torch.clamp(us, min=0, max=self.imgW - 1)
        vs = torch.clamp(vs, min=0, max=self.imgH - 1)

        deps = self.getDistToPlane(Xw)


        if len(Xw.shape) == 2:
            outputImg = torch.zeros(self.imgH, self.imgW, 3)
            zBuffer = torch.ones(self.imgH, self.imgW, dtype=torch.float32) * 1e10
        if len(Xw.shape) == 3:
            outputImg = torch.tensor(Xw.shape[0], self.imgH, self.imgW, 3)
            raise NotImplementedError

        for u, v, d, c in zip(us, vs, deps, colors):
            if zBuffer[v, u] > d:
                zBuffer[v, u] = d
                outputImg[v, u, :] = c

        return outputImg, zBuffer


    def fromCam(self, fromPerspectiveCam: MyPerspectiveCamera, d_dsts, eps=1e-8, mode='bilinear', padding_mode='zeros', align_corners=False) -> MyImageTranslator:        
        assert self.posture.device == fromPerspectiveCam.posture.device and \
            self.k.device == fromPerspectiveCam.k.device

        dtype = self.posture.dtype
        device = self.posture.device

        if type(d_dsts) != torch.Tensor:
            d_dsts = torch.tensor(d_dsts, dtype=dtype, device=device)

        U_dst = torch.cat([self.uv_grid, torch.ones(1, self.imgH, self.imgW, dtype=self.uv_grid.dtype, device=self.uv_grid.device)], dim=0).reshape(3, -1)
        U_dst = U_dst.type(dtype).to(device)

        # calculate r, t, k
        # srcRT, dstRT = fromPerspectiveCam.posture, self.posture
        # srcRT_inv, dstRT_inv = fromPerspectiveCam.posture_inv, self.posture_inv
        # srcK, dstK = fromPerspectiveCam.k, self.k
        # srcK_inv, dstK_inv = fromPerspectiveCam.k_inv, self.k_inv

        warp_uv_all_depth = []
        # d_srcs = []
        normalize_base = torch.tensor([fromPerspectiveCam.imgW * 0.5, fromPerspectiveCam.imgH * 0.5], dtype=dtype, device=device).unsqueeze(1)

        for nowD2 in d_dsts:
            basePoint, direction, _, _ = self.uv2WorldLine(U_dst)
            U_dst2XYWorld4 = basePoint + nowD2.view(1, -1) * direction

            # U_dst2XYCameraDst3 = nowD2.reshape(1, -1) * (dstK_inv @ U_dst)
            # U_dst2XYCameraDst4 = torch.ones(4, U_dst2XYCameraDst3.shape[1], dtype=dtype, device=device)
            # U_dst2XYCameraDst4[:3] = U_dst2XYCameraDst3
            # U_dst2XYWorld4 = dstRT_inv @ U_dst2XYCameraDst4

            grid = fromPerspectiveCam.world2uv(U_dst2XYWorld4, eps)
            grid = (grid - normalize_base) / normalize_base
            grid = grid.reshape(2, self.imgH, self.imgW)
            grid = grid.permute(1, 2, 0)
            warp_uv_all_depth.append(grid)

        allGrid = torch.stack(warp_uv_all_depth).float()

        return MyImageTranslator(allGrid, lambda x:torch.nn.functional.grid_sample(x, allGrid, mode, padding_mode, align_corners), len(d_dsts))  


    def toCam(self, dstPerspectiveCam: MyPerspectiveCamera, d_srcs, eps=1e-8, mode='bilinear', padding_mode='zeros', align_corners=False) -> MyImageTranslator:
        '''
        generate transform matrix and transform function: f: img => dstImg
        
            img:    D x C x H x W    or    C x H x W
            dstImg: D x C x H x W    
                                               ^
                          ^                    |
                          |                    |
                      warp MPI             warp image
        '''
        assert self.posture.device == dstPerspectiveCam.posture.device and \
            self.k.device == dstPerspectiveCam.k.device

        dtype = self.posture.dtype
        device = self.posture.device

        if type(d_srcs) != torch.Tensor:
            d_srcs = torch.tensor(d_srcs, dtype=dtype, device=device)
        if d_srcs.dim() == 1:
            d_srcs = d_srcs.unsqueeze(1)

        U_dst = torch.cat([dstPerspectiveCam.uv_grid, torch.ones(1, dstPerspectiveCam.imgH, dstPerspectiveCam.imgW, dtype=dtype, device=device)], dim=0).reshape(3, -1)
        U_dst = U_dst.type(dtype).to(device)

        # calculate r, t, k
        srcRT, dstRT = self.posture, dstPerspectiveCam.posture
        # srcRT_inv, dstRT_inv = fromPerspectiveCam.posture_inv, self.posture_inv
        # srcK, dstK = fromPerspectiveCam.k, self.k
        # srcK_inv, dstK_inv = fromPerspectiveCam.k_inv, self.k_inv

        warp_uv_all_depth = []
        # d_dsts = []
        normalize_base = torch.tensor([self.imgW * 0.5, self.imgH * 0.5], dtype=dtype, device=device).unsqueeze(1)

        for nowD1 in d_srcs:
            planeFactorCamera = torch.tensor([0., 0., 1., -nowD1], dtype=dtype, device=device).unsqueeze(1)
            # planeFactorWorld = srcRT_inv @ planeFactorCamera
            planeFactorWorld = srcRT.T @ planeFactorCamera

            dstCameraAnchorPoint4, directionVectorU2World4, dstCameraAnchorPoint3, directionVectorU2World3 = dstPerspectiveCam.uv2WorldLine(U_dst)
            
            t_result = (-planeFactorWorld.T @ dstCameraAnchorPoint4) / (planeFactorWorld.T @ directionVectorU2World4 + eps)

            intersectPoint = dstCameraAnchorPoint3 + t_result * directionVectorU2World3
            U_dst2XYCameraDst4 = torch.ones(4, intersectPoint.shape[1], dtype=dtype, device=device)
            U_dst2XYCameraDst4[:3] = intersectPoint

            grid = self.world2uv(U_dst2XYCameraDst4, eps)
            grid = (grid - normalize_base) / normalize_base
            grid = grid.reshape(2, dstPerspectiveCam.imgH, dstPerspectiveCam.imgW)
            grid = grid.permute(1, 2, 0)
            warp_uv_all_depth.append(grid.float())

        allGrid = torch.stack(warp_uv_all_depth)
        
        return MyImageTranslator(allGrid, lambda x:torch.nn.functional.grid_sample(x, allGrid, mode, padding_mode, align_corners), len(d_srcs))

    def to(self, dstPerspectiveCam, d, eps=1e-8, mode='bilinear', padding_mode='zeros', align_corners=False):
        return self.toCam(dstPerspectiveCam, d, eps, mode, padding_mode, align_corners)

    def imread(self, url, cvt_color=None, interpolation=None, dtype=None, device=None):
        device, dtype = picDefaultDeviceAndDtype(device=device, dtype=dtype, defaultDtype=torch.float32)
        img1 = cv2.imread(url)
        img1 = self.resize(img1, interpolation=interpolation, dtype=dtype, device=device)

        if cvt_color is not None:
            img1 = cv2.cvtColor(img1, cvt_color)
        img1 = torch.tensor(img1, dtype=dtype, device=device)
        return img1

    def resize(self, img: np.ndarray, interpolation=None, dtype=None, device=None):
        device, dtype = picDefaultDeviceAndDtype(device=device, dtype=dtype, defaultDtype=torch.float32)
        if interpolation is None:
            interpolation = cv2.INTER_CUBIC
        img = cv2.resize(img, (int(self.imgW), int(self.imgH)), interpolation=interpolation)
        return img

    def formatIntrinsicParameter(fx, fy, s, y0, x0, dtype=None, device=None):
        device, dtype = picDefaultDeviceAndDtype(device=device, dtype=dtype)
        return torch.tensor([[fx, s, x0], [0, fy, y0], [0, 0, 1]], dtype=dtype, device=device)