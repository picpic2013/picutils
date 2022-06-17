from __future__ import annotations

import torch

from picutils.utils import picDefaultDeviceAndDtype, angle2Rad, quaternion2Mat
from picutils.googleutils.transformations import rotation_matrix, translation_matrix, euler_matrix

class MyPosture:
    def __init__(self, posture=None, dtype=None, device=None):
        '''
        @param posture: Xc = posture @ Xw, shape: [4, 4]
        '''
        device, dtype = picDefaultDeviceAndDtype(device=device, dtype=dtype)
        if posture is None:
            posture = torch.eye(4)    
        if type(posture) == torch.Tensor:
            posture = posture.type(dtype).to(device)
        else:
            posture = torch.tensor(posture, dtype=dtype, device=device)


        self.pos = posture

    def getPosture(self) -> torch.Tensor:
        return self.pos

    def rotate(self, angle, axis, point=None, rad=True) -> MyPosture:
        if rad == False:
            angle = angle2Rad(angle)
        rr = rotation_matrix(angle, axis, point)
        if type(rr) != torch.Tensor:
            rr = torch.tensor(rr, dtype=self.pos.dtype)
        self.pos = rr @ self.pos
        return self

    def applyTransform(self, A: torch.Tensor) -> MyPosture:
        assert A.shape == torch.Size([4, 4])
        self.pos = A.type(self.pos.type).to(self.pos.device) @ self.pos
        return self
    
    def translate(self, X) -> MyPosture:
        '''
        x shape: [3]
        '''
        tt = translation_matrix(X)
        if type(tt) != torch.Tensor:
            tt = torch.tensor(tt, dtype=self.pos.dtype)
        self.pos = tt @ self.pos
        return self

    def inv(self, dtype=None, device=None) -> MyPosture:
        device, dtype = picDefaultDeviceAndDtype(device=device, dtype=dtype)
        newPos = MyPosture(torch.linalg.inv(self.getPosture().double()), dtype, device)
        return newPos

    @staticmethod
    def fromMat33(R, t=None, dtype=None, device=None) -> MyPosture:
        '''
        R: 3 x 3
        t: 3 x 1
        '''
        device, dtype = picDefaultDeviceAndDtype(device=device, dtype=dtype)
        if type(R) != torch.Tensor:
            R = torch.tensor(R, dtype=torch.float64)
        if t is None:
            t = torch.zeros(3, 1)
        if type(t) != torch.Tensor:
            t = torch.tensor(t, dtype=torch.float64)

        assert R.shape == torch.Size([3, 3]) and t.shape == torch.Size([3, 1])

        A = torch.zeros(4, 4, dtype=torch.float64)
        A[:3, :3] = R.cpu().double()
        A[:3,  3] = t.cpu().double().squeeze(1)
        A[ 3,  3] = 1
        return MyPosture(A, dtype, device)

    @staticmethod
    def fromRotateAngle(rUp0=None, rRight0=None, rFront0=None, t0=None, rad=True, dtype=None, device=None) -> MyPosture:
        '''
        t0: [x0, y0, z0]
        '''
        device, dtype = picDefaultDeviceAndDtype(device=device, dtype=dtype)
        if rUp0 is None:
            rUp0 = 0
        if rRight0 is None:
            rRight0 = 0
        if rFront0 is None:
            rFront0 = 0
        if t0 is None:
            t0 = [0, 0, 0]

        if rad == False:
            rUp0, rRight0, rFront0 = (angle2Rad(_) for _ in (rUp0, rRight0, rFront0))
        
        R = euler_matrix(rUp0, rRight0, rFront0, 'rxyz')
        trans  = translation_matrix(t0)
        pos = torch.tensor(R @ trans, dtype=torch.float64)

        return MyPosture(pos, dtype, device)

    @staticmethod
    def fromQuaternions(x, y, z, w, t, device=None, dtype=None) -> MyPosture:
        device, dtype = picDefaultDeviceAndDtype(device=device, dtype=dtype)
        pos = quaternion2Mat(x, y, z, w)
        return MyPosture.fromMat33(pos, t, dtype, device)

    def __str__(self) -> str:
        return str(self.pos)