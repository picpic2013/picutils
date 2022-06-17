import os
import json
import numpy as np
from picwarping import MyPosture, formatIntrinsicParameter
from googleutils.utils import _WorldFromCameraFromViewDict

def loadScene(baseUrl, sceneId=0):
    scenes = None
    with open(os.path.join(baseUrl, 'models.json')) as f:
        scenes = json.loads(f.read())

    cameras = [] # [{'pos': ..., 'intr': ..., 'img': ...}]

    for cam in scenes[sceneId]:
        A = _WorldFromCameraFromViewDict(cam)
        
        pos = MyPosture.fromMat44(A)

        cx, cy = cam['principal_point']
        imgW, imgH = cam['width'], cam['height']
        fx, fy = cam['focal_length'], cam['focal_length']
        intrin = formatIntrinsicParameter(fx, fy, 0, cy, cx)

        cameras.append({
            'pos': pos.getPosture(), 
            'intr': intrin, 
            'img': os.path.join(baseUrl, cam['relative_path']), 
            'imgH': int(imgH), 
            'imgW': int(imgW)
        })
    return cameras