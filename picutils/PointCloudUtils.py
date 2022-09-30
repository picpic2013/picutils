import torch
import numpy as np
from plyfile import PlyData, PlyElement
from typing import Tuple

from picutils.RecursiveWarper import make_multi_return_recursive_func
from picutils.MyPerspectiveCamera import MyPerspectiveCamera

@make_multi_return_recursive_func
def generatePointCloud(img: torch.Tensor, dep: torch.Tensor, cam: MyPerspectiveCamera, mask=None) -> Tuple[torch.Tensor]:
    '''
    @param img:  [h, w, 3] (RGB)
    @param dep:  [h, w]    (float32)
    @param cam:  MyPerspectiveCamera
    @param mask: torch.Tensor | None
    '''

    device = dep.device
    dtype = dep.dtype

    img = img.to(device)

    height, width = dep.shape[:2]
    u, v = np.meshgrid(np.arange(0, width), np.arange(0, height))
    u, v = torch.from_numpy(u).to(device).type(dtype), torch.from_numpy(v).to(device).type(dtype)

    if mask is None:
        mask = torch.ones_like(dep).to(device).bool()

    valid_points = mask
    # print("valid_points", valid_points.mean())
    u, v, depth, color = u[valid_points], v[valid_points], dep[valid_points], img[valid_points]
    uvs = torch.stack([u, v])

    _, _, base, direction = cam.uv2WorldLine(uvs)
    xyz_world = base + depth * direction

    return xyz_world.permute(1, 0), color.type(torch.uint8)

def savePointCloud(fileUrl: str, vertexs, vertex_colors) -> None:
    '''
    @param fileUrl:       file url
    @param vertexs:       vertex 3D cords  [n, 3]
    @param vertex_colors: color in [RGB]   [n, 3]
    '''

    if type(vertexs) != list:
        vertexs = [vertexs]
    if type(vertex_colors) != list:
        vertex_colors = [vertex_colors]

    _vertexs = []
    _vertex_colors = []
    for ver, col in zip(vertexs, vertex_colors):
        assert type(ver) == np.ndarray or type(ver) == torch.Tensor
        assert type(col) == np.ndarray or type(col) == torch.Tensor
        assert col.dtype == np.uint8 or col.dtype == torch.uint8

        if type(ver) == np.ndarray:
            _vertexs.append(ver)
        else:
            _vertexs.append(ver.cpu().numpy())
        
        if type(col) == np.ndarray:
            _vertex_colors.append(col)
        else:
            _vertex_colors.append(col.cpu().numpy())

    vertexs = _vertexs
    vertex_colors = _vertex_colors

    # save point cloud
    vertexs = np.concatenate(vertexs, axis=0)
    vertex_colors = np.concatenate(vertex_colors, axis=0)
    vertexs = np.array([tuple(v) for v in vertexs], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    vertex_colors = np.array([tuple(v) for v in vertex_colors], dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    # vertex_colors = np.array([tuple(v) for v in vertex_colors], dtype=[('blue', 'u1'), ('green', 'u1'), ('red', 'u1')])

    vertex_all = np.empty(len(vertexs), vertexs.dtype.descr + vertex_colors.dtype.descr)
    for prop in vertexs.dtype.names:
        vertex_all[prop] = vertexs[prop]
    for prop in vertex_colors.dtype.names:
        vertex_all[prop] = vertex_colors[prop]

    el = PlyElement.describe(vertex_all, 'vertex')
    PlyData([el]).write(fileUrl)