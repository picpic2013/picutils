import torch

class MyImageTranslator:
    def __init__(self, grid, transFunction, d_length):
        self.grid = grid
        self.transFunction = transFunction
        self.d_length = d_length
        
    def __call__(self, img: torch.Tensor):

        permute = False

        if len(img.shape) == 3:
            img = img.expand(self.d_length, *img.shape)
        if img.shape[3] == 3:
            img = img.permute([0, 3, 1, 2])
            permute = True

        img.type(self.grid.dtype).to(self.grid.device)
        img = self.transFunction(img)

        if permute:
            return img.permute([0, 2, 3, 1])
        return img