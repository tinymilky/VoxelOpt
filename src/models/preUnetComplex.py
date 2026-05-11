import torch
import torch.nn as nn
from torch.nn import functional as F
from pathlib import Path
from models.universalmodel.unet import UNet3D

class preUnetComplex(nn.Module):

    def __init__(self, weights_path=None):
        super(preUnetComplex, self).__init__()

        self.unet_flow = UNet3D()
        if weights_path is None:
            weights_path = Path(__file__).resolve().parents[1] / 'unet.pth'
        fp = Path(weights_path)
        if not fp.exists():
            raise FileNotFoundError(f"Pretrained feature extractor weights not found: {fp}")

        load_dict = torch.load(fp, map_location='cpu')['net']
        new_dict = {}
        for key in load_dict.keys():
            if 'backbone' in key:
                new_dict[key.replace('module.backbone.','')] = load_dict[key]
            elif 'precls_conv' in key:
                new_dict[key.replace('module.','')] = load_dict[key]
        self.unet_flow.load_state_dict(new_dict)
        print('unet backbone loaded from:', fp)

        # freeze the weights
        for param in self.unet_flow.parameters():
            param.requires_grad = False

    def normalize(self, x_feas, y_feas):

        feas = torch.cat([x_feas, y_feas], dim=0)
        feas = F.normalize(feas, p=2, dim=1) # [b,c,h,w,d]

        tmp_feas = feas.permute(0,2,3,4,1).reshape(-1, feas.shape[1])
        min_feas = tmp_feas.min(dim=0,keepdim=True)[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        max_feas = tmp_feas.max(dim=0,keepdim=True)[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        feas = (feas - min_feas) / (max_feas - min_feas)
        x_feas, y_feas = feas.chunk(2, dim=0)

        return x_feas, y_feas

    def forward(self, x):

        feas = self.unet_flow(x)

        return feas
