import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.loss import _Loss

from utils.functions import VecInt, SpatialTransformer

class GaussianBlur3D(nn.Module):
    def __init__(self, 
        channels,
        sigma=1,
        kernel_size=0,
        is_half=False,
    ):
        super(GaussianBlur3D, self).__init__()
        self.channels = channels
        if kernel_size == 0:
            kernel_size = int(2.0 * sigma * 2 + 1)

        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size**2).view(kernel_size, kernel_size, kernel_size)
        y_grid = x_grid.transpose(0, 1)
        z_grid = x_grid.transpose(0, 2)
        xyz_grid = torch.stack([x_grid, y_grid, z_grid], dim=-1).float()

        mean = (kernel_size - 1) / 2.
        variance = sigma**2.

        gaussian_kernel = (1. / (2. * np.pi * variance) ** 1.5) * \
                          torch.exp(
                              -torch.sum((xyz_grid - mean) ** 2., dim=-1) /
                              (2 * variance)
                          )
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1, 1)
        if is_half:
            gaussian_kernel = gaussian_kernel.half()
        self.register_buffer('gaussian_kernel', gaussian_kernel)
        self.padding = kernel_size // 2

    def forward(self, x):
        blurred = F.conv3d(x, self.gaussian_kernel, padding=self.padding, groups=self.channels)
        return blurred


class GaussianMsgPass(nn.Module):

    def __init__(self, ks=1, sigma_low=0.1, sigma_high=0.5, is_half=False):

        super(GaussianMsgPass, self).__init__()

        self.ks = ks
        self.sigma_low = sigma_low
        self.sigma_high = sigma_high
        self.is_half = is_half

    def unfold_dim(self, x, dim):

        ks = self.ks

        if dim == 2:
            tx = F.pad(x, (0, 0, 0, 0, ks, ks))
            tx = tx.unfold(2, 2*ks+1, 1)
        elif dim == 3:
            tx = F.pad(x, (0, 0, ks, ks, 0, 0))
            tx = tx.unfold(3, 2*ks+1, 1)
        elif dim == 4:
            tx = F.pad(x, (ks, ks, 0, 0, 0, 0))
            tx = tx.unfold(4, 2*ks+1, 1)

        return tx

    def forward_weights(self, sigmas):
        # M: number of grid cells
        # N: totoal kernel size
        N = self.ks * 2 + 1
        M = sigmas.shape[0]

        x = torch.linspace(-(N // 2), N // 2, N, device=sigmas.device)
        x = x.view(1, N)
        weight = torch.exp(-torch.pow(x, 2) / (2 * torch.pow(sigmas, 2)))
        weights = weight / weight.sum(dim=1, keepdim=True)

        weights = weights.view(M,N).contiguous().unsqueeze(0).unsqueeze(0)

        if self.is_half:
            weights = weights.half()

        return weights # (1,1,M,N)

    def entropy2sigma(self, entropy):

        sigmas = entropy * (self.sigma_high - self.sigma_low) + self.sigma_low
        return sigmas.view(-1,1)

    def forward(self, x, entropy):

        b,c,h1,h2,h3 = x.shape
        sigmas = self.entropy2sigma(entropy)
        weights = self.forward_weights(sigmas)

        for dim in range(2, x.dim()):
            x = self.unfold_dim(x, dim).contiguous()
            x = x.view(b,c,-1,2*self.ks+1) 
            x = (x * weights).sum(-1).view(b,c,h1,h2,h3)

        return x

class costVolComplex(nn.Module):

    def __init__(self,
            img_size = '(192//2,160//2,256//2)',
            ks = '1',
            temp = '0.00005',
            sigma_cap = '0.5',
            is_half = '0',
            is_adaptive = '0',
        ):
        super(costVolComplex, self).__init__()

        self.img_size = eval(img_size)
        self.ks = int(ks)
        self.temp = float(temp)
        self.sigma_cap = float(sigma_cap)
        self.is_half = bool(int(is_half))
        self.is_adaptive = bool(int(is_adaptive))

        print('costVolComplex, img_size: %s, ks: %d, temp: %f, sigma: %f, is_half: %d, is_adaptive: %d' % (self.img_size, self.ks, self.temp, self.sigma_cap, self.is_half, self.is_adaptive))

        ih = self.is_half
        ia = self.is_adaptive
        self.transformers = nn.ModuleList([SpatialTransformer([s // 2**i for s in self.img_size], is_half=ih) for i in range(5)])
        self.transformer = SpatialTransformer(self.img_size, is_half=ih)

        self.integrates = nn.ModuleList([VecInt([s // 2**i for s in self.img_size], 7, is_half=ih) for i in range(5)])
        self.integrate = VecInt(self.img_size, 7, is_half=ih)

        self.convex_opt = convexOptimization(kernel_size=self.ks, temp=self.temp, sigma=self.sigma_cap, is_half=ih, is_adaptive=ia)

    def forward(self, source, target):
        '''
        source: [b, c, h, w, d]
        target: [b, c, h, w, d]
        '''
        if self.is_half:
            source = source.half()
            target = target.half()
        src_0, tgt_0 = source, target

        src_1 = F.interpolate(src_0, scale_factor=0.5, mode='trilinear', align_corners=True)
        tgt_1 = F.interpolate(tgt_0, scale_factor=0.5, mode='trilinear', align_corners=True)

        src_2 = F.interpolate(src_1, scale_factor=0.5, mode='trilinear', align_corners=True)
        tgt_2 = F.interpolate(tgt_1, scale_factor=0.5, mode='trilinear', align_corners=True)

        src_3 = F.interpolate(src_2, scale_factor=0.5, mode='trilinear', align_corners=True)
        tgt_3 = F.interpolate(tgt_2, scale_factor=0.5, mode='trilinear', align_corners=True)

        src_4 = F.interpolate(src_3, scale_factor=0.5, mode='trilinear', align_corners=True)
        tgt_4 = F.interpolate(tgt_3, scale_factor=0.5, mode='trilinear', align_corners=True)

        flow_4 = self.convex_opt(src_4,tgt_4)
        flow_4 = self.integrates[4](flow_4)
        flow_4 = F.interpolate(flow_4, scale_factor=2, mode='trilinear', align_corners=True) * 2
        warped_src_3 = self.transformers[3](src_3, flow_4)

        flow_3 = self.convex_opt(warped_src_3,tgt_3)
        flow_3 = self.integrates[3](flow_3)
        flow_3 = flow_3 + self.transformers[3](flow_4, flow_3)
        flow_3 = F.interpolate(flow_3, scale_factor=2, mode='trilinear', align_corners=True) * 2
        warped_src_2 = self.transformers[2](src_2, flow_3)

        flow_2 = self.convex_opt(warped_src_2,tgt_2)
        flow_2 = self.integrates[2](flow_2)
        flow_2 = flow_2 + self.transformers[2](flow_3, flow_2)
        flow_2 = F.interpolate(flow_2, scale_factor=2, mode='trilinear', align_corners=True) * 2
        warped_src_1 = self.transformers[1](src_1, flow_2)

        flow_1 = self.convex_opt(warped_src_1,tgt_1)
        flow_1 = self.integrates[1](flow_1)
        flow_1 = flow_1 + self.transformers[1](flow_2, flow_1)
        flow_1 = F.interpolate(flow_1, scale_factor=2, mode='trilinear', align_corners=True) * 2
        warped_src_0 = self.transformers[0](src_0, flow_1)

        flow_0 = self.convex_opt(warped_src_0,tgt_0)
        flow_0 = self.integrates[0](flow_0)
        pos_flow = flow_0 + self.transformers[0](flow_1, flow_0)

        return pos_flow

class convexOptimization(nn.Module):

    # the kernel size here is different from the one in conv3d
    # when kernle_size=2 here, the kernel size in conv3d is 5
    def __init__(self, 
        kernel_size=4,
        coeffs=[0.003, 0.01, 0.03, 0.1, 0.3, 1],
        sigma = 1,
        temp = 0.01,
        is_half = False,
        is_adaptive = False,
    ):
        super(convexOptimization, self).__init__()

        self.kernel_size = kernel_size
        self.offsets = self.generate_offsets(kernel_size*2+1) # (2, (ks*2+1)*(ks*2+1))
        self.offsets = self.offsets                           # (1,3,(ks*2+1)**3,1,1)
        self.coeffs = coeffs
        self.sigma = sigma
        self.temp = temp
        self.is_half = is_half
        self.is_adaptive = is_adaptive

        ks = self.kernel_size
        self.blur = GaussianMsgPass(ks=1, sigma_high=self.sigma, is_half=self.is_half)
        if self.is_half:
            self.offsets = self.offsets.half()

        tmp = torch.ones((ks*2+1)**3)
        tmp = F.softmax(tmp, dim=0)
        self.max_entropy = torch.sum(-tmp * torch.log(tmp), dim=0)
        self.blur_3 = GaussianBlur3D(3, sigma=0.5, is_half=self.is_half)
        self.blur_ks = GaussianBlur3D((ks*2+1)**3, sigma=0.5, is_half=self.is_half)

    @staticmethod
    def generate_offsets(ks):

        if type(ks) == int:
            ks = [ks, ks, ks]

        offset_z, offset_y, offset_x = torch.meshgrid(
            torch.arange(ks[0]),
            torch.arange(ks[1]),
            torch.arange(ks[2]),
            indexing='ij',
        )
        center = [k // 2 for k in ks]
        offset_z = offset_z - center[0]
        offset_y = offset_y - center[1]
        offset_x = offset_x - center[2]
        offsets = torch.stack([offset_z.flatten(), offset_y.flatten(), offset_x.flatten()], dim=0)
        # (3, (ks*2+1)**3)
        return offsets.float().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)

    def compute_patch_ncc(self, x, y):

        x_sum = x.sum(dim=1)
        y_sum = y.sum(dim=1)
        x2_sum = (x**2).sum(dim=1)
        y2_sum = (y**2).sum(dim=1)
        xy_sum = (x*y).sum(dim=1)
        u_x = x_sum / np.prod(x.shape)
        u_y = y_sum / np.prod(y.shape)

        cross = xy_sum - u_x * y_sum - u_y * x_sum + u_x * u_y * np.prod(x.shape)
        x_var = x2_sum - 2 * u_x * x_sum + u_x * u_x * np.prod(x.shape)
        y_var = y2_sum - 2 * u_y * y_sum + u_y * u_y * np.prod(y.shape)

        cc = cross * cross / (x_var * y_var + 1e-5)

        return 1. - cc

    def get_cost_vols_ncc(self, x, y):

        b,_,h,w,d = x.shape
        ks = self.kernel_size
        self.offsets = self.offsets.to(x.device)

        x = F.pad(x, (ks,)*3*2)
        x = x.unfold(2, ks*2+1, 1).unfold(3, ks*2+1, 1).unfold(4, ks*2+1, 1) # (b,(ncc_win*2+1)**3,h,w,d,(ks*2+1),(ks*2+1),(ks*2+1))

        cost_vols = torch.zeros(b,h,w,d,ks*2+1,ks*2+1,ks*2+1).to(x.device)
        if self.is_half:
            cost_vols = cost_vols.half()
        for i in range(ks*2+1):
            for j in range(ks*2+1):
                for k in range(ks*2+1):
                    corr = (x[...,i,j,k] - y).abs().mean(1) # (b,h,w,d)
                    # corr = self.compute_patch_ncc(x[...,i,j,k], y) # (b,h,w,d)
                    cost_vols[...,i,j,k] = -corr

        cost_vols = cost_vols.view(b,h,w,d,-1).permute(0,4,1,2,3).contiguous().unsqueeze(1) # (b,1,(ks*2+1)**3,h,w,d)

        return cost_vols

    def get_entropy(self, cost_vols):

        # cost_vols: ((ks*2+1)**3,h,w,d)

        prob_vols = F.softmax(cost_vols*10, dim=0)
        prob_vols = torch.clamp(prob_vols, 1e-6, 1.0)
        prob_vols = prob_vols / torch.sum(prob_vols, dim=0, keepdim=True)
        entropy = torch.sum(-prob_vols * torch.log(prob_vols), dim=0)
        entropy = (entropy - torch.min(entropy)) / (torch.max(entropy) - torch.min(entropy))
        entropy = torch.log2(entropy+1)

        return entropy

    def forward(self, src, tgt):
        '''
        x,y: (b,c,h,w)
        '''

        cost_vols = self.get_cost_vols_ncc(src, tgt)
        entropy = self.get_entropy(cost_vols.squeeze())
        if self.is_adaptive:
            cost_vols = self.blur(cost_vols.squeeze(1),entropy).unsqueeze(1)
            cost_vols = self.blur(cost_vols.squeeze(1),entropy).unsqueeze(1)
            cost_vols = self.blur(cost_vols.squeeze(1),entropy).unsqueeze(1)
        else:
            cost_vols = self.blur_ks(cost_vols.squeeze(1)).unsqueeze(1)
            cost_vols = self.blur_ks(cost_vols.squeeze(1)).unsqueeze(1)
            cost_vols = self.blur_ks(cost_vols.squeeze(1)).unsqueeze(1)

        prob_vols = F.softmax(cost_vols/self.temp, dim=2) # (b,1,(ks*2+1)**2,h,w,d)
        prob_vols = (prob_vols == prob_vols.max(dim=2, keepdim=True)[0])
        flow = torch.sum(prob_vols*self.offsets, dim=2) # (b,3,h,w,d)
        offsets_t = self.offsets # (1,3,(ks*2+1)**3,1,1)

        if self.is_adaptive:
            flow = self.blur(flow, entropy)
            flow = self.blur(flow, entropy)
            flow = self.blur(flow, entropy)
        else:
            flow = self.blur_3(flow)
            flow = self.blur_3(flow)
            flow = self.blur_3(flow)


        coefs = [0.003, 0.01, 0.03, 0.1, 0.3, 1]
        for i, coeff in enumerate(coefs):
            coupled_cost_vols = cost_vols - coeff * (flow.unsqueeze(2) - offsets_t).pow(2).sum(1).unsqueeze(1) # (b,1, (ks*2+1)**3,h,w,d)
            prob_vols = F.softmax(coupled_cost_vols/self.temp, dim=2)
            prob_vols = (prob_vols == prob_vols.max(dim=2, keepdim=True)[0])
            flow = torch.sum(prob_vols*self.offsets, dim=2)

            if self.is_adaptive:
                flow = self.blur(flow, entropy)
                flow = self.blur(flow, entropy)
                flow = self.blur(flow, entropy)
            else:
                flow = self.blur_3(flow)
                flow = self.blur_3(flow)
                flow = self.blur_3(flow)

        return flow
