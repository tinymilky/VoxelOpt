import re
import os
import glob
import scipy
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.nn import functional as F
from collections import deque, OrderedDict
from utils.surface_distance import compute_robust_hausdorff, compute_surface_distances

def extract_pixel_features(x, ncc_win=1):

    if ncc_win == 0:
        return x

    b,_,h,w,d = x.shape

    x = F.pad(x, (ncc_win,)*3*2) # (b,1,h+2*ncc_win,w+2*ncc_win,d+2*ncc_win)
    x = x.unfold(2, ncc_win*2+1, 1).unfold(3, ncc_win*2+1, 1).unfold(4, ncc_win*2+1, 1) # (b,1,h,w,d,(ncc_win*2+1)**3)
    x = x.contiguous().view(b,h,w,d,-1).permute(0,4,1,2,3) # (b,(ncc_win*2+1)**3,h,w,d)

    return x

def get_downsampled_images(img, n_downs=4, mode='trilinear', n_cs=1):

    if n_cs > 0:
        blur = GaussianBlur3D(n_cs, sigma=1).to(img.device)
    out_imgs = [img]
    for _ in range(n_downs):
        if n_cs > 0:
            img = blur(img)
        img = F.interpolate(img, scale_factor=0.5, mode=mode, align_corners=True)
        out_imgs.append(img)

    return out_imgs


def get_downsampled_images_2d(img, n_downs=4, mode='bilinear', n_cs=3):

    if n_cs > 0:
        blur = GaussianBlur2D(n_cs, sigma=1).to(img.device)
    out_imgs = [img]
    for _ in range(n_downs):
        if n_cs > 0:
            img = blur(img)
        img = F.interpolate(img, scale_factor=0.5, mode=mode, align_corners=True)
        out_imgs.append(img)

    return out_imgs

def erode_3d_tensor(input_tensor, kernel_size=3, threshold=1):

    kernel = torch.ones((1, 1, kernel_size, kernel_size, kernel_size), dtype=input_tensor.dtype, device=input_tensor.device)
    kernel = kernel.float().to(input_tensor.device)

    conv_result = F.conv3d(input_tensor, kernel, padding=kernel_size//2)
    eroded_tensor = conv_result >= (kernel_size ** 3) * threshold
    eroded_tensor = eroded_tensor.float()

    return eroded_tensor

def dilate_3d_tensor(input_tensor, kernel_size=3, threshold=1):

    kernel = torch.ones((1, 1, kernel_size, kernel_size, kernel_size), dtype=torch.float32, device=input_tensor.device)

    if input_tensor.dim() == 3:
        input_tensor_unsqueezed = input_tensor.unsqueeze(0).unsqueeze(0)
    else:
        input_tensor_unsqueezed = input_tensor

    conv_result = F.conv3d(input_tensor_unsqueezed, kernel, padding=kernel_size//2)

    dilated_tensor = conv_result >= threshold

    if input_tensor.dim() == 3:
        dilated_tensor = dilated_tensor.float().squeeze(0).squeeze(0)
    else:
        dilated_tensor = dilated_tensor.float()

    return dilated_tensor

def generate_all_lbls(x, y, x_seg, y_seg):
    '''
    x,y: (1,1,96,80,128)
    x_seg, y_seg: (1,1,96,80,128)
    '''

    adj_ma = pd.read_csv('adjacency_matrix.csv')
    adj_ma = adj_ma.to_numpy()
    tiny_lbls = [4,10,15,16,18,19,23,29,32,34]

    x_lbl, y_lbl = torch.unique(x_seg), torch.unique(y_seg)
    lbls = torch.unique(torch.cat((x_lbl, y_lbl)))
    lbls = torch.unique(lbls)

    xs, ys = [], []
    x_segs, y_segs = [], []
    for lbl in lbls:
        x_lbl = (x_seg == lbl).float()
        y_lbl = (y_seg == lbl).float()

        # if lbl in tiny_lbls:
        #     x_lbl = dilate_3d_tensor(x_lbl, kernel_size=3, threshold=1)
        #     y_lbl = dilate_3d_tensor(y_lbl, kernel_size=3, threshold=1)

        #     for idx in np.where(adj_ma[lbl, :] == 1)[0]:
        #         adj_lbl = adj_ma[lbl, idx]
        #         adj_x_lbl = (x_seg == adj_lbl).float()
        #         adj_y_lbl = (y_seg == adj_lbl).float()
        #         vol_cur = x_lbl.sum()+y_lbl.sum()
        #         vol_adj = adj_x_lbl.sum()+adj_y_lbl.sum()
        #         if vol_cur > vol_adj:
        #             x_overlap = (x_lbl * adj_x_lbl)
        #             y_overlap = (y_lbl * adj_y_lbl)
        #             x_lbl = x_lbl - x_overlap
        #             y_lbl = y_lbl - y_overlap

        x_segs.append(x_lbl)
        y_segs.append(y_lbl)
        xs.append(x * x_lbl)
        ys.append(y * y_lbl)

    xs = torch.cat(xs, dim=0)
    ys = torch.cat(ys, dim=0)
    x_segs = torch.cat(x_segs, dim=0)
    y_segs = torch.cat(y_segs, dim=0)

    lbls = lbls.cpu().numpy().tolist()

    return xs, x_segs, ys, y_segs, lbls

def random_lbl_select(x, y, x_seg, y_seg):
    '''
    x,y: b,c,h,w,d, input images
    x_seg, y_seg: b,1,h,w,d,, input segmentation masks, values in [0,1,...,num_classes-1]
    This function randomly selects a label that exists in both x_seg and y_seg, and returns the corresponding binary x_seg and y_seg, and the corresponding x and y masked by the binary segmentation masks.
    '''
    adj_ma = pd.read_csv('adjacency_matrix.csv')
    adj_ma = adj_ma.to_numpy()
    tiny_lbls = [4,10,15,16,18,19,23,29,32,34]

    bs = x.size(0)
    x_out, y_out = [], []
    x_seg_out, y_seg_out = [], []
    selected_lbls = []
    for idx in range(bs):
        x_, x_seg_ = x[idx], x_seg[idx]
        y_, y_seg_ = y[idx], y_seg[idx]
        x_lbl, y_lbl = torch.unique(x_seg_), torch.unique(y_seg_)
        lbls = torch.unique(torch.cat((x_lbl, y_lbl)))
        lbls = torch.unique(lbls)
        selected_lbl = lbls[torch.randint(0, len(lbls), (1,))] #exclude background
        x_seg_c = (x_seg_ == selected_lbl).float()
        y_seg_c = (y_seg_ == selected_lbl).float()
        lbl = int(selected_lbl.item())

        # if lbl in tiny_lbls:
        #     x_seg_c = dilate_3d_tensor(x_seg_c.unsqueeze(0), kernel_size=3, threshold=1).squeeze(0)
        #     y_seg_c = dilate_3d_tensor(y_seg_c.unsqueeze(0), kernel_size=3, threshold=1).squeeze(0)

        #     for idx in np.where(adj_ma[lbl, :] == 1)[0]:
        #         adj_lbl = adj_ma[lbl, idx]
        #         adj_x_lbl = (x_seg_ == adj_lbl).float()
        #         adj_y_lbl = (y_seg_ == adj_lbl).float()
        #         vol_cur = x_seg_c.sum()+y_seg_c.sum()
        #         vol_adj = adj_x_lbl.sum()+adj_y_lbl.sum()
        #         if vol_cur > vol_adj:
        #             x_overlap = (x_seg_c * adj_x_lbl)
        #             y_overlap = (y_seg_c * adj_y_lbl)
        #             x_seg_c = x_seg_c - x_overlap
        #             y_seg_c = y_seg_c - y_overlap
        #             adj_x_lbl = dilate_3d_tensor(adj_x_lbl.unsqueeze(0), kernel_size=3, threshold=1).squeeze(0)
        #             adj_y_lbl = dilate_3d_tensor(adj_y_lbl.unsqueeze(0), kernel_size=3, threshold=1).squeeze(0)
        #             x_overlap = (x_seg_c * adj_x_lbl)
        #             y_overlap = (y_seg_c * adj_y_lbl)
        #             x_seg_c = x_seg_c - x_overlap
        #             y_seg_c = y_seg_c - y_overlap

        selected_lbls.append(lbl)
        x_ = x_ * x_seg_c
        y_ = y_ * y_seg_c
        x_out.append(x_)
        y_out.append(y_)
        x_seg_out.append(x_seg_c)
        y_seg_out.append(y_seg_c)

    x_out = torch.stack(x_out, dim=0)
    y_out = torch.stack(y_out, dim=0)
    x_seg_out = torch.stack(x_seg_out, dim=0)
    y_seg_out = torch.stack(y_seg_out, dim=0)

    return x_out, x_seg_out, y_out, y_seg_out, selected_lbls

def random_linked_lbl_select(x, y, x_seg, y_seg, n_adjs=5):
    adj_ma = pd.read_csv('adjacency_matrix.csv')
    adj_ma = adj_ma.to_numpy()

    bs = x.size(0)
    if bs != 1:
        raise ValueError('Batch size should be 1 for generating linked labels')

    x_out, y_out = [], []
    x_seg_out, y_seg_out = [], []

    x_, x_seg_ = x, x_seg
    y_, y_seg_ = y, y_seg
    x_lbl, y_lbl = torch.unique(x_seg_), torch.unique(y_seg_)
    lbls = torch.unique(torch.cat((x_lbl, y_lbl)))
    lbls = torch.unique(lbls)
    selected_lbl = lbls[torch.randint(1, len(lbls), (1,))] 

    tgt_lbl = int(selected_lbl.item())
    adjs = adj_ma[tgt_lbl, :]
    if np.sum(adjs) < n_adjs:
        adjs = adjs*0+1
        adjs[tgt_lbl] = 0
    adjs[adjs == 0] = 0
    adj_lbls = np.where(adjs == 1)[0]
    # ramdomly shuffle the adjacent labels
    adj_lbls = np.random.permutation(adj_lbls)[:n_adjs]
    all_lbls = np.concatenate((adj_lbls, [tgt_lbl]), axis=0)

    for lbl in all_lbls:
        x_seg_ = (x_seg_ == lbl).float()
        y_seg_ = (y_seg_ == lbl).float()
        x_ = x_ * x_seg_
        y_ = y_ * y_seg_
        x_out.append(x_)
        y_out.append(y_)
        x_seg_out.append(x_seg_)
        y_seg_out.append(y_seg_)

    x_out = torch.cat(x_out, dim=0)
    y_out = torch.cat(y_out, dim=0)
    x_seg_out = torch.cat(x_seg_out, dim=0)
    y_seg_out = torch.cat(y_seg_out, dim=0)

    return x_out, x_seg_out, y_out, y_seg_out, all_lbls

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.vals = []
        self.std = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.vals.append(val)
        self.std = np.std(self.vals)

class SpatialTransformer(nn.Module):

    def __init__(self, size, mode='bilinear', is_half=False):
        super().__init__()

        self.mode = mode
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors, indexing='ij')
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)
        if is_half:
            grid = grid.half()
        self.register_buffer('grid', grid)

    def forward(self, src, flow, is_grid_out=False, mode=None, align_corners=True):

        new_locs = self.grid + flow
        shape = flow.shape[2:]

        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        if mode is None:
            out = F.grid_sample(src, new_locs, align_corners=align_corners, mode=self.mode)
        else:
            out = F.grid_sample(src, new_locs, align_corners=align_corners, mode=mode)

        if is_grid_out:
            return out, new_locs
        return out

class registerSTModel(nn.Module):

    def __init__(self, img_size=(64, 256, 256), mode='bilinear', is_half=False):
        super(registerSTModel, self).__init__()

        self.spatial_trans = SpatialTransformer(img_size, mode, is_half)

    def forward(self, img, flow, is_grid_out=False, align_corners=True):

        out = self.spatial_trans(img,flow,is_grid_out=is_grid_out,align_corners=align_corners)

        return out

class VecInt(nn.Module):

    def __init__(self, inshape, nsteps, is_half=False):
        super().__init__()
        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps

        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape, is_half=is_half)

    def forward(self, vec):

        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)

        return vec

def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power(1 - (epoch) / MAX_EPOCHES, power), 8)
    x = param_group['lr']
    return x

def dice_eval(y_pred, y_true, num_cls, exclude_background=True, output_individual=False):

    y_pred = nn.functional.one_hot(y_pred, num_classes=num_cls)
    y_pred = torch.squeeze(y_pred, 1)
    y_pred = y_pred.permute(0, 4, 1, 2, 3).contiguous()
    y_true = nn.functional.one_hot(y_true, num_classes=num_cls)
    y_true = torch.squeeze(y_true, 1)
    y_true = y_true.permute(0, 4, 1, 2, 3).contiguous()
    intersection = y_pred * y_true
    intersection = intersection.sum(dim=[2, 3, 4])
    union = y_pred.sum(dim=[2, 3, 4]) + y_true.sum(dim=[2, 3, 4])
    dsc = (2.*intersection) / (union + 1e-5)

    dscs = []
    if output_individual:
        dscs = [torch.mean(dsc[:,x:x+1]) for x in range(1,num_cls)]

    if exclude_background:
        out = [torch.mean(torch.mean(dsc[:,1:], dim=1))] + dscs
    else:
        out = [torch.mean(torch.mean(dsc, dim=1))] + dscs

    if len(out) == 1:
        return out[0]
    return tuple(out)

def dice_eval_2D(y_pred, y_true, num_cls, exclude_background=True, output_individual=False):

    y_pred = nn.functional.one_hot(y_pred, num_classes=num_cls)
    y_pred = torch.squeeze(y_pred, 1)
    y_pred = y_pred.permute(0, 3, 1, 2).contiguous()
    y_true = nn.functional.one_hot(y_true, num_classes=num_cls)
    y_true = torch.squeeze(y_true, 1)
    y_true = y_true.permute(0, 3, 1, 2).contiguous()
    intersection = y_pred * y_true
    intersection = intersection.sum(dim=[2, 3])
    union = y_pred.sum(dim=[2, 3]) + y_true.sum(dim=[2, 3])
    dsc = (2.*intersection) / (union + 1e-5)

    dscs = []
    if output_individual:
        dscs = [torch.mean(dsc[:,x:x+1]) for x in range(1,num_cls)]

    if exclude_background:
        out = [torch.mean(torch.mean(dsc[:,1:], dim=1))] + dscs
    else:
        out = [torch.mean(torch.mean(dsc, dim=1))] + dscs

    if len(out) == 1:
        return out[0]
    return tuple(out)


def convert_pytorch_grid2scipy(grid):

    _, H, W, D = grid.shape
    grid_x = (grid[0, ...] + 1) * (D -1)/2
    grid_y = (grid[1, ...] + 1) * (W -1)/2
    grid_z = (grid[2, ...] + 1) * (H -1)/2

    grid = np.stack([grid_z, grid_y, grid_x])

    identity_grid = np.meshgrid(np.arange(H), np.arange(W), np.arange(D), indexing='ij')
    grid = grid - identity_grid

    return grid

# def dice_missing_eval(y_pred, y_true, num_cls, exclude_background=True, output_individual=False):

#     for i in range(1, num_cls):
#         if torch.sum(y_true == i) == 0:
#             y_true = y_true + (y_pred == i).float()

def dice_binary(pred, truth, k = 1):
    truth[truth!=k]=0
    pred[pred!=k]=0
    truth=truth/k
    pred=pred/k
    intersection = np.sum(pred[truth==1.0]) * 2.0
    dice = intersection / (np.sum(pred) + np.sum(truth)+1e-7)

    return dice

def compute_tre(x, y, spacing):
    return np.linalg.norm((x - y) * spacing, axis=1)


class modelSaver():

    def __init__(self, save_path, save_freq, n_checkpoints = 10):

        self.save_path = save_path
        self.save_freq = save_freq
        self.best_score = -1e6
        self.best_loss = 1e6
        self.n_checkpoints = n_checkpoints
        self.epoch_fifos = deque([])
        self.score_fifos = deque([])
        self.loss_fifos = deque([])

        self.initModelFifos()

    def initModelFifos(self):

        epoch_epochs = []
        score_epochs = []
        loss_epochs  = []

        file_list = glob.glob(os.path.join(self.save_path, '*epoch*.pth'))
        if file_list:
            for file_ in file_list:
                file_name = "net_epoch_(.*)_score_.*.pth.*"
                result = re.findall(file_name, file_)
                if(result):
                    epoch_epochs.append(int(result[0]))

                file_name = "best_score_.*_net_epoch_(.*).pth.*"
                result = re.findall(file_name, file_)
                if(result):
                    score_epochs.append(int(result[0]))

                file_name = "best_loss_.*_net_epoch_(.*).pth.*"
                result = re.findall(file_name, file_)
                if(result):
                    loss_epochs.append(int(result[0]))

        score_epochs.sort()
        epoch_epochs.sort()
        loss_epochs.sort()

        if file_list:
            for file_ in file_list:
                for epoch_epoch in epoch_epochs:
                    file_name = "net_epoch_" + str(epoch_epoch) + "_score_.*.pth.*"
                    result = re.findall(file_name, file_)
                    if(result):
                        self.epoch_fifos.append(result[0])

                for score_epoch in score_epochs:
                    file_name = "best_score_.*_net_epoch_" + str(score_epoch) +".pth.*"
                    result = re.findall(file_name, file_)
                    if(result):
                        self.score_fifos.append(result[0])

                for loss_epoch in loss_epochs:
                    file_name = "best_loss_.*_net_epoch_" + str(loss_epoch) +".pth.*"
                    result = re.findall(file_name, file_)
                    if(result):
                        self.loss_fifos.append(result[0])

        print("----->>>> BEFORE: epoch_fifos length: %d, score_fifos_length: %d, loss_fifos_length: %d" % (len(self.epoch_fifos), len(self.score_fifos), len(self.loss_fifos)))

        self.updateFIFOs()

        print("----->>>> AFTER: epoch_fifos length: %d, score_fifos_length: %d, loss_fifos_length: %d" % (len(self.epoch_fifos), len(self.score_fifos), len(self.loss_fifos)))

    def saveModel(self, model, epoch, avg_score, loss=None):

        torch.save(model.state_dict(), os.path.join(self.save_path, 'net_latest.pth'))

        if epoch % self.save_freq == 0:

            file_name = ('net_epoch_%d_score_%.4f.pth' % (epoch, avg_score))
            self.epoch_fifos.append(file_name)

            save_path = os.path.join(self.save_path, file_name)
            torch.save(model.state_dict(), save_path)

        if avg_score >= self.best_score:

            self.best_score = avg_score
            file_name = ('best_score_%.4f_net_epoch_%d.pth' % (avg_score, epoch))
            self.score_fifos.append(file_name)

            save_path = os.path.join(self.save_path, file_name)
            torch.save(model.state_dict(), save_path)

        if loss is not None and loss <= self.best_loss:

            self.best_loss = loss
            file_name = ('best_loss_%.4f_net_epoch_%d.pth' % (loss, epoch))
            self.loss_fifos.append(file_name)

            save_path = os.path.join(self.save_path, file_name)
            torch.save(model.state_dict(), save_path)

        self.updateFIFOs()

    def updateFIFOs(self):

        while(len(self.epoch_fifos) > self.n_checkpoints):
            file_name = self.epoch_fifos.popleft()
            file_path = os.path.join(self.save_path, file_name)
            if os.path.exists(file_path):
                os.remove(file_path)

        while(len(self.score_fifos) > self.n_checkpoints):
            file_name = self.score_fifos.popleft()
            file_path = os.path.join(self.save_path, file_name)
            if os.path.exists(file_path):
                os.remove(file_path)

        while(len(self.loss_fifos) > self.n_checkpoints):
            file_name = self.loss_fifos.popleft()
            file_path = os.path.join(self.save_path, file_name)
            if os.path.exists(file_path):
                os.remove(file_path)

def convert_state_dict(state_dict, is_multi = False):

    new_state_dict = OrderedDict()

    if is_multi:
        if next(iter(state_dict)).startswith("module."):
            return state_dict  # abort if dict is a DataParallel model_state

        for k, v in state_dict.items():
            name = 'module.' + k  # add `module.`
            new_state_dict[name] = v
    else:

        if not next(iter(state_dict)).startswith("module."):
            return state_dict  # abort if dict is not a DataParallel model_state

        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v

    return new_state_dict

def jacobian_determinant(disp):
    _, _, H, W, D = disp.shape

    gradx  = np.array([-0.5, 0, 0.5]).reshape(1, 3, 1, 1)
    grady  = np.array([-0.5, 0, 0.5]).reshape(1, 1, 3, 1)
    gradz  = np.array([-0.5, 0, 0.5]).reshape(1, 1, 1, 3)

    gradx_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], gradx, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :, :], gradx, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], gradx, mode='constant', cval=0.0)], axis=1)

    grady_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], grady, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :, :], grady, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], grady, mode='constant', cval=0.0)], axis=1)

    gradz_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], gradz, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :, :], gradz, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], gradz, mode='constant', cval=0.0)], axis=1)

    grad_disp = np.concatenate([gradx_disp, grady_disp, gradz_disp], 0)

    jacobian = grad_disp + np.eye(3, 3).reshape(3, 3, 1, 1, 1)
    jacobian = jacobian[:, :, 2:-2, 2:-2, 2:-2]
    jacdet = jacobian[0, 0, :, :, :] * (jacobian[1, 1, :, :, :] * jacobian[2, 2, :, :, :] - jacobian[1, 2, :, :, :] * jacobian[2, 1, :, :, :]) -\
             jacobian[1, 0, :, :, :] * (jacobian[0, 1, :, :, :] * jacobian[2, 2, :, :, :] - jacobian[0, 2, :, :, :] * jacobian[2, 1, :, :, :]) +\
             jacobian[2, 0, :, :, :] * (jacobian[0, 1, :, :, :] * jacobian[1, 2, :, :, :] - jacobian[0, 2, :, :, :] * jacobian[1, 1, :, :, :])

    return jacdet

def jacobian_determinant_2d(disp):
    # Assuming disp has shape [2, H, W], representing displacement in two dimensions
    H, W = disp.shape[1], disp.shape[2]

    # Define gradients for x and y directions
    gradx = np.array([-0.5, 0, 0.5]).reshape(1, 3, 1)
    grady = np.array([-0.5, 0, 0.5]).reshape(1, 1, 3)

    # Compute gradients of displacement components
    gradx_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :], gradx, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :], gradx, mode='constant', cval=0.0)], axis=1)

    grady_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :], grady, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :], grady, mode='constant', cval=0.0)], axis=1)

    # Stack gradients to form the Jacobian matrix for each point
    grad_disp = np.concatenate([gradx_disp, grady_disp], 0)
    # Add identity matrix since the displacement is relative to an identity grid
    jacobian = grad_disp + np.eye(2).reshape(2, 2, 1, 1)

    # Crop the edges to reduce edge effects
    # Note: Adjust this line if you need a different cropping strategy
    jacobian_cropped = jacobian[:, :, 2:-2, 2:-2]

    # Calculate determinant of the 2x2 Jacobian at each point
    jacdet = jacobian_cropped[0, 0, :, :] * jacobian_cropped[1, 1, :, :] - jacobian_cropped[0, 1, :, :] * jacobian_cropped[1, 0, :, :]

    return jacdet

def compute_HD95(moving, fixed, moving_warped,num_classes=14,spacing=np.ones(3)):

    hd95 = 0
    count = 0
    for i in range(1, num_classes):
        if ((fixed==i).sum()==0) or ((moving==i).sum()==0):
            continue
        if ((moving_warped==i).sum()==0):
            hd95 += compute_robust_hausdorff(compute_surface_distances((fixed==i), (moving==i), spacing), 95.)
        else:
            hd95 += compute_robust_hausdorff(compute_surface_distances((fixed==i), (moving_warped==i), spacing), 95.)
        count += 1
    hd95 /= count

    return hd95

def computeJacDetVal(jac_det, img_size):

    jac_det_val = np.sum(jac_det <= 0) / np.prod(img_size)

    return jac_det_val

def computeSDLogJ(jac_det, rho=3):

    log_jac_det = np.log(np.abs((jac_det+rho).clip(1e-8, 1e8)))
    std_dev_jac = np.std(log_jac_det)

    return std_dev_jac

class GaussianBlur3D(nn.Module):
    def __init__(self, channels, sigma=1, kernel_size=0):
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

        self.register_buffer('gaussian_kernel', gaussian_kernel)
        self.padding = kernel_size // 2

    def forward(self, x):
        blurred = F.conv3d(x, self.gaussian_kernel, padding=self.padding, groups=self.channels)
        return blurred

class GaussianBlur2D(nn.Module):
    def __init__(self, channels, sigma=1, kernel_size=0):
        super(GaussianBlur2D, self).__init__()
        self.channels = channels

        if kernel_size == 0:
            kernel_size = int(2.0 * sigma * 2 + 1)

        # Create a 2D Gaussian kernel
        coord = torch.arange(kernel_size)
        grid = coord.repeat(kernel_size).view(kernel_size, kernel_size)
        xy_grid = torch.stack([grid, grid.t()], dim=-1).float()

        mean = (kernel_size - 1) / 2.
        variance = sigma**2.

        gaussian_kernel = (1. / (2. * np.pi * variance)) * \
                          torch.exp(
                              -torch.sum((xy_grid - mean)**2., dim=-1) / \
                              (2 * variance)
                          )
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

        self.register_buffer('gaussian_kernel', gaussian_kernel)
        self.padding = kernel_size // 2

    def forward(self, x):
        blurred = F.conv2d(x, self.gaussian_kernel, padding=self.padding, groups=self.channels)
        return blurred

class AnisotropicGaussianBlur3D(nn.Module):
    def __init__(self, channels, sigma=(1, 1, 1), kernel_size=0):
        super(AnisotropicGaussianBlur3D, self).__init__()
        self.channels = channels
        sigma_d, sigma_h, sigma_w = sigma

        if kernel_size == 0:
            kernel_size_d = int(2.0 * sigma_d * 2 + 1)
            kernel_size_h = int(2.0 * sigma_h * 2 + 1)
            kernel_size_w = int(2.0 * sigma_w * 2 + 1)
        else:
            kernel_size_d = kernel_size_h = kernel_size_w = kernel_size

        # Create a 3D Gaussian kernel for each dimension
        d_coord = torch.arange(kernel_size_d)
        h_coord = torch.arange(kernel_size_h)
        w_coord = torch.arange(kernel_size_w)

        d_grid = d_coord.repeat(kernel_size_h, kernel_size_w, 1).permute(2, 0, 1)
        h_grid = h_coord.repeat(kernel_size_d, kernel_size_w, 1).permute(1, 0, 2)
        w_grid = w_coord.repeat(kernel_size_d, kernel_size_h, 1).permute(1, 2, 0)

        mean_d = (kernel_size_d - 1) / 2.
        mean_h = (kernel_size_h - 1) / 2.
        mean_w = (kernel_size_w - 1) / 2.

        variance_d = sigma_d ** 2.
        variance_h = sigma_h ** 2.
        variance_w = sigma_w ** 2.

        # Calculate the Gaussian kernel
        gaussian_kernel = (1. / ((2. * np.pi) ** 1.5 * sigma_d * sigma_h * sigma_w)) * \
                          torch.exp(-(((d_grid - mean_d) ** 2.) / (2 * variance_d) +
                                      ((h_grid - mean_h) ** 2.) / (2 * variance_h) +
                                      ((w_grid - mean_w) ** 2.) / (2 * variance_w)))

        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # Reshape to 3d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size_d, kernel_size_h, kernel_size_w)
        gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1, 1)

        self.register_buffer('gaussian_kernel', gaussian_kernel)
        self.padding_d = kernel_size_d // 2
        self.padding_h = kernel_size_h // 2
        self.padding_w = kernel_size_w // 2

    def forward(self, x):
        x = F.pad(x, (self.padding_w, self.padding_w, self.padding_h, self.padding_h, self.padding_d, self.padding_d), mode='replicate')
        blurred = F.conv3d(x, self.gaussian_kernel, groups=self.channels)
        return blurred
