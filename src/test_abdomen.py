import os
import time
import torch
import argparse
import numpy as np
import pandas as pd
import torch.nn.functional as F
from pathlib import Path
from collections import OrderedDict

from utils.functions import registerSTModel, dice_eval, AverageMeter, dice_binary, jacobian_determinant, compute_HD95, computeSDLogJ
from loaders.abdomenreg_loader import abdomenreg_loader
from models.costVolComplex import costVolComplex
from models.mind import MINDSSC

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = REPO_ROOT / "abdomenreg"

label2text_dict = OrderedDict([
    # [0,  'background'], # 0: background added later after text description
    [1,  'spleen'],
    [2,  'right kidney'],
    [3,  'left kidney'],
    [4,  'gall bladder'],
    [5,  'esophagus'],
    [6,  'liver'],
    [7,  'stomach'],
    [8,  'aorta'],
    [9,  'inferior vena cava'],
    [10, 'portal and splenic vein'],
    [11, 'pancreas'],
    [12, 'left adrenal gland'],
    [13, 'right adrenal gland'],
])

def run(opt):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    reg_mode_ne = registerSTModel(opt['img_size'], 'nearest').to(device)

    dataset = abdomenreg_loader(root_dir=opt['data_path'], split=opt['split'], clips=opt['clips'])
    print("Dataset size: ", len(dataset))

    organ_eval_dsc = [AverageMeter() for i in range(1,14)]
    eval_dsc_all = AverageMeter()
    init_dsc_all = AverageMeter()
    eval_det = AverageMeter()
    eval_std_det = AverageMeter()
    eval_hd95 = AverageMeter()
    init_hd95 = AverageMeter()
    time_meter = AverageMeter()

    df_data = []
    keys = ['idx1', 'idx2'] + [label2text_dict[i] for i in range(1,14)] + ['val_dice', 'init_dice', 'jac_det', 'std_dev', 'hd95', 'init_hd95', 'time (s)']

    model = costVolComplex(**opt['nkwargs']).to(device).eval()

    n_pairs = len(dataset) if opt['max_pairs'] is None else min(opt['max_pairs'], len(dataset))
    for idx in range(n_pairs):

        loop_df_data = []
        x_img, x_seg, y_img, y_seg, idx1, idx2, x_raw, y_raw = dataset[idx]
        x_img = x_img.to(device).float()
        y_img = y_img.to(device).float()
        x_seg = x_seg.to(device).float()
        y_seg = y_seg.to(device).float()

        if opt['fea_type'] == 'raw':
            x_img = x_raw.to(device).float()
            y_img = y_raw.to(device).float()
        elif opt['fea_type'] == 'mind':
            with torch.inference_mode():
                x_img = MINDSSC(x_raw.to(device).float())
                y_img = MINDSSC(y_raw.to(device).float())
        elif opt['fea_type'] in ['foundation', 'seg']:
            x_img = x_img
            y_img = y_img
        else:
            raise ValueError(f"Unknown fea_type: {opt['fea_type']}")

        x_img = F.interpolate(x_img, scale_factor=0.5, mode='trilinear', align_corners=True)
        y_img = F.interpolate(y_img, scale_factor=0.5, mode='trilinear', align_corners=True)
        x_seg = F.interpolate(x_seg, scale_factor=0.5, mode='nearest')
        y_seg = F.interpolate(y_seg, scale_factor=0.5, mode='nearest')

        # record idx
        loop_df_data.append(idx1)
        loop_df_data.append(idx2)

        st = time.time()
        with torch.inference_mode():
            pos_flow = model(x_img, y_img).float()
        et = time.time()

        run_time = et - st
        time_meter.update(run_time, 1)

        def_out = reg_mode_ne(x_seg.float(), pos_flow)
        for idx in range(1,14):
            dsc_idx = dice_binary(def_out.long().squeeze().cpu().numpy(), y_seg.long().squeeze().cpu().numpy(), idx)
            loop_df_data.append(dsc_idx)
            organ_eval_dsc[idx-1].update(dsc_idx, 1)
        dsc1 = dice_eval(def_out.long(), y_seg.long(), 14)
        eval_dsc_all.update(dsc1.item(), 1)
        dsc2 = dice_eval(x_seg.long(), y_seg.long(), 14)
        init_dsc_all.update(dsc2.item(), 1)

        jac_det = jacobian_determinant(pos_flow.detach().cpu().numpy())
        jac_det_val = np.sum(jac_det <= 0) / np.prod(x_seg.shape)
        eval_det.update(jac_det_val, 1)

        std_dev_jac = computeSDLogJ(jac_det)
        eval_std_det.update(std_dev_jac, 1)

        moving = x_seg.long().squeeze().cpu().numpy()
        fixed = y_seg.long().squeeze().cpu().numpy()
        moving_warped = def_out.long().squeeze().cpu().numpy()
        hd95_1 = compute_HD95(moving, fixed, moving_warped,14,np.array(opt['voxel_spacing']))
        eval_hd95.update(hd95_1, 1)
        hd95_2 = compute_HD95(moving, fixed, moving,14,np.array(opt['voxel_spacing']))
        init_hd95.update(hd95_2, 1)
        print('idx1 {:d}, idx2 {:d}, val dice {:.4f}, init dice {:.4f}, jac det {:.4f}, std dev {:.4f}, hd95 {:.4f}, init hd95 {:.4f}, time {:.2f}'.format(idx1, idx2, dsc1.item(), dsc2.item(), jac_det_val, std_dev_jac, hd95_1, hd95_2, run_time))

        loop_df_data += [dsc1.item(), dsc2.item(), jac_det_val, std_dev_jac, hd95_1, hd95_2, run_time]
        df_data.append(loop_df_data)

    avg_organ_eval_dsc = [organ_eval_dsc[i].avg for i in range(13)]
    avg_df_data = [0,0] + avg_organ_eval_dsc + [eval_dsc_all.avg, init_dsc_all.avg, eval_det.avg, eval_std_det.avg, eval_hd95.avg, init_hd95.avg, time_meter.avg]
    df_data.append(avg_df_data)

    print('Avg val dice {:.4f}, Avg init dice {:.4f}, Avg jac det {:.4f}, Avg std dev {:.4f}, Avg hd95 {:.4f}, Avg init hd95 {:.4f}, Avg time {:.2f}'.format(eval_dsc_all.avg, init_dsc_all.avg, eval_det.avg, eval_std_det.avg, eval_hd95.avg, init_hd95.avg, time_meter.avg))

    folder = Path(opt['output_dir'])
    folder.mkdir(parents=True, exist_ok=True)

    ks = opt['nkwargs']['ks']
    is_half = opt['nkwargs']['is_half']
    is_adaptive = opt['nkwargs']['is_adaptive']
    fea_type = opt['fea_type']
    name = 'results_ks%s_half%s_ada%s_%s.csv' % (ks, is_half, is_adaptive, fea_type)
    if opt['max_pairs'] is not None:
        name = name.replace('.csv', '_n%s.csv' % opt['max_pairs'])

    df = pd.DataFrame(df_data, columns=keys)
    fp = folder / name
    df.to_csv(fp, index=False)
    print("Saved results:", fp)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = "Evaluate VoxelOpt on the abdomen CT test split.")
    parser.add_argument("--data_path", type = Path, default = DEFAULT_DATA_PATH)
    parser.add_argument("--split", type = str, default = 'test', choices=['train', 'val', 'test'])
    parser.add_argument("--output_dir", type = Path, default = REPO_ROOT / 'logs_abct')
    parser.add_argument("--gpu_id", type = str, default = '0')
    parser.add_argument("--fea_type", type = str, default = 'foundation', choices=['foundation', 'seg', 'raw', 'mind'])
    parser.add_argument("--clip_min", type = float, default = -500.0)
    parser.add_argument("--clip_max", type = float, default = 800.0)
    parser.add_argument("--voxel_spacing", type = float, nargs = 3, default = [4.0, 4.0, 4.0])
    parser.add_argument("--max_pairs", type = int, default = None)

    args, unknowns = parser.parse_known_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    opt = vars(args)
    opt['img_size'] = (192//2, 160//2, 256//2)
    opt['clips'] = [opt.pop('clip_min'), opt.pop('clip_max')]
    opt['nkwargs'] = {
        'img_size': str(opt['img_size']),
        'ks': '1',
        'is_adaptive': '1',
        'is_half': '1',
        'sigma_cap': '0.5',
    }
    for item in unknowns:
        if "=" not in item:
            raise ValueError(f"Model override must use key=value syntax: {item}")
        key, value = item.split("=", 1)
        opt['nkwargs'][key] = value

    print("Model kwargs:", opt['nkwargs'])
    run(opt)
