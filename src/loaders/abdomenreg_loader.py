import torch
import itertools
import numpy as np
import nibabel as nib
from pathlib import Path
from torch.utils.data import Dataset

class abdomenreg_loader(Dataset):

    def __init__(self,
            root_dir = None,
            split = 'train', # train, val or test
            clips = [-500, 800],
            load_features = True,
        ):

        if root_dir is None:
            root_dir = Path(__file__).resolve().parents[2] / "abdomenreg"
        self.root_dir = Path(root_dir)
        self.split = split
        self.clips = clips
        self.load_features = load_features

        if self.split == 'train':
            idxs = np.arange(1,21)
        elif self.split == 'val':
            idxs = np.arange(21,24)
        elif self.split == 'test':
            idxs = np.arange(24,31)
        elif self.split == 'all':
            idxs = np.arange(1,31)
        else:
            raise ValueError(f"Unknown split: {self.split}")

        img_fps = [self.root_dir / 'img' / ("img%s.nii.gz" % (str(idx).zfill(4))) for idx in idxs]
        lbl_fps = [self.root_dir / 'label' / ("label%s.nii.gz" % (str(idx).zfill(4))) for idx in idxs]
        fea_fps = [self.root_dir / 'fea' / ("img%s.npy" % (str(idx).zfill(4))) for idx in idxs]

        missing = [str(fp) for fp in img_fps + lbl_fps if not fp.exists()]
        if missing:
            raise FileNotFoundError("Missing abdomen registration files:\n" + "\n".join(missing[:10]))

        self.ori_img_fps = [str(fp) for fp in img_fps]
        self.ori_lbl_fps = [str(fp) for fp in lbl_fps]

        save_fp = self.root_dir / 'save'
        save_fp.mkdir(parents=True, exist_ok=True)
        save_fps = [save_fp / ("subject%s.npz" % (str(idx).zfill(4))) for idx in idxs]

        self.save_fps = {idx: str(save_fp) for idx, save_fp in zip(idxs, save_fps)}
        self.img_fps = list(itertools.permutations([str(fp) for fp in img_fps], 2))
        self.lbl_fps = list(itertools.permutations([str(fp) for fp in lbl_fps], 2))
        self.fea_fps = list(itertools.permutations([str(fp) for fp in fea_fps], 2))
        self.sub_idx = list(itertools.permutations(idxs, 2))

        print('----->>>> %s set has %d subjects' % (self.split, len(img_fps)))
        print('----->>>> %s set has %d pairs' % (self.split, len(self.sub_idx)))

    def __len__(self):
        return len(self.img_fps)

    def __getitem__(self, idx):

        sub_idx1, sub_idx2 = self.sub_idx[idx]
        fea_fp1, fea_fp2 = self.fea_fps[idx]
        lbl_fp1, lbl_fp2 = self.lbl_fps[idx]
        img_fp1, img_fp2 = self.img_fps[idx]

        lbl1 = np.array(nib.load(lbl_fp1).get_fdata(), dtype='float32')
        lbl2 = np.array(nib.load(lbl_fp2).get_fdata(), dtype='float32')

        img1 = np.array(nib.load(img_fp1).get_fdata(), dtype='float32')
        img2 = np.array(nib.load(img_fp2).get_fdata(), dtype='float32')

        if self.load_features:
            if not (Path(fea_fp1).exists() and Path(fea_fp2).exists()):
                raise FileNotFoundError(
                    "Missing feature maps. Run get_unet_features.py before test_abdomen.py."
                )
            fea1 = np.load(fea_fp1)
            fea2 = np.load(fea_fp2)
            src_fea, tgt_fea = torch.from_numpy(fea1), torch.from_numpy(fea2)
        else:
            src_fea = tgt_fea = torch.empty(0)

        src_lbl = torch.from_numpy(lbl1).unsqueeze(0).unsqueeze(0)
        tgt_lbl = torch.from_numpy(lbl2).unsqueeze(0).unsqueeze(0)
        if self.clips is not None:
            src_img = torch.from_numpy(img1).unsqueeze(0).unsqueeze(0)
            tgt_img = torch.from_numpy(img2).unsqueeze(0).unsqueeze(0)
            src_img = torch.clamp(src_img, self.clips[0], self.clips[1])
            tgt_img = torch.clamp(tgt_img, self.clips[0], self.clips[1])
            src_img = (src_img - self.clips[0]) / (self.clips[1] - self.clips[0])
            tgt_img = (tgt_img - self.clips[0]) / (self.clips[1] - self.clips[0])

        return src_fea, src_lbl, tgt_fea, tgt_lbl, sub_idx1, sub_idx2, src_img, tgt_img

if __name__ == '__main__':

    pass
