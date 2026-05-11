import argparse
import os
from pathlib import Path

import nibabel as nib
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = REPO_ROOT / "abdomenreg"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract pretrained segmentation features for VoxelOpt abdomen CT registration."
    )
    parser.add_argument("--data_path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--clip_min", type=float, default=-500.0)
    parser.add_argument("--clip_max", type=float, default=800.0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_dtype", choices=["float32", "float16"], default="float16")
    parser.add_argument("--max_subjects", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.gpu_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    import torch

    from loaders.abdomenreg_loader import abdomenreg_loader
    from models.preUnetComplex import preUnetComplex

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print("Device:", device)

    fea_dir = args.data_path / "fea"
    fea_dir.mkdir(parents=True, exist_ok=True)
    abct_loader = abdomenreg_loader(
        root_dir=args.data_path,
        split=args.split,
        clips=[args.clip_min, args.clip_max],
        load_features=False,
    )

    unet_fea = preUnetComplex().to(device).eval()

    img_fps = abct_loader.ori_img_fps
    if args.max_subjects is not None:
        img_fps = img_fps[:args.max_subjects]

    for img_fp in img_fps:
        fea_fp = fea_dir / Path(img_fp).name.replace(".nii.gz", ".npy")
        if fea_fp.exists() and not args.overwrite:
            print("exists:", fea_fp)
            continue

        img = nib.load(img_fp).get_fdata(dtype=np.float32)
        img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)
        img = torch.clamp(img, args.clip_min, args.clip_max)
        img = (img - args.clip_min) / (args.clip_max - args.clip_min)

        with torch.inference_mode():
            if args.amp and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    fea = unet_fea(img)
            else:
                fea = unet_fea(img)

        fea_np = fea.cpu().numpy()
        if args.save_dtype == "float16":
            fea_np = fea_np.astype(np.float16)
        np.save(fea_fp, fea_np)
        print("saved:", fea_fp, tuple(fea.shape))
        del img, fea, fea_np
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
