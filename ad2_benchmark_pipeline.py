import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import tifffile
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dataset.dataset import MVTecDataset_train, get_data_transforms
from model.de_resnet import de_wide_resnet50_2
from model.resnet import wide_resnet50_2
from utils.utils_test import cal_anomaly_map
from utils.utils_train import MultiProjectionLayer, Revisit_RDLoss, loss_fucntion


AD2_OBJECTS = [
    "can",
    "fabric",
    "fruit_jelly",
    "rice",
    "sheet_metal",
    "vial",
    "wallplugs",
    "walnuts",
]


def setup_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AD2SplitDataset(Dataset):
    def __init__(self, data_root: str, mad2_object: str, split: str, transform, image_size: int):
        assert split in {"validation", "test_private", "test_private_mixed"}
        self.data_root = Path(data_root)
        self.object_name = mad2_object
        self.split = split
        self.transform = transform
        self.image_size = image_size
        self.image_paths = self._load_paths()

    def _load_paths(self):
        object_dir = self.data_root / self.object_name
        if self.split in {"test_private", "test_private_mixed"}:
            return sorted((object_dir / self.split).glob("*.png"))
        return sorted((object_dir / self.split / "good").glob("*.png"))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]
        image = cv2.resize(image / 255.0, (self.image_size, self.image_size))
        sample = self.transform(image)

        out_info = {
            "sample": sample,
            "image_path": str(image_path),
            "orig_h": orig_h,
            "orig_w": orig_w,
        }

        if self.split in {"test_private", "test_private_mixed"}:
            out_info["rel_out_path_cont"] = str(
                Path("anomaly_images") / self.object_name / self.split / f"{image_path.stem}.tiff"
            )
            out_info["rel_out_path_thresh"] = str(
                Path("anomaly_images_thresholded") / self.object_name / self.split / f"{image_path.stem}.png"
            )

        return out_info


def build_models(device: str):
    encoder, bn = wide_resnet50_2(pretrained=True)
    encoder = encoder.to(device)
    encoder.eval()
    bn = bn.to(device)

    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(device)

    proj_layer = MultiProjectionLayer(base=64).to(device)
    return encoder, bn, decoder, proj_layer


def train_one_object(
    object_name: str,
    args,
    device: str,
    checkpoint_dir: Path,
):
    data_transform, _ = get_data_transforms(args.image_size, args.image_size)
    train_path = os.path.join(args.data_root, object_name, "train")

    train_data = MVTecDataset_train(
        root=train_path,
        transform=data_transform,
        image_size=args.image_size,
    )
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=args.num_workers,
    )

    encoder, bn, decoder, proj_layer = build_models(device)
    proj_loss = Revisit_RDLoss()
    optimizer_proj = torch.optim.Adam(
        list(proj_layer.parameters()), lr=args.proj_lr, betas=(0.5, 0.999)
    )
    optimizer_distill = torch.optim.Adam(
        list(decoder.parameters()) + list(bn.parameters()),
        lr=args.distill_lr,
        betas=(0.5, 0.999),
    )

    for epoch in tqdm(range(1, args.epochs + 1), desc=f"train:{object_name}"):
        bn.train()
        decoder.train()
        proj_layer.train()

        optimizer_proj.zero_grad(set_to_none=True)
        optimizer_distill.zero_grad(set_to_none=True)

        for i, (img, img_noise, _) in enumerate(train_loader):
            img = img.to(device, non_blocking=True)
            img_noise = img_noise.to(device, non_blocking=True)

            inputs = encoder(img)
            inputs_noise = encoder(img_noise)
            feature_space_noise, feature_space = proj_layer(inputs, features_noise=inputs_noise)

            l_proj = proj_loss(inputs_noise, feature_space_noise, feature_space)
            outputs = decoder(bn(feature_space))
            l_distill = loss_fucntion(inputs, outputs)
            loss = l_distill + args.weight_proj * l_proj
            loss.backward()

            if (i + 1) % args.accumulation_steps == 0:
                optimizer_proj.step()
                optimizer_distill.step()
                optimizer_proj.zero_grad(set_to_none=True)
                optimizer_distill.zero_grad(set_to_none=True)

        if len(train_loader) % args.accumulation_steps != 0:
            optimizer_proj.step()
            optimizer_distill.step()
            optimizer_proj.zero_grad(set_to_none=True)
            optimizer_distill.zero_grad(set_to_none=True)

    checkpoint_path = checkpoint_dir / object_name / f"wres50_{object_name}.pth"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "proj": proj_layer.state_dict(),
            "decoder": decoder.state_dict(),
            "bn": bn.state_dict(),
        },
        checkpoint_path,
    )
    return checkpoint_path


def load_models_from_checkpoint(checkpoint_path: Path, device: str):
    encoder, bn, decoder, proj_layer = build_models(device)
    ckp = torch.load(checkpoint_path, map_location="cpu")
    proj_layer.load_state_dict(ckp["proj"])
    bn.load_state_dict(ckp["bn"])
    decoder.load_state_dict(ckp["decoder"])
    encoder.eval()
    bn.eval()
    decoder.eval()
    proj_layer.eval()
    return encoder, bn, decoder, proj_layer


@torch.no_grad()
def infer_anomaly_map(
    sample: torch.Tensor,
    orig_h: int,
    orig_w: int,
    encoder,
    proj_layer,
    bn,
    decoder,
    device: str,
):
    img = sample.to(device, non_blocking=True)
    inputs = encoder(img)
    features = proj_layer(inputs)
    outputs = decoder(bn(features))
    anomaly_map, _ = cal_anomaly_map(inputs, outputs, img.shape[-1], amap_mode="a")
    anomaly_map = gaussian_filter(anomaly_map, sigma=4)
    anomaly_map = cv2.resize(
        anomaly_map.astype(np.float32),
        (orig_w, orig_h),
        interpolation=cv2.INTER_LINEAR,
    )
    return anomaly_map


def compute_validation_threshold(
    val_loader,
    encoder,
    proj_layer,
    bn,
    decoder,
    device: str,
):
    total_count = 0
    sum_scores = 0.0
    sumsq_scores = 0.0

    for batch in tqdm(val_loader, desc="val-threshold", leave=False):
        sample = batch["sample"]
        orig_h = int(batch["orig_h"].item())
        orig_w = int(batch["orig_w"].item())
        anomaly_map = infer_anomaly_map(
            sample, orig_h, orig_w, encoder, proj_layer, bn, decoder, device
        )
        flat = anomaly_map.astype(np.float64).ravel()
        total_count += flat.size
        sum_scores += flat.sum()
        sumsq_scores += np.square(flat).sum()

    mean_score = sum_scores / total_count
    var_score = max((sumsq_scores / total_count) - mean_score**2, 0.0)
    std_score = np.sqrt(var_score)
    threshold = mean_score + 3.0 * std_score
    return float(threshold), float(mean_score), float(std_score)


def export_private_split(
    split_loader,
    encoder,
    proj_layer,
    bn,
    decoder,
    device: str,
    submission_dir: Path,
    threshold: float,
    write_thresholded: bool,
):
    for batch in tqdm(split_loader, desc="export", leave=False):
        sample = batch["sample"]
        orig_h = int(batch["orig_h"].item())
        orig_w = int(batch["orig_w"].item())
        anomaly_map = infer_anomaly_map(
            sample, orig_h, orig_w, encoder, proj_layer, bn, decoder, device
        )

        out_cont = submission_dir / batch["rel_out_path_cont"][0]
        out_cont.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(out_cont, anomaly_map.astype(np.float16))

        if write_thresholded:
            out_thresh = submission_dir / batch["rel_out_path_thresh"][0]
            out_thresh.parent.mkdir(parents=True, exist_ok=True)
            binary = (anomaly_map > threshold).astype(np.uint8) * 255
            Image.fromarray(binary).save(out_thresh)


def parse_args():
    parser = argparse.ArgumentParser(description="Train RD++ and export AD2 submission files.")
    parser.add_argument("--data_root", required=True, type=str, help="Path to mvtec_ad_2 root.")
    parser.add_argument("--output_root", default="./ad2_benchmark_run", type=str)
    parser.add_argument("--classes", nargs="+", default=AD2_OBJECTS, choices=AD2_OBJECTS)
    parser.add_argument("--image_size", default=256, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--accumulation_steps", default=2, type=int)
    parser.add_argument("--proj_lr", default=0.001, type=float)
    parser.add_argument("--distill_lr", default=0.005, type=float)
    parser.add_argument("--weight_proj", default=0.2, type=float)
    parser.add_argument("--seed", default=111, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--checkpoint_root", default=None, type=str)
    parser.add_argument("--no_thresholded", action="store_true")
    parser.add_argument("--check_submission", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_root = Path(args.output_root)
    checkpoint_root = (
        Path(args.checkpoint_root) if args.checkpoint_root else output_root / "checkpoints"
    )
    submission_dir = output_root / "submission"
    write_thresholded = not args.no_thresholded

    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    submission_dir.mkdir(parents=True, exist_ok=True)

    data_transform, _ = get_data_transforms(args.image_size, args.image_size)
    threshold_stats = {}

    for object_name in args.classes:
        print(f"\n=== Processing {object_name} ===")
        checkpoint_path = checkpoint_root / object_name / f"wres50_{object_name}.pth"
        if not args.skip_train:
            checkpoint_path = train_one_object(
                object_name=object_name,
                args=args,
                device=device,
                checkpoint_dir=checkpoint_root,
            )
        elif not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found for {object_name}: {checkpoint_path}"
            )

        encoder, bn, decoder, proj_layer = load_models_from_checkpoint(checkpoint_path, device)

        val_data = AD2SplitDataset(
            data_root=args.data_root,
            mad2_object=object_name,
            split="validation",
            transform=data_transform,
            image_size=args.image_size,
        )
        val_loader = DataLoader(
            val_data,
            batch_size=1,
            shuffle=False,
            pin_memory=True,
            num_workers=args.num_workers,
        )
        threshold, mean_score, std_score = compute_validation_threshold(
            val_loader, encoder, proj_layer, bn, decoder, device
        )
        threshold_stats[object_name] = {
            "threshold": threshold,
            "mean": mean_score,
            "std": std_score,
        }
        print(
            f"{object_name}: validation threshold = mean + 3*std = "
            f"{mean_score:.6f} + 3*{std_score:.6f} = {threshold:.6f}"
        )

        for split in ("test_private", "test_private_mixed"):
            split_data = AD2SplitDataset(
                data_root=args.data_root,
                mad2_object=object_name,
                split=split,
                transform=data_transform,
                image_size=args.image_size,
            )
            split_loader = DataLoader(
                split_data,
                batch_size=1,
                shuffle=False,
                pin_memory=True,
                num_workers=args.num_workers,
            )
            print(f"{object_name}: exporting {split} ({len(split_data)} images)")
            export_private_split(
                split_loader=split_loader,
                encoder=encoder,
                proj_layer=proj_layer,
                bn=bn,
                decoder=decoder,
                device=device,
                submission_dir=submission_dir,
                threshold=threshold,
                write_thresholded=write_thresholded,
            )

        del encoder, bn, decoder, proj_layer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with open(output_root / "thresholds.json", "w") as fp:
        json.dump(threshold_stats, fp, indent=2)

    print(f"\nSubmission directory created at: {submission_dir}")
    if args.check_submission:
        checker_script = (
            Path(__file__).resolve().parent
            / "MVTecAD2_public_code_utils"
            / "check_and_prepare_data_for_upload.py"
        )
        subprocess.run(
            [sys.executable, str(checker_script), str(submission_dir)],
            check=True,
        )
        print("Submission check completed and archive generated.")
    else:
        print(
            "Run local checker before upload:\n"
            f"  {sys.executable} MVTecAD2_public_code_utils/check_and_prepare_data_for_upload.py {submission_dir}"
        )


if __name__ == "__main__":
    main()
