# train.py (fixed: single-real-image inference-only benchmark with safe forward)
import os
import sys
import time
import random
import cv2
import torch
import argparse
import pandas as pd
import numpy as np
from timeit import default_timer as timer
from time import perf_counter
from sklearn.metrics import roc_auc_score, confusion_matrix
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import wandb
from tqdm import tqdm
import timm
import torch.nn as nn
import torch.nn.functional as F

from utils.landmark_cache import (
    load_landmark_cache,
    save_landmark_cache,
    cached_extract_landmark_coords,
)
from utils.datasets import StegDataset
from utils.utils import save_checkpoint, time_to_str
from models.vit_learnable import ViTWithRegionBias

# -----------------------------
# 환경 변수 및 결정론적 설정 추가
# -----------------------------
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True)

# -----------------------------
# 시드 고정 함수 (DataLoader 포함)
# -----------------------------
def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)
    np.random.seed(args.seed + worker_id)
    torch.manual_seed(args.seed + worker_id)

# -----------------------------
# 검증 루프
# -----------------------------
def validate(loader, model, criterion, device, landmark_cache, extract_fn):
    model.eval()
    total_loss, correct = 0.0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels, paths in tqdm(loader, desc="Validate", leave=False):
            images, labels = images.to(device), labels.to(device)

            landmarks_list = [
                cached_extract_landmark_coords(p, extract_fn, landmark_cache) for p in paths
            ]

            outputs = model(images, landmarks_list)
            loss = criterion(outputs, labels)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            total_loss += loss.item()
            correct += (preds == labels).sum().item()
            all_preds.append(probs.cpu())
            all_labels.append(labels.cpu())

    avg_loss = total_loss / len(loader)
    acc = 100.0 * correct / len(loader.dataset)
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    auc = roc_auc_score(all_labels, all_preds[:, 1])
    tn, fp, fn, tp = confusion_matrix(all_labels, np.argmax(all_preds, axis=1), labels=[0, 1]).ravel()
    apcer = fp / (tn + fp + 1e-6)
    bpcer = fn / (tp + fn + 1e-6)
    acer = (apcer + bpcer) / 2
    return avg_loss, acc, auc, acer, apcer, bpcer

# -----------------------------
# --- Benchmark helpers: CBAM2D + wrapper + safe forward
# -----------------------------
class CBAM2D(nn.Module):
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.channels = channels
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=True),
        )
        pad = (spatial_kernel - 1) // 2
        self.spatial = nn.Conv2d(2, 1, kernel_size=spatial_kernel, padding=pad, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        avg_pool = F.adaptive_avg_pool2d(x, 1).view(B, C)
        max_pool = F.adaptive_max_pool2d(x, 1).view(B, C)
        ca = torch.sigmoid(self.mlp(avg_pool) + self.mlp(max_pool)).view(B, C, 1, 1)
        x = x * ca
        avg_c = torch.mean(x, dim=1, keepdim=True)
        max_c = torch.max(x, dim=1, keepdim=True).values
        sa = torch.cat([avg_c, max_c], dim=1)
        sa = torch.sigmoid(self.spatial(sa))
        x = x * sa
        return x

def apply_cbam_on_patches(patch_tokens, cbam2d):
    B, N, D = patch_tokens.shape
    H = W = int(N ** 0.5)
    assert H * W == N, "patch count N must be a square (H*W)."
    p = patch_tokens.permute(0, 2, 1).reshape(B, D, H, W)
    p = cbam2d(p)
    p = p.reshape(B, D, N).permute(0, 2, 1)
    return p

class ViT_CBAM_Wrapper(nn.Module):
    def __init__(self, vit):
        super().__init__()
        self.vit = vit
        if hasattr(vit, "embed_dim"):
            embed_dim = vit.embed_dim
        else:
            embed_dim = getattr(vit, "num_features", None)
            if embed_dim is None:
                embed_dim = vit.head.in_features
        self.cbam = CBAM2D(embed_dim)

    def forward(self, x):
        B = x.size(0)
        patch_tokens = self.vit.patch_embed(x)
        cls_token = self.vit.cls_token.expand(B, -1, -1)
        pos = self.vit.pos_embed.to(patch_tokens.device).to(patch_tokens.dtype)
        x_tokens = torch.cat((cls_token, patch_tokens), dim=1) + pos
        x_tokens = self.vit.pos_drop(x_tokens)
        cls_tok, pats = x_tokens[:, :1], x_tokens[:, 1:]
        pats = apply_cbam_on_patches(pats, self.cbam)
        x_tokens = torch.cat((cls_tok, pats), dim=1)
        for blk in self.vit.blocks:
            x_tokens = blk(x_tokens)
        x_tokens = self.vit.norm(x_tokens)
        return self.vit.head(x_tokens[:, 0])

def safe_forward_call(model, img, landmarks=None):
    try:
        # try passing landmarks list-like (model expects landmarks_list)
        return model(img, landmarks)
    except TypeError:
        # fallback to single-arg forward if supported
        return model(img)

# -----------------------------
# 메인 함수
# -----------------------------
def main(args):
    start_time = timer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 랜드마크 모드 설정 및 캐시 로드
    if args.landmark_mode == "full":
        from utils.landmark_fullpoint import (
            extract_landmark_full_coords as extract_landmark_coords,
        )
        cache_file = "landmark_cache_full.pkl"
    else:
        from utils.landmark_5point import (
            extract_landmark_5pt_coords as extract_landmark_coords,
        )
        cache_file = "landmark_cache_5pt.pkl"

    landmark_cache = load_landmark_cache(cache_file)
    print(f"Loaded landmark cache with {len(landmark_cache)} entries from {cache_file}")

    # 모델 초기화 (원본 코드 유지)
    model = ViTWithRegionBias(
        num_classes=args.num_classes,
        default_weight=args.default_weight,
        enhanced_weight=args.enhanced_weight,
        patch_size=args.patch_size,
        learnable=not args.fix_weight,
        scalar_learnable=args.scalar_learnable,
    ).to(device)

    # -----------------------------
    # BENCHMARK BLOCK: single random real image from dataset (inference-only)
    # -----------------------------
    def load_image_tensor(path, transform):
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        augmented = transform(image=img)
        tensor = augmented["image"].unsqueeze(0)
        return tensor

    def pick_one_image_from_csv(csv_root, datasets, landmark_cache):
        if datasets:
            csv_file = os.path.join(csv_root, f"{datasets}_valid.csv")
        else:
            csv_file = os.path.join(csv_root, "valid.csv")
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            if "path" in df.columns:
                paths = df["path"].tolist()
            else:
                paths = df.iloc[:, 0].tolist()
            valid = [p for p in paths if (p in landmark_cache) and os.path.exists(p)]
            if valid:
                return random.choice(valid)
        valid_cache_keys = [k for k in landmark_cache.keys() if os.path.exists(k)]
        if not valid_cache_keys:
            raise RuntimeError("No valid image files found that match landmark cache or csv.")
        return random.choice(valid_cache_keys)

    def stats(arr):
        a = np.array(arr)
        return a.mean(), a.std(), np.median(a), np.percentile(a,95), np.percentile(a,99)

    def do_single_image_benchmark(model, device, img_tensor, img_path=None, landmarks=None, n_runs=100, warmup=10):
        """
        Single-image benchmark, no batching.
        - model: model object (may accept model(img, landmarks) or model(img))
        - img_tensor: torch tensor shape (1,3,224,224), already preprocessed
        - img_path: optional path (used for measuring cache lookup with cached_extract_landmark_coords)
        - landmarks: optional preloaded landmarks (list). If img_path provided, we measure lookup via cached_extract_landmark_coords.
        Returns dict with arrays and stats.
        """
        model.eval()
        model.to(device)

        has_make_wm = hasattr(getattr(model, "region_attn", None), "make_weight_map")

        # warmup - use safe_forward_call
        with torch.no_grad():
            for _ in range(warmup):
                if img_path is not None:
                    try:
                        lm0 = cached_extract_landmark_coords(img_path, extract_landmark_coords, landmark_cache)
                    except Exception:
                        lm0 = landmarks
                else:
                    lm0 = landmarks
                _ = safe_forward_call(model, img_tensor.to(device), [lm0] if lm0 is not None else None)
                if device.type == "cuda":
                    torch.cuda.synchronize()

        lookup_times = []
        wm_times = []
        forward_times = []

        with torch.no_grad():
            for _ in range(n_runs):
                # landmark lookup timing
                if img_path is not None:
                    t_l0 = perf_counter()
                    lm = cached_extract_landmark_coords(img_path, extract_landmark_coords, landmark_cache)
                    t_l1 = perf_counter()
                    lookup_times.append((t_l1 - t_l0) * 1000.0)
                    used_landmarks = lm
                else:
                    lookup_times.append(0.0)
                    used_landmarks = landmarks

                # weight-map generation timing
                if has_make_wm and (used_landmarks is not None):
                    t_w0 = perf_counter()
                    try:
                        _ = model.region_attn.make_weight_map(used_landmarks)
                    except Exception:
                        pass
                    t_w1 = perf_counter()
                    wm_times.append((t_w1 - t_w0) * 1000.0)
                else:
                    wm_times.append(0.0)

                # forward timing using safe_forward_call
                t_f0 = perf_counter()
                _ = safe_forward_call(model, img_tensor.to(device), [used_landmarks] if used_landmarks is not None else None)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t_f1 = perf_counter()
                forward_times.append((t_f1 - t_f0) * 1000.0)

        forward_arr = np.array(forward_times)
        lookup_arr = np.array(lookup_times)
        wm_arr = np.array(wm_times)

        def stats_local(a):
            a = np.array(a)
            if a.size == 0:
                return (0.0, 0.0, 0.0, 0.0, 0.0)
            return (a.mean(), a.std(), np.median(a), np.percentile(a,95), np.percentile(a,99))

        return {
            "forward_times": forward_arr,
            "lookup_times": lookup_arr,
            "wm_times": wm_arr,
            "forward_stats": stats_local(forward_arr),
            "lookup_stats": stats_local(lookup_arr),
            "wm_stats": stats_local(wm_arr),
        }

    if args.benchmark:
        print(">> BENCHMARK MODE: single random real image from dataset (using cached landmarks if present).")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            img_path = pick_one_image_from_csv(args.csv_root, args.datasets, landmark_cache)
        except Exception as e:
            print("Cannot pick an image from CSV/cache:", e)
            sys.exit(1)

        print("Selected image:", img_path)
        try:
            valid_transform
        except NameError:
            valid_transform = A.Compose(
                [
                    A.Resize(224, 224),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ]
            )

        img_tensor = load_image_tensor(img_path, valid_transform)

        if img_path in landmark_cache:
            img_landmarks = landmark_cache[img_path]
        else:
            img_landmarks = [(112.0, 112.0)] * 500

        # RWM
        try:
            res_rwm = do_single_image_benchmark(model, device, img_tensor, img_path=img_path, landmarks=None, n_runs=100, warmup=10)
            f_mean, f_std, f_med, f_p95, f_p99 = res_rwm["forward_stats"]
            print(f"RWM forward mean {f_mean:.2f} ms, std {f_std:.2f}, median {f_med:.2f}, p95 {f_p95:.2f}, p99 {f_p99:.2f}")
            print(f"RWM landmark lookup mean/std/p95/p99: {res_rwm['lookup_stats']}")
            print(f"RWM weight-map mean/std/p95/p99: {res_rwm['wm_stats']}")
            np.save("rwm_forward_times.npy", res_rwm["forward_times"])
            np.save("rwm_lookup_times.npy", res_rwm["lookup_times"])
            np.save("rwm_wm_times.npy", res_rwm["wm_times"])
        except Exception as e:
            print("Error during RWM benchmark:", e)

        # CBAM
        base_vit = timm.create_model("vit_small_patch16_224", pretrained=True)
        in_feats = base_vit.head.in_features
        base_vit.head = nn.Linear(in_feats, args.num_classes)
        cbam_model = ViT_CBAM_Wrapper(base_vit).to(device)

        try:
            res_cbam = do_single_image_benchmark(cbam_model, device, img_tensor, img_path=None, landmarks=None, n_runs=100, warmup=10)
            f_mean, f_std, f_med, f_p95, f_p99 = res_cbam["forward_stats"]
            print(f"CBAM forward mean {f_mean:.2f} ms, std {f_std:.2f}, median {f_med:.2f}, p95 {f_p95:.2f}, p99 {f_p99:.2f}")
            np.save("cbam_forward_times.npy", res_cbam["forward_times"])
        except Exception as e:
            print("Error during CBAM benchmark:", e)

        print(">> BENCHMARK DONE (single real image).")
        sys.exit(0)

    # -----------------------------
    # W&B 설정 (이후 원본 훈련 흐름 유지)
    # -----------------------------
    if args.use_wandb:
        wandb.init(project="KCI2025", entity="")
        if args.run_name == "auto":
            mode = "fixed" if args.fix_weight else "learnable"
            args.run_name = f"region_{mode}_{args.landmark_mode}_{args.batch_size}_{args.lr}"
            if args.suffix:
                args.run_name += f"_{args.suffix}"
        wandb.run.name = args.run_name
        wandb.config.update(vars(args))

    # CSV 경로
    if args.datasets:
        train_path = os.path.join(args.csv_root, f"{args.datasets}_train.csv")
        valid_path = os.path.join(args.csv_root, f"{args.datasets}_valid.csv")
        test_path = os.path.join(args.csv_root, f"{args.datasets}_test.csv")
    else:
        train_path = os.path.join(args.csv_root, "train.csv")
        valid_path = os.path.join(args.csv_root, "valid.csv")
        test_path = os.path.join(args.csv_root, "test.csv")

    train_df = pd.read_csv(train_path)
    valid_df = pd.read_csv(valid_path)
    test_df = pd.read_csv(test_path)

    # Transform
    train_transform = A.Compose(
        [
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )

    valid_transform = A.Compose(
        [
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )

    # DataLoader (원본)
    train_loader = DataLoader(
        StegDataset(train_df, transform=train_transform, return_path=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers_per_loader,
        worker_init_fn=worker_init_fn,
    )
    valid_loader = DataLoader(
        StegDataset(valid_df, transform=valid_transform, return_path=True),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers_per_loader,
        worker_init_fn=worker_init_fn,
    )
    test_loader = DataLoader(
        StegDataset(test_df, transform=valid_transform, return_path=True),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers_per_loader,
        worker_init_fn=worker_init_fn,
    )

    criterion = torch.nn.CrossEntropyLoss()

    # Optimizer
    vit_params = [p for n, p in model.named_parameters() if "region_attn" not in n and p.requires_grad]
    eye_param = [model.region_attn.eye_weight]
    nose_param = [model.region_attn.nose_weight]
    mouth_param = [model.region_attn.mouth_weight]

    optimizer = torch.optim.AdamW(
        [
            {"params": vit_params, "lr": args.lr, "weight_decay": args.weight_decay},
            {"params": eye_param, "lr": args.lr * 1.5, "weight_decay": 0.0},
            {"params": nose_param, "lr": args.lr * 1.5, "weight_decay": 0.0},
            {"params": mouth_param, "lr": args.lr * 1.5, "weight_decay": 0.0},
        ]
    )

    best_valid_acer, best_valid_auc, best_epoch = float("inf"), 0.0, 0

    # 학습 루프 (원본)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, correct = 0.0, 0

        for images, labels, paths in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            images, labels = images.to(device), labels.to(device)
            landmarks_list = [
                cached_extract_landmark_coords(p, extract_landmark_coords, landmark_cache) for p in paths
            ]
            outputs = model(images, landmarks_list)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()

        train_acc = 100.0 * correct / len(train_loader.dataset)
        train_loss = total_loss / len(train_loader)

        valid_loss, valid_acc, valid_auc, valid_acer, apcer, bpcer = validate(
            valid_loader, model, criterion, device, landmark_cache, extract_landmark_coords
        )

        if (valid_acer < best_valid_acer) or (valid_acer == best_valid_acer and valid_auc > best_valid_auc):
            best_valid_acer, best_valid_auc, best_epoch = valid_acer, valid_auc, epoch
            save_checkpoint(
                {"epoch": epoch, "state_dict": model.state_dict(), "best_valid_acer": best_valid_acer},
                args.gpus,
                True,
                f"{args.ckpt_root}/{args.run_name}",
                f"ep{epoch}.pth.tar",
            )

        print(
            f"[Epoch {epoch}] Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
            f"Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.2f} | "
            f"AUC: {valid_auc:.4f}, ACER: {valid_acer:.4f} | "
            f"Time: {time_to_str(timer() - start_time, mode='min')}"
        )

        if args.use_wandb:
            log_dict = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "valid_loss": valid_loss,
                "valid_acc": valid_acc,
                "valid_auc": valid_auc,
                "valid_apcer": apcer,
                "valid_bpcer": bpcer,
                "valid_acer": valid_acer,
            }
            if hasattr(model.region_attn.eye_weight, "data") and model.region_attn.eye_weight.ndim == 1:
                log_dict.update(
                    {
                        "eye_weight_mean": model.region_attn.eye_weight.data.mean().item(),
                        "nose_weight_mean": model.region_attn.nose_weight.data.mean().item(),
                        "mouth_weight_mean": model.region_attn.mouth_weight.data.mean().item(),
                    }
                )
            wandb.log(log_dict)

    # 테스트 및 마무리
    best_ckpt = torch.load(f"{args.ckpt_root}/{args.run_name}/ep{best_epoch}.pth.tar")
    model.load_state_dict(best_ckpt["state_dict"])
    test_loss, test_acc, test_auc, test_acer, _, _ = validate(
        test_loader, model, criterion, device, landmark_cache, extract_landmark_coords
    )

    print(f"\n[Best Epoch {best_epoch}] Test Acc: {test_acc:.2f}% | AUC: {test_auc:.4f} | ACER: {test_acer:.4f}")

    if args.use_wandb:
        wandb.run.summary["test_acc"] = test_acc
        wandb.run.summary["test_auc"] = test_auc
        wandb.run.summary["test_acer"] = test_acer

    save_landmark_cache(landmark_cache, cache_file)


# -----------------------------
# CLI 인자 & RNG 고정
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Region Bias Training (Deterministic)")
    parser.add_argument("--csv_root", type=str, default="./csv/")
    parser.add_argument("--ckpt_root", type=str, default="./ckpt/")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--workers_per_loader", type=int, default=4)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--default_weight", type=float, default=1.0)
    parser.add_argument("--enhanced_weight", type=float, default=1.2)
    parser.add_argument("--fix_weight", action="store_true")
    parser.add_argument("--scalar_learnable", action="store_true")
    parser.add_argument("--landmark_mode", choices=["full", "5pt"], default="full")
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--run_name", type=str, default="auto")
    parser.add_argument("--datasets", type=str, default=None)
    parser.add_argument("--benchmark", action="store_true", help="Run forward benchmark: single real image from dataset and exit")

    args = parser.parse_args()

    if args.datasets:
        args.csv_root = "./csv/"
        if args.suffix == "":
            args.suffix = f"gen_{args.datasets}"

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    main(args)
