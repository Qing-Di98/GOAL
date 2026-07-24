"""
GOAL with Multi-Positive Contrastive Learning
改进方案一: top-K segments + multi-positive contrastive loss

Each original image/text gets K segment pairs instead of 1,
preventing overfitting to a single best local match.
"""

import sys
import os
# Add project root to path for imports from utils/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import json
import argparse
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
import lightning as L
import transformers
import torch.nn.functional as F
import shutil
import time
import numpy as np
from utils.func import *
from utils.transforms import *
try:
    from deepspeed.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict
except ImportError:
    convert_zero_checkpoint_to_fp32_state_dict = None
import math
import random
import wandb

torch.autograd.set_detect_anomaly(True)


def clip_loss(sim):
    """Standard CLIP contrastive loss (bi-directional)"""
    gt = torch.arange(len(sim), dtype=torch.long, device=sim.device)
    return (torch.nn.CrossEntropyLoss()(sim, gt) + torch.nn.CrossEntropyLoss()(sim.t(), gt)) / 2.0


def multi_positive_clip_loss(similarity, positive_mask):
    """
    Multi-positive contrastive loss.

    Args:
        similarity: (N, M) similarity matrix
        positive_mask: (N, M) binary mask where 1 = positive pair

    For each anchor (row), all positives in positive_mask are treated as targets.
    Loss: -log( sum_over_positives(exp(sim)) / sum_over_all(exp(sim)) )
    """
    N, M = similarity.shape
    # Mask out NaN/Inf
    similarity = torch.nan_to_num(similarity, nan=-1e4, posinf=1e4, neginf=-1e4)

    # Compute numerator: log-sum-exp over positives for each anchor
    pos_sim = similarity * positive_mask.float()
    pos_sim[pos_sim == 0] = -1e4
    pos_logsumexp = torch.logsumexp(pos_sim, dim=1)  # (N,)

    # Compute denominator: log-sum-exp over ALL candidates for each anchor
    all_logsumexp = torch.logsumexp(similarity, dim=1)  # (N,)

    # Loss per anchor
    loss_per_anchor = all_logsumexp - pos_logsumexp  # (N,)

    # Only compute loss for anchors that have at least one positive
    has_positive = positive_mask.sum(dim=1) > 0
    if has_positive.sum() == 0:
        return torch.tensor(0.0, device=similarity.device)

    return loss_per_anchor[has_positive].mean()


def get_patch_tokens_from_bbox(patch_tokens, bbox, b, original_image_size, image_size=224, patch_size=16):
    org_width, org_height = original_image_size
    x1 = int(round(bbox['x1'][b].item() * image_size / org_width))
    y1 = int(round(bbox['y1'][b].item() * image_size / org_height))
    x2 = int(round(bbox['x2'][b].item() * image_size / org_width))
    y2 = int(round(bbox['y2'][b].item() * image_size / org_height))
    x1 = max(0, min(x1, image_size-1))
    y1 = max(0, min(y1, image_size-1))
    x2 = max(0, min(x2, image_size))
    y2 = max(0, min(y2, image_size))
    patch_x1 = x1 // patch_size
    patch_y1 = y1 // patch_size
    patch_x2 = (x2 + patch_size - 1) // patch_size
    patch_y2 = (y2 + patch_size - 1) // patch_size
    num_patches = (image_size // patch_size)
    indices = []
    for i in range(patch_y1, patch_y2):
        for j in range(patch_x1, patch_x2):
            indices.append(i * num_patches + j + 1)
    relevant_tokens = patch_tokens[:, indices, :]
    pooled_tokens = torch.mean(relevant_tokens, dim=1)
    return pooled_tokens


def get_text_tokens_from_segment(text_tokens, org_text, seg_text, processor):
    org_text = ' '.join(org_text.split()).strip()
    seg_text = ' '.join(seg_text.split()).strip()
    sentences = org_text.split('.')
    sentences = [s.strip() for s in sentences if s.strip()]
    seg_pos = org_text.find(seg_text)
    current_pos = 0
    for i, sent in enumerate(sentences):
        sent = sent.strip()
        if sent == seg_text:
            seg_pos = current_pos
            break
        current_pos += len(sent) + 2
    assert seg_pos != -1, f"Segment text not found in original text"
    seg_tokens = processor(text=seg_text, return_tensors="pt", padding=False, truncation=False)
    seg_token_length = len(seg_tokens.input_ids[0]) - 2
    text_before = org_text[:seg_pos]
    tokens_before = processor(text=text_before, return_tensors="pt", padding=False, truncation=False)
    start_idx = len(tokens_before.input_ids[0])
    max_length = text_tokens.shape[1]
    if start_idx >= max_length:
        end_idx = max_length - 1
        start_idx = max(1, end_idx - seg_token_length)
    else:
        end_idx = min(start_idx + seg_token_length, max_length - 1)
    relevant_tokens = text_tokens[:, start_idx:end_idx, :]
    if relevant_tokens.shape[1] == 0:
        relevant_tokens = text_tokens[:, 1:min(1 + seg_token_length, max_length), :]
    pooled_tokens = torch.mean(relevant_tokens, dim=1)
    return pooled_tokens


class DLoaderMultiPositive(Dataset):
    """
    DataLoader that selects top-K segments per sample for multi-positive training.
    """
    def __init__(self, data_list, processor, new_max_token, top_k=3):
        self.data_list = data_list
        self.processor = processor
        self.new_max_token = new_max_token
        self.top_k = top_k

    def __len__(self):
        return len(self.data_list)

    def _load_image(self, name):
        img = Image.open(name).convert("RGB")
        return img, img.size

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Retry with random sample if current has missing files (NFS issues)
        max_retries = 1000
        orig_idx = idx
        for _ in range(max_retries):
            try:
                item = self.data_list[idx]
                org_image, org_image_size = self._load_image(item["original_filename"])
                break
            except FileNotFoundError:
                idx = random.randint(0, len(self.data_list) - 1)
        else:
            raise RuntimeError(f"Could not find valid sample after {max_retries} retries (started at {orig_idx})")

        org_caption = item["original_caption"]

        # Select top-K segments by similarity score
        segments = item["segment"]
        sorted_segs = sorted(segments, key=lambda x: x["similarity_score"], reverse=True)
        top_segs = sorted_segs[:min(self.top_k, len(sorted_segs))]
        k = len(top_segs)

        org_data = self.processor(images=org_image, text=org_caption, return_tensors="pt",
                                  truncation=True, padding="max_length", max_length=self.new_max_token)

        seg_images_list = []
        seg_texts_list = []
        bboxes_list = []
        for seg in top_segs:
            try:
                seg_img, _ = self._load_image(seg["filename"])
            except FileNotFoundError:
                continue  # Skip segments with missing files (NFS)
            seg_data = self.processor(images=seg_img, text=seg["caption"], return_tensors="pt",
                                      truncation=True, padding="max_length", max_length=self.new_max_token)
            seg_images_list.append(seg_data.pixel_values[0])
            seg_texts_list.append(seg_data.input_ids[0])
            bboxes_list.append(seg["bbox_coordinates"])

        return (org_data.pixel_values[0], org_data.input_ids[0],
                org_image_size, org_caption, item["original_filename"],
                seg_images_list, seg_texts_list, bboxes_list,
                [s["caption"] for s in top_segs], [s["filename"] for s in top_segs],
                k)


def collate_multi_positive(batch):
    """Custom collate for multi-positive batches."""
    org_images = torch.stack([b[0] for b in batch])
    org_texts = torch.stack([b[1] for b in batch])
    org_image_sizes = [b[2] for b in batch]
    org_captions = [b[3] for b in batch]
    org_filenames = [b[4] for b in batch]

    # Collect all seg images and texts
    all_seg_images = []
    all_seg_texts = []
    all_bboxes = []
    all_seg_captions = []
    all_seg_filenames = []
    seg_counts = []

    for b in batch:
        seg_counts.append(b[10])  # k for this sample
        all_seg_images.extend(b[5])
        all_seg_texts.extend(b[6])
        all_bboxes.extend(b[7])
        all_seg_captions.extend(b[8])
        all_seg_filenames.extend(b[9])

    seg_images = torch.stack(all_seg_images) if all_seg_images else torch.empty(0)
    seg_texts = torch.stack(all_seg_texts) if all_seg_texts else torch.empty(0)

    return (org_images, org_texts, org_image_sizes, org_captions, org_filenames,
            seg_images, seg_texts, all_bboxes, all_seg_captions, all_seg_filenames,
            seg_counts)


def build_multi_positive_mask(batch_size, seg_counts, device):
    """
    Build a multi-positive mask for contrastive learning.

    For each original image i (0..B-1), its positive segment indices
    in the seg list are [start_i, start_i + k_i).

    Returns:
        i2t_mask: (B, total_seg) mask for image→text matching
        t2i_mask: (total_seg, B) mask for text→image matching
    """
    total_seg = sum(seg_counts)
    i2t_mask = torch.zeros(batch_size, total_seg, device=device)

    seg_start = 0
    for i in range(batch_size):
        k = seg_counts[i]
        i2t_mask[i, seg_start:seg_start + k] = 1.0
        seg_start += k

    t2i_mask = i2t_mask.t()  # (total_seg, B)

    return i2t_mask, t2i_mask


def main(args):
    wandb.init(project="CLIP_Training_real", config=args)

    fabric = L.Fabric(
        accelerator="cuda",
        devices=args.world_size,
        strategy="auto",
        precision="bf16"
    )

    fabric.launch()
    fabric.seed_everything(1337 + fabric.global_rank)

    if fabric.global_rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)

    with open(args.dataset, encoding='utf-8') as f:
        train_list = json.load(f)

    with fabric.device:
        processor = transformers.AutoProcessor.from_pretrained(args.model)
        model = transformers.CLIPModel.from_pretrained(args.model)
        longclip_pos_embeddings(model, args.new_max_token)

        if args.ckpt or args.resume_epoch > 0:
            if args.ckpt:
                ckpt_path = args.ckpt
            else:
                ckpt_path = os.path.join(args.output_dir,
                    f"GOAL_multi_pos_{os.path.splitext(os.path.basename(args.model))[0]}_"
                    f"{os.path.splitext(os.path.basename(args.dataset))[0]}_{args.resume_epoch}_{args.image_size}.pth")
            if fabric.global_rank == 0:
                print(f"Loading checkpoint from {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            model.load_state_dict(checkpoint)
            if fabric.global_rank == 0:
                print("Checkpoint loaded successfully")

        print_trainable_parameters(fabric, model)

    dataset_train = DLoaderMultiPositive(train_list, processor, args.new_max_token, top_k=args.top_k)

    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        shuffle=True,
        collate_fn=collate_multi_positive,
    )

    train_loader = fabric.setup_dataloaders(train_loader)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.init_lr, weight_decay=args.weight_decay)
    model, optimizer = fabric.setup(model, optimizer)

    train(fabric, model, optimizer, train_loader, processor, args)


def train(fabric, model, optimizer, train_loader, processor, args):
    mse_loss = torch.nn.MSELoss()
    iter_count = args.resume_epoch * len(train_loader)
    total_iter = len(train_loader) * args.epochs

    for epoch in range(args.resume_epoch, args.epochs):
        epoch_loss = 0.0
        epoch_loss_org = 0.0
        epoch_loss_seg = 0.0
        epoch_loss_patch = 0.0
        epoch_loss_text = 0.0
        epoch_loss_multi = 0.0

        for i, batch in enumerate(train_loader):
            # Cosine LR
            lr = (args.init_lr - args.min_lr) * 0.5 * (1.0 + math.cos(math.pi * iter_count / total_iter)) + args.min_lr
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            (org_image, org_text, org_image_sizes, org_captions, org_filenames,
             seg_images, seg_texts, bboxes, seg_captions, seg_filenames,
             seg_counts) = batch

            batch_size = org_image.shape[0]
            total_seg = seg_images.shape[0]
            eps = 1e-8

            # ==== Global embeddings ====
            all_images = torch.cat([org_image, seg_images], dim=0)
            all_texts = torch.cat([org_text, seg_texts], dim=0)
            outputs = model(pixel_values=all_images, input_ids=all_texts, output_hidden_states=True)

            org_image_embeds = outputs.image_embeds[:batch_size]
            seg_image_embeds = outputs.image_embeds[batch_size:]
            org_text_embeds = outputs.text_embeds[:batch_size]
            seg_text_embeds = outputs.text_embeds[batch_size:]

            # ==== Multi-positive masks ====
            if args.enable_multi_positive:
                i2t_mask, t2i_mask = build_multi_positive_mask(batch_size, seg_counts, org_image.device)

                # Normalize embeddings
                org_i_norm = F.normalize(org_image_embeds, dim=-1)
                seg_t_norm = F.normalize(seg_text_embeds, dim=-1)
                org_t_norm = F.normalize(org_text_embeds, dim=-1)
                seg_i_norm = F.normalize(seg_image_embeds, dim=-1)

                # Multi-positive: org_image → all seg_texts
                sim_i2t = model.logit_scale.exp() * (org_i_norm @ seg_t_norm.t())  # (B, total_seg)
                loss_i2t = multi_positive_clip_loss(sim_i2t, i2t_mask)

                # Multi-positive: seg_text → all org_images
                sim_t2i = model.logit_scale.exp() * (seg_t_norm @ org_i_norm.t())  # (total_seg, B)
                loss_t2i = multi_positive_clip_loss(sim_t2i, t2i_mask)

                loss_multi = (loss_i2t + loss_t2i) / 2.0
            else:
                loss_multi = torch.tensor(0.0, device=org_image.device)

            # ==== Original CLIP loss (org↔org) ====
            x_i_org = F.normalize(org_image_embeds + eps)
            x_t_org = F.normalize(org_text_embeds + eps)
            sim_org = model.logit_scale.exp() * x_i_org @ x_t_org.t()
            loss_org = clip_loss(sim_org)

            # ==== Segment CLIP loss: best seg image ↔ best seg text ====
            best_i_idx = [sum(seg_counts[:b]) for b in range(batch_size)]
            best_seg_i = seg_image_embeds[best_i_idx]
            best_seg_t = seg_text_embeds[best_i_idx]
            x_i_best = F.normalize(best_seg_i + eps)
            x_t_best = F.normalize(best_seg_t + eps)
            sim_seg = model.logit_scale.exp() * x_i_best @ x_t_best.t()
            loss_seg = clip_loss(sim_seg)

            # ==== Build best-segment indices ====
            best_indices = []
            s = 0
            for b in range(batch_size):
                best_indices.append(s)
                s += seg_counts[b]

            # ==== Patch features from best segment bboxes ====
            vision_outputs = model.vision_model(seg_images, output_hidden_states=True)
            org_patch_tokens = vision_outputs.hidden_states[-1]

            patch_pooled_list = []
            seg_start = 0
            for b in range(batch_size):
                bbox_dict = bboxes[seg_start]
                bbox = {k: torch.tensor([v], device=org_image.device) for k, v in bbox_dict.items()}
                img_w = org_image_sizes[b][0]
                img_h = org_image_sizes[b][1]
                pooled = get_patch_tokens_from_bbox(
                    org_patch_tokens[seg_start:seg_start + 1], bbox, 0,
                    (img_w, img_h), image_size=args.image_size, patch_size=16)
                patch_pooled_list.append(pooled)
                seg_start += seg_counts[b]

            patch_pooled = torch.cat(patch_pooled_list, dim=0)
            patch_pooled = model.vision_model.post_layernorm(patch_pooled)
            patch_pooled = model.visual_projection(patch_pooled)
            patch_pooled = F.normalize(patch_pooled + eps, dim=-1)
            seg_image_embeds_norm = F.normalize(seg_image_embeds + eps, dim=-1)

            # ==== Text segment features from best segments ====
            text_outputs = model.text_model(all_texts[:batch_size], output_hidden_states=True)
            org_text_tokens = text_outputs.hidden_states[-1][:batch_size]

            text_pooled_list = []
            for b in range(batch_size):
                best_cap = seg_captions[sum(seg_counts[:b])]
                pooled = get_text_tokens_from_segment(
                    org_text_tokens[b:b + 1], org_captions[b], best_cap, processor)
                text_pooled_list.append(pooled)

            text_pooled = torch.cat(text_pooled_list, dim=0)
            text_pooled = model.text_model.final_layer_norm(text_pooled)
            text_pooled = model.text_projection(text_pooled)
            text_pooled = F.normalize(text_pooled + eps, dim=-1)
            seg_text_embeds_norm = F.normalize(seg_text_embeds + eps, dim=-1)

            if args.enable_local_infonce:
                # ==== Local InfoNCE: patch ↔ all seg images ====
                tau = args.local_temperature
                sim_patch = (patch_pooled @ seg_image_embeds_norm.t()) / tau  # (B, total_seg)
                # Positive mask: each anchor's positive is its best segment
                patch_pos_mask = torch.zeros(batch_size, total_seg, device=org_image.device)
                for b in range(batch_size):
                    patch_pos_mask[b, best_indices[b]] = 1.0
                loss_patch = multi_positive_clip_loss(sim_patch, patch_pos_mask)

                # ==== Local InfoNCE: text segment ↔ all seg texts ====
                sim_text = (text_pooled @ seg_text_embeds_norm.t()) / tau  # (B, total_seg)
                text_pos_mask = patch_pos_mask  # same: best segment per original
                loss_text = multi_positive_clip_loss(sim_text, text_pos_mask)
            else:
                # Fallback: original MSE loss
                sim_patch_mse = patch_pooled @ seg_image_embeds_norm[best_indices].t()
                loss_patch = mse_loss(torch.diag(sim_patch_mse), torch.ones(batch_size, device=org_image.device))
                sim_text_mse = text_pooled @ seg_text_embeds_norm[best_indices].t()
                loss_text = mse_loss(torch.diag(sim_text_mse), torch.ones(batch_size, device=org_image.device))

            # Logging: best-segment diagonal similarity (for monitoring collapse)
            patch_diag = torch.diag(patch_pooled @ seg_image_embeds_norm[best_indices].t())
            text_diag = torch.diag(text_pooled @ seg_text_embeds_norm[best_indices].t())

            # ==== Total loss ====
            loss = loss_org + 0.5 * loss_seg + loss_patch + loss_text + loss_multi

            epoch_loss += loss.item()
            epoch_loss_org += loss_org.item()
            epoch_loss_seg += loss_seg.item()
            epoch_loss_patch += loss_patch.item()
            epoch_loss_text += loss_text.item()
            epoch_loss_multi += loss_multi.item()

            fabric.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            if fabric.global_rank == 0:
                wandb.log({
                    "iter": iter_count, "lr": lr, "loss": loss.item(),
                    "loss_org": loss_org.item(), "loss_seg": loss_seg.item(),
                    "loss_patch": loss_patch.item(), "loss_text": loss_text.item(),
                    "loss_multi": loss_multi.item(),
                    "epoch": epoch, "progress": (iter_count / total_iter) * 100,
                    "logit_scale": model.logit_scale.exp().item(),
                    "patch_similarity": patch_diag.mean().item(),
                    "text_similarity": text_diag.mean().item(),
                })

            fabric.print(f"epoch {epoch} iter {iter_count} ({(iter_count/total_iter)*100:.4f}%) "
                         f"lr {lr:.6f} loss {loss.item():.4f} "
                         f"org:{loss_org.item():.4f} seg:{loss_seg.item():.4f} "
                         f"multi:{loss_multi.item():.4f} "
                         f"patch:{loss_patch.item():.4f} text:{loss_text.item():.4f} "
                         f"patch_sim:{patch_diag.mean().item():.4f} text_sim:{text_diag.mean().item():.4f}")
            iter_count += 1

        # Save checkpoint
        save_path = os.path.join(args.output_dir,
                                 f"GOAL_multi_pos_{os.path.splitext(os.path.basename(args.model))[0]}_"
                                 f"{os.path.splitext(os.path.basename(args.dataset))[0]}_{epoch+1}_{args.image_size}.pth")
        fabric.barrier()
        if fabric.global_rank == 0:
            model_state_dict = model.state_dict()
            cpu_state_dict = {k: v.cpu() for k, v in model_state_dict.items()}
            torch.save(cpu_state_dict, save_path)
            fabric.print(f"Model saved to {save_path}")
        fabric.barrier()


def get_args_parser():
    parser = argparse.ArgumentParser('GOAL Multi-Positive Training', add_help=False)
    parser.add_argument('--batch_size', default=4, type=int, help='Batch size per GPU')
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--image_size', default=224, type=int)
    parser.add_argument('--new_max_token', default=248, type=int)
    parser.add_argument('--dataset', default='datasets/DCI_segment_sim_bbox_del_org.json', type=str)
    parser.add_argument('--model', default='openai/clip-vit-base-patch16', type=str)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--init_lr', type=float, default=5e-6, metavar='LR')
    parser.add_argument('--min_lr', type=float, default=0, metavar='LR')
    parser.add_argument('--output_dir', default='finetune_out_multi_pos',
                        help='path where to save')
    parser.add_argument('--save_interval', default=1, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--pin_mem', action='store_true', help='Pin CPU memory')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.add_argument('--ckpt', type=str, default=None, help='path to checkpoint')
    parser.add_argument('--resume_epoch', type=int, default=0, help='Resume from this epoch')
    parser.add_argument('--world_size', default=1, type=int, help='distributed processes')

    # Multi-positive specific
    parser.add_argument('--top_k', default=3, type=int, help='Number of top segments per sample')
    parser.add_argument('--enable_multi_positive', default=True, action='store_true',
                        help='Enable multi-positive contrastive loss')
    parser.add_argument('--disable_multi_positive', action='store_true',
                        help='Disable multi-positive contrastive loss')
    # Local InfoNCE
    parser.add_argument('--enable_local_infonce', default=True, action='store_true',
                        help='Use InfoNCE for local alignment (instead of MSE)')
    parser.add_argument('--disable_local_infonce', action='store_true',
                        help='Disable local InfoNCE, use MSE fallback')
    parser.add_argument('--local_temperature', type=float, default=0.07,
                        help='Temperature for local InfoNCE loss')
    parser.set_defaults(pin_mem=True)
    return parser


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.disable_multi_positive:
        args.enable_multi_positive = False
    if args.disable_local_infonce:
        args.enable_local_infonce = False
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
