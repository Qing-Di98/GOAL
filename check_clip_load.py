# check_clip_load.py
import argparse
import torch
import transformers

from utils.func import longclip_pos_embeddings


def unwrap_ckpt(ckpt):
    if isinstance(ckpt, dict):
        for key in ["state_dict", "model", "model_state_dict"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    return ckpt


def strip_prefix(state_dict):
    new_state = {}
    for k, v in state_dict.items():
        for prefix in ["module.", "model."]:
            if k.startswith(prefix):
                k = k[len(prefix):]
        new_state[k] = v
    return new_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="openai/clip-vit-base-patch16")
    parser.add_argument("--new_max_token", type=int, default=248)
    parser.add_argument("--ckpt", default=None, help="optional fine-tuned checkpoint path")
    args = parser.parse_args()

    print(f"Loading base CLIP: {args.model}")
    model = transformers.CLIPModel.from_pretrained(args.model)

    # 和训练代码保持一致：如果训练时用了 248 token，这里也要先扩展位置编码
    longclip_pos_embeddings(model, args.new_max_token)

    print("Base CLIP loaded successfully.")
    print("Text position embedding:", tuple(model.text_model.embeddings.position_embedding.weight.shape))
    print("Vision patch embedding:", tuple(model.vision_model.embeddings.patch_embedding.weight.shape))
    print("Logit scale:", model.logit_scale.exp().item())

    if args.ckpt is not None:
        print(f"\nLoading checkpoint: {args.ckpt}")
        ckpt = torch.load(args.ckpt, map_location="cpu")
        ckpt = strip_prefix(unwrap_ckpt(ckpt))

        incompatible = model.load_state_dict(ckpt, strict=False)

        print("\nCheckpoint load report:")
        print("Missing keys:", len(incompatible.missing_keys))
        print("Unexpected keys:", len(incompatible.unexpected_keys))

        if incompatible.missing_keys[:10]:
            print("First missing keys:", incompatible.missing_keys[:10])
        if incompatible.unexpected_keys[:10]:
            print("First unexpected keys:", incompatible.unexpected_keys[:10])

        model_state = model.state_dict()
        matched = 0
        checked = 0
        max_diff = 0.0

        for k, v in ckpt.items():
            if k in model_state and model_state[k].shape == v.shape:
                checked += 1
                diff = (model_state[k].cpu() - v.cpu()).abs().max().item()
                max_diff = max(max_diff, diff)
                if torch.allclose(model_state[k].cpu(), v.cpu(), atol=1e-6):
                    matched += 1

        print(f"\nMatched tensors: {matched}/{checked}")
        print(f"Max abs diff after loading: {max_diff:.8f}")

        if matched == checked and checked > 0:
            print("Checkpoint weights loaded successfully.")
        else:
            print("Some checkpoint tensors were not exactly matched. Check missing/unexpected keys above.")

    # 简单 forward 检查
    processor = transformers.AutoProcessor.from_pretrained(args.model)
    inputs = processor(
        text=["a photo of a cat"],
        images=torch.zeros(224, 224, 3, dtype=torch.uint8).numpy(),
        return_tensors="pt",
        padding="max_length",
        max_length=args.new_max_token,
    )

    with torch.no_grad():
        outputs = model(**inputs)

    print("\nForward check:")
    print("image_embeds:", tuple(outputs.image_embeds.shape))
    print("text_embeds:", tuple(outputs.text_embeds.shape))
    print("Forward succeeded.")


if __name__ == "__main__":
    main()