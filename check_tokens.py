import torch
from models.dinov3_encoder import DINOv3Encoder


BACKBONE_NAME = "dinov3_model"

# 建议用 CPU，不抢正在训练的 GPU
device = torch.device("cpu")


encoder = DINOv3Encoder(
    BACKBONE_NAME,
    trainable=False
).to(device)

encoder.eval()


print("=" * 60)
print("DINOv3 CONFIG")
print("=" * 60)

print("model path/name:",
      BACKBONE_NAME)

print("patch_size:",
      getattr(
          encoder.backbone.config,
          "patch_size",
          "N/A"
      ))

print("num_register_tokens:",
      getattr(
          encoder.backbone.config,
          "num_register_tokens",
          "N/A"
      ))

print("hidden_size:",
      getattr(
          encoder.backbone.config,
          "hidden_size",
          "N/A"
      ))

print("model_type:",
      getattr(
          encoder.backbone.config,
          "model_type",
          "N/A"
      ))


def check(size):

    print("\n" + "=" * 60)
    print(f"INPUT = {size} x {size}")
    print("=" * 60)

    x = torch.randn(
        1,
        3,
        size,
        size,
        device=device
    )

    with torch.inference_mode():

        outputs = encoder.backbone(
            pixel_values=x,
            output_hidden_states=True
        )

    hs = outputs.hidden_states

    print("number of hidden states:",
          len(hs))

    print("hs[-3]:",
          tuple(hs[-3].shape))

    print("hs[-2]:",
          tuple(hs[-2].shape))

    print("hs[-1]:",
          tuple(hs[-1].shape))

    B, N, C = hs[-1].shape

    print()
    print("total token count:", N)
    print("embedding dim:", C)

    patch_size = getattr(
        encoder.backbone.config,
        "patch_size",
        None
    )

    num_register_tokens = getattr(
        encoder.backbone.config,
        "num_register_tokens",
        None
    )

    if patch_size is not None:

        grid = size // patch_size

        expected_patch_tokens = (
            grid * grid
        )

        print()
        print("size // patch_size:",
              grid)

        print("expected patch grid:",
              f"{grid} x {grid}")

        print("expected patch tokens:",
              expected_patch_tokens)

        print("extra tokens:",
              N - expected_patch_tokens)

    if num_register_tokens is not None:

        print(
            "expected special tokens "
            "(CLS + registers):",
            1 + num_register_tokens
        )

    # ---------------------------------------------------
    # 模拟你目前训练代码里的切法
    # 不修改模型
    # ---------------------------------------------------

    current_slice = hs[-1][:, 1:-4, :]

    num_after_slice = (
        current_slice.shape[1]
    )

    side = int(
        num_after_slice ** 0.5
    )

    print()
    print("--- CURRENT CODE TEST ---")

    print(
        "feat[:, 1:-4, :] shape:",
        tuple(current_slice.shape)
    )

    print(
        "tokens after slice:",
        num_after_slice
    )

    print(
        "sqrt(tokens):",
        side
    )

    print(
        "side * side:",
        side * side
    )

    print(
        "can reshape square:",
        side * side == num_after_slice
    )


check(784)
check(1024)