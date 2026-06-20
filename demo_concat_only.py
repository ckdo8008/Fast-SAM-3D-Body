import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch

parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, parent_dir)

from notebook.utils import setup_sam_3d_body
from tools.vis_utils import visualize_sample_together


def apply_run_demo_defaults():
    defaults = {
        "GPU_HAND_PREP": "1",
        "LAYER_DTYPE": "fp32",
        "SKIP_KEYPOINT_PROMPT": "1",
        "IMG_SIZE": "512",
        "USE_COMPILE": "1",
        "USE_COMPILE_BACKBONE": "1",
        "DECODER_COMPILE": "1",
        "COMPILE_MODE": "reduce-overhead",
        "COMPILE_WARMUP_BATCH_SIZES": "1",
        "MHR_USE_CUDA_GRAPH": "0",
        "KEYPOINT_PROMPT_INTERM_INTERVAL": "999",
        "BODY_INTERM_PRED_LAYERS": "0,1,2",
        "HAND_INTERM_PRED_LAYERS": "0,1",
        "MHR_NO_CORRECTIVES": "1",
        "FOV_TRT": "1",
        "FOV_FAST": "1",
        "FOV_MODEL": "s",
        "FOV_LEVEL": "0",
        "DEBUG_NAN": "0",
        "DEBUG_HAND_PREP": "0",
        "DEBUG_BACKBONE_INPUT": "0",
        "INTERM_TIMING": "0",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)


def main(args):
    apply_run_demo_defaults()

    t_start = time.time()

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)

    print("=" * 60)
    print("SAM 3D Body - Concat Image Only")
    print("=" * 60)
    print(f"Image: {args.image_path}")
    print(f"Output: {args.output_path}")
    print(f"Detector: {args.detector} ({args.detector_model})")
    print(f"Hand Box Source: {args.hand_box_source}")

    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"Image not found: {args.image_path}")

    estimator = setup_sam_3d_body(
        hf_repo_id=args.model,
        detector_name=args.detector,
        detector_model=args.detector_model,
        local_checkpoint_path=args.local_checkpoint,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    timings = []
    outputs_for_render = None
    print("-" * 60)
    print(f"Running image processing {args.runs} times...")
    print("-" * 60)

    for run_idx in range(args.runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        outputs = estimator.process_one_image(
            args.image_path,
            hand_box_source=args.hand_box_source,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = time.time() - t0
        timings.append(elapsed)
        print(f"Run {run_idx + 1:02d}/{args.runs}: {elapsed:.4f}s")

        if run_idx == args.render_run - 1:
            outputs_for_render = outputs

    if outputs_for_render is None:
        outputs_for_render = outputs

    if not outputs_for_render:
        raise RuntimeError("No person detected; concat image was not created.")

    timings_np = np.array(timings, dtype=np.float64)
    print("-" * 60)
    print("Image processing timing")
    print("-" * 60)
    print(f"Runs: {args.runs}")
    print(f"Average: {timings_np.mean():.4f}s")
    print(f"Std: {timings_np.std():.4f}s")
    print(f"Min: {timings_np.min():.4f}s")
    print(f"Max: {timings_np.max():.4f}s")

    img_cv2 = cv2.imread(args.image_path)
    if img_cv2 is None:
        raise RuntimeError(f"Failed to read image: {args.image_path}")

    t0 = time.time()
    concat_img = visualize_sample_together(img_cv2, outputs_for_render, estimator.faces)
    cv2.imwrite(args.output_path, concat_img.astype("uint8"))
    print(f"Render+save: {time.time() - t0:.4f}s")

    print("=" * 60)
    print(f"Saved: {args.output_path}")
    print(f"Total: {time.time() - t_start:.4f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run SAM 3D Body and save only the concat visualization image."
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default="./notebook/images/dancing.jpg",
        help="Input image path.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./output/concat_only.jpg",
        help="Output concat image path.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/sam-3d-body-dinov3",
        choices=["facebook/sam-3d-body-dinov3", "facebook/sam-3d-body-vith"],
        help="SAM 3D Body model id.",
    )
    parser.add_argument(
        "--detector",
        type=str,
        default="yolo_pose",
        choices=["vitdet", "yolo", "yolo_pose"],
        help="Person detector.",
    )
    parser.add_argument(
        "--detector_model",
        type=str,
        default="./checkpoints/yolo/yolo11m-pose.pt",
        help="Detector model path. Defaults to the same YOLO-Pose model as run_demo.sh.",
    )
    parser.add_argument(
        "--hand_box_source",
        type=str,
        default="yolo_pose",
        choices=["body_decoder", "yolo_pose"],
        help="Hand box source for full-body inference.",
    )
    parser.add_argument(
        "--local_checkpoint",
        type=str,
        default="./checkpoints/sam-3d-body-dinov3",
        help="Local checkpoint directory containing model.ckpt and model_config.yaml.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of consecutive image-processing runs to time.",
    )
    parser.add_argument(
        "--render_run",
        type=int,
        default=1,
        help="1-based run index used for the saved concat image.",
    )

    parsed_args = parser.parse_args()
    if parsed_args.runs < 1:
        raise ValueError("--runs must be at least 1")
    if parsed_args.render_run < 1 or parsed_args.render_run > parsed_args.runs:
        raise ValueError("--render_run must be between 1 and --runs")
    main(parsed_args)
