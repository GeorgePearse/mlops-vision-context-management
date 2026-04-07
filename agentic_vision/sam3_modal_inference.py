# type: ignore
"""Modal app for SAM3 inference with point prompt support."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import modal
from loguru import logger

SAM3_REPO_DIR = Path("/opt/sam3")
MODELS_ROOT = Path("/opt/pipelines/models")
HANDLER_CONFIGS_ROOT = Path("/opt/pipelines/handler_configs")
MODEL_VOLUME_MOUNT = "/root/artifacts"
MODAL_TIMEOUT_SECONDS = 3600
SAM3_DEFAULT_HANDLER_NAME = os.getenv("SAM3_HANDLER_NAME", "premier_sam3")
MAX_INFERENCE_CONTAINERS = 5

# Paths relative to the core repo for building the image
CORE_REPO_ROOT = Path("/home/georgepearse/core-worktrees/main")
LOCAL_SAM3_REPO_DIR = CORE_REPO_ROOT / "machine_learning/packages/sam3"
LOCAL_MODELS_DIR = CORE_REPO_ROOT / "lib/python/pipelines/pipelines/models"
LOCAL_HANDLER_CONFIGS_DIR = CORE_REPO_ROOT / "lib/python/pipelines/pipelines/handler_configs"
LOCAL_API_DIR = CORE_REPO_ROOT / "backend/api/api"
LOCAL_MODEL_URI_MAP = CORE_REPO_ROOT / "backend/sam3_inference/sam3_inference/model_uris.yaml"
API_PYTHON_ROOT = "/opt/backend_api"
CONTAINER_MODEL_URI_MAP_PATH = Path("/root/model_uris.yaml")

image = (
    modal.Image.from_registry("nvcr.io/nvidia/cuda:12.6.0-devel-ubuntu22.04", add_python="3.12")
    .run_commands(
        "pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126",
        "pip install google-cloud-storage pillow pyyaml loguru",
    )
    .add_local_dir(LOCAL_SAM3_REPO_DIR, remote_path=str(SAM3_REPO_DIR), copy=True)
    .add_local_dir(LOCAL_MODELS_DIR, remote_path=str(MODELS_ROOT), copy=True)
    .add_local_dir(LOCAL_HANDLER_CONFIGS_DIR, remote_path=str(HANDLER_CONFIGS_ROOT), copy=True)
    .add_local_dir(LOCAL_API_DIR, remote_path=f"{API_PYTHON_ROOT}/api", copy=True)
    .add_local_file(str(LOCAL_MODEL_URI_MAP), remote_path=str(CONTAINER_MODEL_URI_MAP_PATH), copy=True)
    .run_commands(f'cd {SAM3_REPO_DIR} && pip install -e ".[train,dev,notebooks]"')
    .env({"PYTHONPATH": API_PYTHON_ROOT})
)

app = modal.App(name="sam3-vision-agents", image=image)

model_volume = modal.Volume.from_name(
    "sam3-checkpoint-cache",
    environment_name="main",
)


@dataclass
class SegmentationResult:
    """Result of segmenting a single box."""
    label: str
    bbox: tuple[float, float, float, float]
    points: list[tuple[float, float]]
    score: float
    segmentation_area: float


def _configure_gcp_credentials() -> str:
    """Configure GOOGLE_APPLICATION_CREDENTIALS from MODEL_TRAINING_SA_KEY."""
    sa_key = os.environ.get("MODEL_TRAINING_SA_KEY")
    if not sa_key:
        raise ValueError("MODEL_TRAINING_SA_KEY environment variable is missing.")

    gcp_key_data = json.loads(sa_key)
    with tempfile.NamedTemporaryFile(delete=False) as temp_key_file:
        temp_key_file.write(json.dumps(gcp_key_data).encode())
        temp_key_path = temp_key_file.name

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_key_path
    return temp_key_path


def _parse_gcs_uri(uri: str) -> tuple[str, str]:
    """Split a GCS URI into bucket and blob path."""
    if not uri.startswith("gs://"):
        raise ValueError(f"Expected GCS URI, got {uri}")
    bucket, _, blob = uri[5:].partition("/")
    if not bucket or not blob:
        raise ValueError(f"Invalid GCS URI: {uri}")
    return bucket, blob


def _download_gcs_image(frame_uri: str, destination: Path) -> None:
    """Download the GCS object at frame_uri to destination."""
    from google.cloud import storage

    bucket_name, blob_name = _parse_gcs_uri(frame_uri)
    client = storage.Client()
    blob = client.bucket(bucket_name).blob(blob_name)
    blob.download_to_filename(destination)


def _load_model_uri_map() -> dict[str, str]:
    """Load model name -> checkpoint URI mapping from model_uris.yaml."""
    import yaml

    if not CONTAINER_MODEL_URI_MAP_PATH.exists():
        return {}
    return yaml.safe_load(CONTAINER_MODEL_URI_MAP_PATH.read_text()) or {}


def _resolve_model_name_from_handler(handler_name: str) -> str:
    """Resolve model name from handler config YAML."""
    import yaml

    config_path = HANDLER_CONFIGS_ROOT / f"{handler_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Handler config not found: {config_path}")
    config = yaml.safe_load(config_path.read_text()) or {}
    model_name = config.get("model_name")
    if not model_name:
        raise ValueError(f"Handler config {config_path} missing 'model_name'")
    return model_name


def _checkpoint_path_for_model(model_name: str) -> str:
    """Get checkpoint path for a model name."""
    uri_map = _load_model_uri_map()
    if model_name in uri_map:
        return uri_map[model_name]
    checkpoint_path = MODELS_ROOT / model_name / "checkpoint.pt"
    if checkpoint_path.exists():
        return str(checkpoint_path)
    raise FileNotFoundError(f"No checkpoint found for model '{model_name}'")


def _resolve_checkpoint_path(checkpoint_uri: str, model_name: str) -> Path:
    """Resolve checkpoint URI to local path, downloading from GCS if needed."""
    from google.cloud import storage

    if checkpoint_uri.startswith("gs://"):
        local_path = Path(MODEL_VOLUME_MOUNT) / model_name / "checkpoint.pt"
        if local_path.exists():
            return local_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        bucket_name, blob_name = _parse_gcs_uri(checkpoint_uri)
        client = storage.Client()
        blob = client.bucket(bucket_name).blob(blob_name)
        blob.download_to_filename(local_path)
        return local_path
    return Path(checkpoint_uri)


def _maybe_convert_checkpoint(checkpoint_path: Path) -> Path:
    """Convert checkpoint format if needed."""
    import torch

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model" in ckpt:
        return checkpoint_path

    converted_path = checkpoint_path.parent / "checkpoint_converted.pt"
    if converted_path.exists():
        return converted_path

    torch.save({"model": ckpt}, converted_path)
    return converted_path


def _bbox_points(bbox: tuple[float, float, float, float]) -> list[tuple[float, float]]:
    """Convert bbox to corner points as fallback polygon."""
    x1, y1, x2, y2 = bbox
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def _mask_to_polygon_points(
    mask_tensor,
    width: int,
    height: int,
    fallback_bbox: tuple[float, float, float, float],
) -> list[tuple[float, float]]:
    """Convert binary mask to polygon points in normalized [0,1] coords."""
    import cv2
    import numpy as np

    mask_np = (mask_tensor > 0.5).numpy().astype(np.uint8)
    if mask_np.sum() == 0:
        return _bbox_points(fallback_bbox)

    mask_resized = cv2.resize(mask_np, (width, height), interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return _bbox_points(fallback_bbox)

    largest = max(contours, key=cv2.contourArea)
    epsilon = 0.005 * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon, True)

    points = [(float(pt[0][0]) / width, float(pt[0][1]) / height) for pt in approx]
    return points if len(points) >= 3 else _bbox_points(fallback_bbox)


def _pairwise_iou_xyxy(boxes1, boxes2):
    """Compute pairwise IoU between two sets of xyxy boxes."""
    import torch

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=1e-6)


@app.function(
    gpu="A10G",
    timeout=MODAL_TIMEOUT_SECONDS,
    max_containers=MAX_INFERENCE_CONTAINERS,
    volumes={MODEL_VOLUME_MOUNT: model_volume},
    secrets=[modal.Secret.from_name("model-training", environment_name="main")],
    retries=0,
)
def segment_boxes_with_points(
    frame_uri: str,
    boxes: list[list[float]],
    handler_name: str,
    positive_points: list[list[list[float]]] | None = None,
    negative_points: list[list[list[float]]] | None = None,
) -> list[dict[str, Any]]:
    """Generate segmentation masks for bounding boxes with optional point prompts.

    Args:
        frame_uri: GCS URI for the frame image.
        boxes: List of [x1, y1, x2, y2] bounding boxes in normalized [0, 1] coords.
        handler_name: Handler name to resolve model checkpoint.
        positive_points: Per-box list of positive point prompts [[x, y], ...] in [0, 1] coords.
        negative_points: Per-box list of negative point prompts [[x, y], ...] in [0, 1] coords.

    Returns:
        List of result dicts with keys: label, bbox, points, score, segmentation_area.
    """
    import sam3.model.sam3_image_processor as sam3_image_processor
    import torch
    from PIL import Image, ImageOps
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    sam3_image_processor.PILImage = Image.Image

    resolved_handler_name = handler_name.strip()
    if not resolved_handler_name:
        raise ValueError("handler_name must be provided.")

    # Initialize point lists if not provided
    if positive_points is None:
        positive_points = [[] for _ in boxes]
    if negative_points is None:
        negative_points = [[] for _ in boxes]

    # Resolve checkpoint
    direct_checkpoint_uri = _load_model_uri_map().get(resolved_handler_name)
    if direct_checkpoint_uri:
        resolved_model_name = resolved_handler_name
    else:
        resolved_model_name = _resolve_model_name_from_handler(resolved_handler_name)
        direct_checkpoint_uri = _checkpoint_path_for_model(resolved_model_name)

    logger.info(
        f"SAM3 segment_boxes_with_points: model={resolved_model_name}, "
        f"frame={frame_uri}, boxes={len(boxes)}, "
        f"has_positive_points={any(positive_points)}, has_negative_points={any(negative_points)}"
    )

    temp_key_path: str | None = None
    image_path: Path | None = None
    converted_checkpoint: Path | None = None
    try:
        temp_key_path = _configure_gcp_credentials()
        with tempfile.NamedTemporaryFile(suffix=Path(frame_uri).suffix or ".jpg", delete=False) as temp_image:
            image_path = Path(temp_image.name)
        _download_gcs_image(frame_uri, image_path)

        image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
        width, height = image.size

        checkpoint = _resolve_checkpoint_path(direct_checkpoint_uri, resolved_model_name)
        converted_checkpoint = _maybe_convert_checkpoint(checkpoint)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = build_sam3_image_model(
            bpe_path=str(SAM3_REPO_DIR / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"),
            checkpoint_path=str(converted_checkpoint),
            load_from_hf=False,
            device=device,
            eval_mode=True,
            enable_segmentation=True,
        )
        processor = Sam3Processor(model=model, device=device, confidence_threshold=0.01)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            state = processor.set_image(image)
            results: list[dict[str, Any]] = []

            for idx, box in enumerate(boxes):
                x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
                fallback_bbox = (x1, y1, x2, y2)
                fallback_points = _bbox_points(fallback_bbox)

                # Convert xyxy normalized to cxcywh normalized for add_geometric_prompt
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                bw = x2 - x1
                bh = y2 - y1

                if bw <= 0 or bh <= 0:
                    results.append({
                        "label": "box",
                        "bbox": fallback_bbox,
                        "points": fallback_points,
                        "score": 0.0,
                        "segmentation_area": 0.0,
                    })
                    continue

                try:
                    processor.reset_all_prompts(state)

                    # Add box prompt
                    state = processor.add_geometric_prompt(
                        box=[cx, cy, bw, bh],
                        label=True,
                        state=state,
                    )

                    # Add positive point prompts if provided
                    box_positive_points = positive_points[idx] if idx < len(positive_points) else []
                    box_negative_points = negative_points[idx] if idx < len(negative_points) else []

                    if box_positive_points or box_negative_points:
                        geometric_prompt = state.get("geometric_prompt")
                        if geometric_prompt is not None:
                            all_points = []
                            all_labels = []

                            for pt in box_positive_points:
                                all_points.append([float(pt[0]), float(pt[1])])
                                all_labels.append(True)  # positive

                            for pt in box_negative_points:
                                all_points.append([float(pt[0]), float(pt[1])])
                                all_labels.append(False)  # negative

                            if all_points:
                                points_tensor = torch.tensor(all_points, device=device, dtype=torch.float32).view(len(all_points), 1, 2)
                                labels_tensor = torch.tensor(all_labels, device=device, dtype=torch.bool).view(len(all_points), 1)
                                geometric_prompt.append_points(points_tensor, labels_tensor)

                                # Re-run inference with updated prompts
                                state = processor._forward_grounding(state)

                    masks = state.get("masks")
                    detected_boxes = state.get("boxes")
                    scores = state.get("scores")

                    if masks is None or detected_boxes is None or scores is None or masks.shape[0] == 0:
                        results.append({
                            "label": "box",
                            "bbox": fallback_bbox,
                            "points": fallback_points,
                            "score": 0.0,
                            "segmentation_area": 0.0,
                        })
                        continue

                    # Pick the detection with highest IoU to our input box
                    input_box_px = torch.tensor(
                        [[x1 * width, y1 * height, x2 * width, y2 * height]],
                        dtype=torch.float32,
                    )
                    det_boxes_cpu = detected_boxes.detach().cpu().to(dtype=torch.float32)
                    ious = _pairwise_iou_xyxy(input_box_px, det_boxes_cpu).squeeze(0)
                    best_idx = int(ious.argmax().item())

                    mask_tensor = masks[best_idx].detach().cpu()
                    if mask_tensor.ndim == 3:
                        mask_tensor = mask_tensor[0]

                    segmentation_area = float(mask_tensor.float().mean().item())
                    points = _mask_to_polygon_points(mask_tensor, width, height, fallback_bbox)
                    results.append({
                        "label": "box",
                        "bbox": fallback_bbox,
                        "points": points,
                        "score": float(scores[best_idx].item()),
                        "segmentation_area": segmentation_area,
                    })
                except Exception as exc:
                    logger.warning(f"SAM3 segmentation failed for box {box}: {exc}")
                    results.append({
                        "label": "box",
                        "bbox": fallback_bbox,
                        "points": fallback_points,
                        "score": 0.0,
                        "segmentation_area": 0.0,
                    })

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        logger.info(f"SAM3 segment_boxes_with_points complete: {len(results)} masks for {len(boxes)} boxes")
        return results
    finally:
        if temp_key_path and Path(temp_key_path).exists():
            Path(temp_key_path).unlink(missing_ok=True)
        if image_path and image_path.exists():
            image_path.unlink(missing_ok=True)
