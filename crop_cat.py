#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import shutil
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageOps
from ultralytics import YOLO

try:
    import cv2
except ImportError:
    cv2 = None


CAT_CLASS_ID = 15  # COCO class id for "cat"
MODELS_DIR = Path("Models")
PICTURES_DIR_NAME = "Pictures"
CROPPED_DIR_NAME = "Cropped"
OK_DIR_NAME = "Ok"
SKIP_DIR_NAME = "Skip"
RETRY_CHOICES = ("all", "skip", "ok")
HEAD_TARGET_RATIO_IN_SQUARE = 0.30
CAT_FACE_CASCADE_FILES = (
    "haarcascade_frontalcatface_extended.xml",
    "haarcascade_frontalcatface.xml",
)
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".jfif",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automatically crop cat photos from a folder."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(PICTURES_DIR_NAME),
        help=f"Input folder containing images (default: {PICTURES_DIR_NAME}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"Output folder (default: <input>/{CROPPED_DIR_NAME}).",
    )
    parser.add_argument(
        "--model",
        default=str(MODELS_DIR / "yolov8n.pt"),
        help="YOLO model file/name (default: Models/yolov8n.pt).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Detection confidence threshold (default: 0.25).",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.10,
        help="Padding ratio around the detected cat (default: 0.10).",
    )
    parser.add_argument(
        "--output-box",
        action="store_true",
        help=(
            "Force square output crop. The square is biased to keep the cat head "
            "near the upper part of the image."
        ),
    )
    parser.add_argument(
        "--retry",
        choices=RETRY_CHOICES,
        default=None,
        help=(
            "Restore images before processing: "
            f"'all' restores {OK_DIR_NAME}/{SKIP_DIR_NAME} and clears output, "
            f"'skip' restores only {SKIP_DIR_NAME}, "
            f"'ok' restores only {OK_DIR_NAME}."
        ),
    )
    return parser.parse_args()


def list_images(folder: Path) -> Iterable[Path]:
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def expand_box(x1: float, y1: float, x2: float, y2: float, pad: float, w: int, h: int):
    bw = x2 - x1
    bh = y2 - y1
    dx = bw * pad
    dy = bh * pad
    return (
        max(0, int(round(x1 - dx))),
        max(0, int(round(y1 - dy))),
        min(w, int(round(x2 + dx))),
        min(h, int(round(y2 + dy))),
    )


def expand_square_box_with_head_focus(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    pad: float,
    w: int,
    h: int,
    head_point: tuple[float, float] | None = None,
):
    ex1, ey1, ex2, ey2 = expand_box(x1, y1, x2, y2, pad, w, h)
    if head_point is not None:
        focus_x, focus_y = head_point
    else:
        focus_x = (ex1 + ex2) / 2.0
        focus_y = y1 + ((y2 - y1) * 0.30)

    side_limit = max(1, min(w, h))
    ex1, ey1, ex2, ey2 = shrink_box_to_square_capacity(
        ex1, ey1, ex2, ey2, side_limit, focus_x, focus_y
    )

    bw = max(1, ex2 - ex1)
    bh = max(1, ey2 - ey1)
    side = max(1, min(max(bw, bh), side_limit))

    desired_left = int(round(focus_x - (side / 2.0)))
    desired_top = int(round(focus_y - (side * HEAD_TARGET_RATIO_IN_SQUARE)))

    min_left = max(0, ex2 - side)
    max_left = min(w - side, ex1)
    min_top = max(0, ey2 - side)
    max_top = min(h - side, ey1)

    if min_left <= max_left:
        left = min(max(desired_left, min_left), max_left)
    else:
        left = min(max(desired_left, 0), w - side)

    if min_top <= max_top:
        top = min(max(desired_top, min_top), max_top)
    else:
        top = min(max(desired_top, 0), h - side)

    left = max(0, min(left, w - side))
    top = max(0, min(top, h - side))

    return (left, top, left + side, top + side)


def shrink_box_to_square_capacity(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    side_limit: int,
    focus_x: float,
    focus_y: float,
):
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    target_w = min(bw, side_limit)
    target_h = min(bh, side_limit)

    if target_w == bw and target_h == bh:
        return x1, y1, x2, y2

    desired_left = int(round(focus_x - (target_w / 2.0)))
    min_left = x1
    max_left = x2 - target_w

    keep_focus_left = int(math.ceil(focus_x - target_w))
    keep_focus_right = int(math.floor(focus_x))
    focus_left = max(min_left, keep_focus_left)
    focus_right = min(max_left, keep_focus_right)
    if focus_left <= focus_right:
        left = min(max(desired_left, focus_left), focus_right)
    else:
        left = min(max(desired_left, min_left), max_left)

    desired_top = int(round(focus_y - (target_h * HEAD_TARGET_RATIO_IN_SQUARE)))
    min_top = y1
    max_top = y2 - target_h

    keep_focus_top = int(math.ceil(focus_y - target_h))
    keep_focus_bottom = int(math.floor(focus_y))
    focus_top = max(min_top, keep_focus_top)
    focus_bottom = min(max_top, keep_focus_bottom)
    if focus_top <= focus_bottom:
        top = min(max(desired_top, focus_top), focus_bottom)
    else:
        top = min(max(desired_top, min_top), max_top)

    return left, top, left + target_w, top + target_h


_CAT_FACE_CASCADE = None
_CAT_FACE_CASCADE_LOADED = False


def get_cat_face_cascade():
    global _CAT_FACE_CASCADE, _CAT_FACE_CASCADE_LOADED
    if _CAT_FACE_CASCADE_LOADED:
        return _CAT_FACE_CASCADE

    _CAT_FACE_CASCADE_LOADED = True
    if cv2 is None:
        return None

    cascade_root = Path(cv2.data.haarcascades)
    for filename in CAT_FACE_CASCADE_FILES:
        cascade_path = cascade_root / filename
        if not cascade_path.exists():
            continue
        cascade = cv2.CascadeClassifier(str(cascade_path))
        if cascade.empty():
            continue
        _CAT_FACE_CASCADE = cascade
        return _CAT_FACE_CASCADE

    return None


def detect_cat_head_point(image: Image.Image, cat_box) -> tuple[float, float] | None:
    cascade = get_cat_face_cascade()
    if cascade is None:
        return None

    x1, y1, x2, y2 = [int(v) for v in cat_box]
    if x2 <= x1 or y2 <= y1:
        return None

    roi = image.crop((x1, y1, x2, y2)).convert("RGB")
    if roi.width < 24 or roi.height < 24:
        return None

    roi_bgr = cv2.cvtColor(np.array(roi), cv2.COLOR_RGB2BGR)
    roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    faces = cascade.detectMultiScale(
        roi_gray,
        scaleFactor=1.05,
        minNeighbors=4,
        minSize=(24, 24),
    )
    if len(faces) == 0:
        return None

    fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
    return (x1 + fx + (fw / 2.0), y1 + fy + (fh / 2.0))


def save_crop(image: Image.Image, box, out_path: Path) -> None:
    crop = image.crop(box)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {}
    if out_path.suffix.lower() in {".jpg", ".jpeg"}:
        kwargs["quality"] = 95
        kwargs["optimize"] = True
    crop.save(out_path, **kwargs)


def pillow_load_image(path: Path) -> Image.Image:
    with Image.open(path) as im_raw:
        return ImageOps.exif_transpose(im_raw).copy()


def predict_with_fallback(model: YOLO, img_path: Path, conf: float):
    try:
        return model(str(img_path), conf=conf, verbose=False)[0], None
    except FileNotFoundError:
        # Ultralytics does not accept some extensions by path (e.g. .jfif).
        # Fallback: load image with Pillow and pass the image directly.
        image = pillow_load_image(img_path)
        return model(image, conf=conf, verbose=False)[0], image


def clear_directory_contents(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for path in folder.iterdir():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def move_source_image(src_path: Path, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    candidate = target_dir / src_path.name

    if candidate.exists():
        stem = src_path.stem
        suffix = src_path.suffix
        index = 1
        while True:
            candidate = target_dir / f"{stem}_{index}{suffix}"
            if not candidate.exists():
                break
            index += 1

    return Path(shutil.move(str(src_path), str(candidate)))


def restore_images_from_subfolders(input_dir: Path, folder_names: tuple[str, ...]) -> int:
    restored = 0
    for folder_name in folder_names:
        source_dir = input_dir / folder_name
        if not source_dir.exists() or not source_dir.is_dir():
            continue

        for img_path in list(list_images(source_dir)):
            move_source_image(img_path, input_dir)
            restored += 1

    return restored


def handle_retry_all(input_dir: Path, output_dir: Path) -> bool:
    try:
        clear_directory_contents(output_dir)
    except OSError as exc:
        print(f"[error] Could not clear output folder {output_dir}: {exc}")
        return False

    restored = restore_images_from_subfolders(input_dir, (OK_DIR_NAME, SKIP_DIR_NAME))
    print(f"[info] Cleared output folder: {output_dir}")
    print(
        f"[info] Restored {restored} images from "
        f"{OK_DIR_NAME}/{SKIP_DIR_NAME} into {input_dir}"
    )
    return True


def handle_retry_skip(input_dir: Path) -> None:
    restored = restore_images_from_subfolders(input_dir, (SKIP_DIR_NAME,))
    print(f"[info] Restored {restored} images from {SKIP_DIR_NAME} into {input_dir}")


def handle_retry_ok(input_dir: Path) -> None:
    restored = restore_images_from_subfolders(input_dir, (OK_DIR_NAME,))
    print(f"[info] Restored {restored} images from {OK_DIR_NAME} into {input_dir}")


def resolve_model_arg(model_arg: str) -> str:
    # If only a .pt filename is provided, keep model files under Models/.
    candidate = Path(model_arg)
    if (
        not candidate.is_absolute()
        and candidate.parent == Path(".")
        and candidate.suffix.lower() == ".pt"
    ):
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        return str(MODELS_DIR / candidate.name)
    return model_arg


def main() -> int:
    args = parse_args()
    input_dir = args.input.resolve()
    output_dir = (args.output or (args.input / CROPPED_DIR_NAME)).resolve()
    model_arg = resolve_model_arg(args.model)

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"[error] Input folder not found: {input_dir}")
        return 1

    if output_dir == input_dir:
        print("[error] Output folder cannot be the same as input folder.")
        return 1

    print(f"[info] Loading model: {model_arg}")
    model = None
    try:
        model = YOLO(model_arg)
    except Exception as exc:
        print(f"[error] Could not load model '{model_arg}': {exc}")
        return 1

    assert model is not None

    if args.retry == "all":
        if not handle_retry_all(input_dir, output_dir):
            return 1
    elif args.retry == "skip":
        handle_retry_skip(input_dir)
    elif args.retry == "ok":
        handle_retry_ok(input_dir)

    images = list(list_images(input_dir))
    if not images:
        print(f"[error] No supported images found in: {input_dir}")
        return 1

    print(f"[info] Found {len(images)} images.")

    saved = 0
    skipped = 0
    moved_ok = 0
    moved_skip = 0
    ok_dir = input_dir / OK_DIR_NAME
    skip_dir = input_dir / SKIP_DIR_NAME
    warned_head_detection_unavailable = False

    for img_path in images:
        try:
            result, loaded_image = predict_with_fallback(model, img_path, args.conf)
        except Exception as exc:
            skipped += 1
            move_source_image(img_path, skip_dir)
            moved_skip += 1
            print(f"[skip] {img_path.name}: prediction failed ({exc})")
            continue

        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            skipped += 1
            move_source_image(img_path, skip_dir)
            moved_skip += 1
            print(f"[skip] {img_path.name}: no detections")
            continue

        cat_boxes = []
        for i in range(len(boxes)):
            cls = int(boxes.cls[i].item())
            if cls != CAT_CLASS_ID:
                continue
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            area = max(0.0, (x2 - x1) * (y2 - y1))
            cat_boxes.append((area, x1, y1, x2, y2))

        if not cat_boxes:
            skipped += 1
            move_source_image(img_path, skip_dir)
            moved_skip += 1
            print(f"[skip] {img_path.name}: no cat detected")
            continue

        im = loaded_image if loaded_image is not None else pillow_load_image(img_path)
        width, height = im.size

        _, x1, y1, x2, y2 = max(cat_boxes, key=lambda b: b[0])
        if args.output_box:
            expanded_box = expand_box(x1, y1, x2, y2, args.padding, width, height)
            head_point = detect_cat_head_point(im, expanded_box)
            if (
                head_point is None
                and cv2 is None
                and not warned_head_detection_unavailable
            ):
                print(
                    "[warn] Head detection unavailable (opencv-python not installed). "
                    "Using bbox-based focus."
                )
                warned_head_detection_unavailable = True
            box = expand_square_box_with_head_focus(
                x1,
                y1,
                x2,
                y2,
                args.padding,
                width,
                height,
                head_point=head_point,
            )
        else:
            box = expand_box(x1, y1, x2, y2, args.padding, width, height)
        out_path = output_dir / img_path.name
        save_crop(im, box, out_path)
        saved += 1
        print(f"[ok] {img_path.name}: saved")

        move_source_image(img_path, ok_dir)
        moved_ok += 1

    print(
        f"\n[done] Saved: {saved} | Skipped: {skipped} | "
        f"Moved {OK_DIR_NAME}: {moved_ok} | "
        f"Moved {SKIP_DIR_NAME}: {moved_skip} | Output: {output_dir}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
