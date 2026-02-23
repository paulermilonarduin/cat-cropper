#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
import shutil
import sys
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageOps
from ultralytics import YOLO


CAT_CLASS_ID = 15  # COCO class id for "cat"
MODELS_DIR = Path("Models")
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
        default=Path("Pictures"),
        help="Input folder containing images (default: Pictures).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output folder (default: <input>/CROPPED).",
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
        "--retry-all",
        action="store_true",
        help="Restore images from OK/SKIP, clear output, then reprocess all photos.",
    )
    parser.add_argument(
        "--retry-skip",
        action="store_true",
        help="Restore images from SKIP into input folder before processing.",
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


def save_crop(image: Image.Image, box, out_path: Path) -> None:
    crop = image.crop(box)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {}
    if out_path.suffix.lower() in {".jpg", ".jpeg"}:
        kwargs["quality"] = 95
        kwargs["optimize"] = True
    crop.save(out_path, **kwargs)


def load_image(path: Path) -> Image.Image:
    with Image.open(path) as im_raw:
        return ImageOps.exif_transpose(im_raw).copy()


def predict_with_fallback(model: YOLO, img_path: Path, conf: float):
    try:
        return model(str(img_path), conf=conf, verbose=False)[0], None
    except FileNotFoundError:
        # Ultralytics does not accept some extensions by path (e.g. .jfif).
        # Fallback: load image with Pillow and pass the image directly.
        image = load_image(img_path)
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
    output_dir = (args.output or (args.input / "CROPPED")).resolve()
    model_arg = resolve_model_arg(args.model)

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"[error] Input folder not found: {input_dir}")
        return 1

    if output_dir == input_dir:
        print("[error] Output folder cannot be the same as input folder.")
        return 1

    print(f"[info] Loading model: {model_arg}")
    model = None
    for attempt in range(1, 4):
        try:
            model = YOLO(model_arg)
            break
        except Exception as exc:
            if attempt < 3:
                print(f"[warn] Model load failed (attempt {attempt}/3): {exc}")
                time.sleep(1)
            else:
                print(f"[error] Could not load model '{model_arg}': {exc}")
                print("[hint] Retry, or pass a local model path with --model.")
                print("[hint] Example: python crop_cat.py --model Models/yolo11n.pt")
                return 1

    assert model is not None

    if args.retry_all:
        try:
            clear_directory_contents(output_dir)
        except OSError as exc:
            print(f"[error] Could not clear output folder {output_dir}: {exc}")
            return 1
        restored = restore_images_from_subfolders(input_dir, ("OK", "SKIP"))
        print(f"[info] Cleared output folder: {output_dir}")
        print(f"[info] Restored {restored} images from OK/SKIP into {input_dir}")
    elif args.retry_skip:
        restored = restore_images_from_subfolders(input_dir, ("SKIP",))
        print(f"[info] Restored {restored} images from SKIP into {input_dir}")

    images = list(list_images(input_dir))
    if not images:
        print(f"[error] No supported images found in: {input_dir}")
        return 1

    print(f"[info] Found {len(images)} images.")

    saved = 0
    skipped = 0
    moved_ok = 0
    moved_skip = 0
    ok_dir = input_dir / "OK"
    skip_dir = input_dir / "SKIP"

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

        im = loaded_image if loaded_image is not None else load_image(img_path)
        width, height = im.size

        _, x1, y1, x2, y2 = max(cat_boxes, key=lambda b: b[0])
        box = expand_box(x1, y1, x2, y2, args.padding, width, height)
        out_path = output_dir / img_path.name
        save_crop(im, box, out_path)
        saved += 1
        print(f"[ok] {img_path.name}: saved")

        move_source_image(img_path, ok_dir)
        moved_ok += 1

    print(
        f"\n[done] Saved: {saved} | Skipped: {skipped} | "
        f"Moved OK: {moved_ok} | Moved skip: {moved_skip} | Output: {output_dir}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
