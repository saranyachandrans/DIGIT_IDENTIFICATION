"""Predict a sentence from an image using the trained EMNIST letter CNN.

How it works (classical segmentation + your CNN):
  1. Convert to grayscale, make letters light on dark, binarize (Otsu).
  2. Split the image into text lines   (horizontal projection).
  3. Split each line into letters      (vertical projection).
  4. Wide gaps between letters become spaces.
  5. Each letter crop -> 28x28 -> CNN -> A-Z.

Usage:
    python predict_sentence.py path/to/image.png
    python predict_sentence.py path/to/image.png --debug   # saves boxes image

Limitations: letters must be separated (not touching/cursive), roughly the
same size, on a plain background. The model only knows A-Z (case-merged).
"""
import argparse
import string
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from src.utils import load_config, resolve

config = load_config("config.yaml")
IMG_SIZE = config["data_ingestion"]["image_size"]
TRANSPOSE = config["app"]["transpose_input"]
LETTERS = list(string.ascii_uppercase)

SPACE_RATIO = 2.0     # gap wider than this x the line's median gap => space
FEW_GAPS_FACTOR = 0.6 # lines with < 3 gaps: gap wider than this x line height => space
MIN_PIXELS = 3        # ignore specks smaller than this (px wide or tall)


# ---------------------------------------------------------------- preprocessing
def otsu_threshold(gray: np.ndarray) -> int:
    hist = np.bincount(gray.ravel(), minlength=256).astype(float)
    total = gray.size
    sum_all = np.dot(np.arange(256), hist)
    w_b = sum_b = 0.0
    best, thresh = 0.0, 127
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b, m_f = sum_b / w_b, (sum_all - sum_b) / w_f
        between = w_b * w_f * (m_b - m_f) ** 2
        if between > best:
            best, thresh = between, t
    return thresh


def load_binary(path) -> tuple[np.ndarray, np.ndarray]:
    """Return (gray_light_on_dark uint8, binary bool mask of ink)."""
    img = ImageOps.exif_transpose(Image.open(path))
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    gray = np.asarray(img.convert("L"), dtype=np.uint8)
    if gray.mean() > 127:           # dark ink on light paper -> flip
        gray = 255 - gray
    mask = gray > otsu_threshold(gray)
    return gray, mask


# ---------------------------------------------------------------- segmentation
def runs(profile: np.ndarray) -> list[tuple[int, int]]:
    """Start/end (end exclusive) of consecutive True stretches."""
    out, start = [], None
    for i, v in enumerate(profile):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(profile)))
    return out


def segment(mask: np.ndarray):
    """Yield lines; each line is a list of (x0, y0, x1, y1, space_before)."""
    lines = []
    for y0, y1 in runs(mask.any(axis=1)):
        if y1 - y0 < MIN_PIXELS:
            continue
        line_mask = mask[y0:y1]
        line_h = y1 - y0
        boxes, spans = [], []
        for x0, x1 in runs(line_mask.any(axis=0)):
            if x1 - x0 < MIN_PIXELS and line_h > 4 * MIN_PIXELS:
                continue  # tiny speck
            rows = np.where(line_mask[:, x0:x1].any(axis=1))[0]
            by0, by1 = y0 + rows.min(), y0 + rows.max() + 1
            spans.append((x0, x1))
            boxes.append([x0, by0, x1, by1, False])

        # Decide which gaps are word spaces, relative to this line's own spacing
        gaps = [spans[i + 1][0] - spans[i][1] for i in range(len(spans) - 1)]
        if gaps:
            if len(gaps) >= 3:
                limit = SPACE_RATIO * float(np.median(gaps))
            else:
                limit = FEW_GAPS_FACTOR * line_h
            for i, g in enumerate(gaps):
                boxes[i + 1][4] = g > limit
        boxes = [tuple(b) for b in boxes]
        if boxes:
            lines.append(boxes)
    return lines


def crop_to_input(gray: np.ndarray, box) -> np.ndarray:
    """Crop one letter, centre in a padded square, resize to 28x28."""
    x0, y0, x1, y1, _ = box
    crop = gray[y0:y1, x0:x1]
    h, w = crop.shape
    side = max(h, w)
    margin = max(1, int(side * 0.2))
    canvas = np.zeros((side + 2 * margin,) * 2, dtype=np.uint8)
    oy, ox = (canvas.shape[0] - h) // 2, (canvas.shape[1] - w) // 2
    canvas[oy:oy + h, ox:ox + w] = crop
    img = Image.fromarray(canvas).resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    if TRANSPOSE:
        img = img.transpose(Image.TRANSPOSE)
    return np.asarray(img, dtype="float32") / 255.0


# ---------------------------------------------------------------- prediction
_model = None


def get_model():
    global _model
    if _model is None:
        from tensorflow.keras.models import load_model
        path = resolve(config["training"]["model_path"])
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}. Train it first.")
        _model = load_model(path)
    return _model


def predict_sentence(image_path, debug_path=None) -> str:
    gray, mask = load_binary(image_path)
    lines = segment(mask)
    if not lines:
        return ""

    # Batch all letters through the CNN in one call
    boxes = [b for line in lines for b in line]
    X = np.stack([crop_to_input(gray, b) for b in boxes])[..., None]
    probs = get_model().predict(X, verbose=0)
    preds = [LETTERS[i] for i in probs.argmax(axis=1)]

    # Rebuild text with spaces and line breaks
    text, k = [], 0
    for line in lines:
        chars = []
        for box in line:
            if box[4]:
                chars.append(" ")
            chars.append(preds[k])
            k += 1
        text.append("".join(chars))

    if debug_path:
        vis = Image.fromarray(gray).convert("RGB")
        d = ImageDraw.Draw(vis)
        for (x0, y0, x1, y1, _), p in zip(boxes, preds):
            d.rectangle([x0, y0, x1, y1], outline="red")
            d.text((x0, max(0, y0 - 12)), p, fill="yellow")
        vis.save(debug_path)

    return "\n".join(text)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Predict a sentence from an image")
    ap.add_argument("image")
    ap.add_argument("--debug", action="store_true", help="save <image>_boxes.png")
    args = ap.parse_args()

    dbg = None
    if args.debug:
        p = Path(args.image)
        dbg = p.with_name(p.stem + "_boxes.png")

    print("Predicted text:")
    print(predict_sentence(args.image, dbg))
    if dbg:
        print(f"Segmentation preview saved to {dbg}")