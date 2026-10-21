"""Read a sentence from an image with pre-trained OCR models (no custom model).

Engines:
  easyocr : detects text regions + recognizes them. Best for printed text,
            photos, multi-line images, and fairly neat handwriting.
  trocr   : Microsoft TrOCR (handwritten). Best for handwriting, but it reads
            ONE line of text at a time (this script splits lines for you).

Install:
    pip install easyocr                          # for --engine easyocr
    pip install transformers torch pillow numpy  # for --engine trocr

Usage:
    python ocr_sentence.py mysentence.png
    python ocr_sentence.py mysentence.png --engine trocr
"""
import argparse
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
from PIL import Image, ImageOps


# ----------------------------------------------------------------- EasyOCR
def read_with_easyocr(path: str, min_conf: float = 0.2) -> str:
    import easyocr

    reader = easyocr.Reader(["en"], gpu=False)  # set gpu=True if you have CUDA
    results = reader.readtext(path)             # [(box, text, confidence), ...]

    # Each item: box = 4 corner points. Sort into reading order (top->bottom, left->right)
    items = []
    for box, text, conf in results:
        if conf < min_conf:
            continue
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        items.append({
            "text": text,
            "x": min(xs),
            "yc": (min(ys) + max(ys)) / 2,
            "h": max(ys) - min(ys),
        })
    if not items:
        return ""

    items.sort(key=lambda i: i["yc"])
    lines, current = [], [items[0]]
    for it in items[1:]:
        # same line if vertical centres are close (< half a text height apart)
        if abs(it["yc"] - np.mean([c["yc"] for c in current])) < 0.5 * max(it["h"], 1):
            current.append(it)
        else:
            lines.append(current)
            current = [it]
    lines.append(current)

    return "\n".join(
        " ".join(w["text"] for w in sorted(line, key=lambda w: w["x"]))
        for line in lines
    )


# ------------------------------------------------------------------- TrOCR
def split_lines(img: Image.Image, pad: int = 8) -> list[Image.Image]:
    """Split an image into text-line crops using a horizontal projection."""
    gray = np.asarray(ImageOps.exif_transpose(img).convert("L"), dtype=np.uint8)
    ink = gray < 128 if gray.mean() > 127 else gray > 128
    rows = ink.any(axis=1)

    crops, start = [], None
    for i, v in enumerate(list(rows) + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= 8:  # ignore specks
                y0, y1 = max(0, start - pad), min(len(rows), i + pad)
                crops.append(img.crop((0, y0, img.width, y1)))
            start = None
    return crops or [img]


def read_with_trocr(path: str, model_name: str = "microsoft/trocr-base-handwritten") -> str:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)

    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    texts = []
    for line_img in split_lines(img):
        pixel_values = processor(images=line_img, return_tensors="pt").pixel_values
        ids = model.generate(pixel_values, max_new_tokens=64)
        texts.append(processor.batch_decode(ids, skip_special_tokens=True)[0].strip())
    return "\n".join(texts)


# -------------------------------------------------------------------- main
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="OCR a sentence from an image")
    ap.add_argument("image")
    ap.add_argument("--engine", choices=["easyocr", "trocr"], default="easyocr")
    args = ap.parse_args()

    if args.engine == "easyocr":
        text = read_with_easyocr(args.image)
    else:
        text = read_with_trocr(args.image)

    print("Predicted text:")
    print(text)