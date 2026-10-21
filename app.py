"""Flask app: upload an image of a handwritten letter and get a prediction.

Run:  python app.py   ->  http://127.0.0.1:5000
"""
import base64
import io
import string

import numpy as np
from flask import Flask, render_template, request
from PIL import Image, ImageOps

from src.utils import load_config, resolve

config = load_config("config.yaml")
APP_CFG = config["app"]
IMG_SIZE = config["data_ingestion"]["image_size"]
LETTERS = list(string.ascii_uppercase)  # labels 0-25 -> A-Z

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = APP_CFG["max_upload_mb"] * 1024 * 1024

_model = None


def get_model():
    """Load the trained model once, on first use."""
    global _model
    if _model is None:
        from tensorflow.keras.models import load_model

        path = resolve(config["training"]["model_path"])
        if not path.exists():
            raise FileNotFoundError(
                f"Model not found at {path}. Run: python run_pipeline.py --stage train"
            )
        _model = load_model(path)
    return _model


def allowed_file(filename: str) -> bool:
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in APP_CFG["allowed_extensions"]
    )


def preprocess(file_stream) -> tuple[np.ndarray, Image.Image]:
    """Convert an uploaded image to a (1, 28, 28, 1) float array.

    Steps: fix EXIF rotation -> grayscale -> (invert) -> (crop+pad to square)
    -> resize to 28x28 -> (transpose) -> scale to 0-1.
    Returns the model input and the processed PIL image (for display).
    """
    img = Image.open(file_stream)
    img = ImageOps.exif_transpose(img)

    # Flatten transparency onto white, then convert to grayscale
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    img = img.convert("L")

    arr = np.asarray(img, dtype=np.uint8)

    # EMNIST letters are light on dark; most uploads are dark ink on paper
    if APP_CFG["auto_invert"] and arr.mean() > 127:
        arr = 255 - arr

    # Crop to the letter and centre it in a square with a margin
    if APP_CFG["crop_to_letter"]:
        mask = arr > max(30, int(arr.max() * 0.25))
        if mask.any():
            rows, cols = np.where(mask)
            arr = arr[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
        h, w = arr.shape
        side = max(h, w)
        margin = max(1, int(side * 0.2))
        canvas = np.zeros((side + 2 * margin, side + 2 * margin), dtype=np.uint8)
        y0 = (canvas.shape[0] - h) // 2
        x0 = (canvas.shape[1] - w) // 2
        canvas[y0:y0 + h, x0:x0 + w] = arr
        arr = canvas

    img = Image.fromarray(arr).resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)

    if APP_CFG["transpose_input"]:
        img = img.transpose(Image.TRANSPOSE)

    x = np.asarray(img, dtype="float32") / 255.0
    return x.reshape(1, IMG_SIZE, IMG_SIZE, 1), img


def to_data_uri(img: Image.Image, scale: int = 1) -> str:
    if scale > 1:
        img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html")

    file = request.files.get("image")
    if file is None or file.filename == "":
        return render_template("index.html", error="Please choose an image file.")
    if not allowed_file(file.filename):
        allowed = ", ".join(APP_CFG["allowed_extensions"])
        return render_template("index.html", error=f"Unsupported file type. Use: {allowed}.")

    try:
        x, processed = preprocess(file.stream)
        probs = get_model().predict(x, verbose=0)[0]
    except FileNotFoundError as e:
        return render_template("index.html", error=str(e))
    except Exception as e:  # unreadable/corrupt image etc.
        return render_template("index.html", error=f"Could not process image: {e}")

    top = np.argsort(probs)[::-1][: APP_CFG["top_k"]]
    predictions = [
        {"letter": LETTERS[i], "confidence": float(probs[i]) * 100} for i in top
    ]

    file.stream.seek(0)
    original = Image.open(file.stream)
    original = ImageOps.exif_transpose(original)
    original.thumbnail((280, 280))

    return render_template(
        "index.html",
        predictions=predictions,
        best=predictions[0],
        original_uri=to_data_uri(original.convert("RGB")),
        processed_uri=to_data_uri(processed, scale=8),
    )


@app.errorhandler(413)
def too_large(_):
    return render_template(
        "index.html", error=f"File too large (max {APP_CFG['max_upload_mb']} MB)."
    ), 413


if __name__ == "__main__":
    app.run(host=APP_CFG["host"], port=APP_CFG["port"], debug=APP_CFG["debug"])
