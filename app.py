"""
Server web lokal untuk klasifikasi tingkat kepadatan area parkir.

Back-end memakai pustaka standar Python (http.server) + TensorFlow hanya untuk
memuat model dan melakukan prediksi. Front-end berupa HTML/CSS/JS biasa.
Jalankan:  python app.py   lalu buka  http://127.0.0.1:8000
"""
import json
import mimetypes
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import tensorflow as tf

# ----------------------------- Konfigurasi -----------------------------
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = (BASE_DIR / "static").resolve()
MODEL_PATH = BASE_DIR / "model" / "cnn_scratch_kepadatan_parkir_v2.keras"
THRESHOLD_PATH = BASE_DIR / "model" / "ambang_kepadatan_parkir_v2.txt"

HOST = "127.0.0.1"          # hanya bisa diakses dari komputer ini
PORT = 8000
IMG_SIZE = (224, 224)        # harus sama dengan saat training
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/bmp"}
LOW_CONFIDENCE = 0.50        # di bawah ini, hasil ditandai "kurang yakin"


# ------------------------ Membaca file ambang (.txt) ------------------------
def load_threshold_info(path: Path):
    cut_points, class_names = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        match = re.match(r"P\d+=([\d.]+)$", line)
        if match:
            cut_points.append(float(match.group(1)))
        elif line.startswith("Urutan kelas:"):
            class_names = [n.strip() for n in line.split(":", 1)[1].split(",")]
    if len(cut_points) != 4 or len(class_names) != 5:
        raise ValueError(
            f"Format {path.name} tidak sesuai: ditemukan {len(cut_points)} ambang "
            f"dan {len(class_names)} kelas (seharusnya 4 dan 5)."
        )
    return cut_points, class_names


def pct(value: float) -> str:
    return f"{value * 100:.2f}".replace(".", ",") + "%"


def build_ranges(cut_points):
    """Rentang rasio petak terisi per kelas (sesuai np.digitize(..., right=True))."""
    ranges = [f"≤ {pct(cut_points[0])}"]
    for i in range(1, 4):
        ranges.append(f"> {pct(cut_points[i - 1])} sampai {pct(cut_points[i])}")
    ranges.append(f"> {pct(cut_points[3])}")
    return ranges


CUT_POINTS, CLASS_NAMES = load_threshold_info(THRESHOLD_PATH)
CLASS_RANGES = build_ranges(CUT_POINTS)

# ------------------------------ Memuat model ------------------------------
print("Memuat model:", MODEL_PATH.name)
MODEL = tf.keras.models.load_model(MODEL_PATH, compile=False)
MODEL_LOCK = threading.Lock()  # model dipakai bergantian antar-permintaan
print("Model siap. Input:", MODEL.input_shape, "| Output:", MODEL.output_shape)


def preprocess(image_bytes: bytes):
    """Sama persis dengan pipeline training: decode RGB -> resize 224x224 -> uint8.
    Pembagian 255 TIDAK dilakukan di sini karena sudah ada layer Rescaling di model."""
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image = tf.image.resize(image, IMG_SIZE, antialias=True)
    image = tf.cast(tf.round(image), tf.uint8)
    return tf.expand_dims(image, axis=0)


def predict(image_bytes: bytes):
    batch = preprocess(image_bytes)
    with MODEL_LOCK:
        probs = MODEL(batch, training=False).numpy()[0]
    best = int(np.argmax(probs))
    return {
        "label": CLASS_NAMES[best],
        "range": CLASS_RANGES[best],
        "confidence": float(probs[best]),
        "low_confidence": bool(probs[best] < LOW_CONFIDENCE),
        "probabilities": [
            {"label": name, "range": rng, "probability": float(p)}
            for name, rng, p in zip(CLASS_NAMES, CLASS_RANGES, probs)
        ],
    }


# ------------------------------ HTTP handler ------------------------------
class Handler(BaseHTTPRequestHandler):
    server_version = "ParkirLokal/1.0"

    # --- utilitas respons ---
    def _send(self, status, body: bytes, content_type: str):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def _error(self, status, message):
        self._json(status, {"error": message})

    def _serve_static(self, relative: str):
        target = (STATIC_DIR / relative).resolve()
        if STATIC_DIR not in target.parents or not target.is_file():
            return self._error(404, "Halaman tidak ditemukan.")
        ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        if target.suffix == ".js":
            ctype = "text/javascript"
        if ctype.startswith("text/") or ctype == "text/javascript":
            ctype += "; charset=utf-8"
        self._send(200, target.read_bytes(), ctype)

    # --- rute ---
    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            return self._serve_static("index.html")
        if path == "/api/info":
            return self._json(200, {
                "classes": [{"label": n, "range": r} for n, r in zip(CLASS_NAMES, CLASS_RANGES)],
                "max_upload_mb": MAX_UPLOAD_BYTES // (1024 * 1024),
            })
        if path.startswith("/static/"):
            return self._serve_static(path[len("/static/"):])
        return self._error(404, "Halaman tidak ditemukan.")

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path != "/api/predict":
            return self._error(404, "Endpoint tidak ditemukan.")

        content_type = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if content_type not in ALLOWED_TYPES:
            return self._error(415, "Format gambar harus JPG, PNG, atau BMP.")

        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            return self._error(411, "Ukuran data tidak diketahui.")
        if length <= 0:
            return self._error(400, "Gambar kosong.")
        if length > MAX_UPLOAD_BYTES:
            return self._error(413, f"Ukuran gambar maksimal {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")

        image_bytes = self.rfile.read(length)
        try:
            result = predict(image_bytes)
        except tf.errors.InvalidArgumentError:
            return self._error(400, "File tidak dapat dibaca sebagai gambar yang valid.")
        except Exception as exc:  # kesalahan tak terduga di sisi server
            print("Kesalahan prediksi:", repr(exc))
            return self._error(500, "Terjadi kesalahan saat memproses gambar.")
        return self._json(200, result)


if __name__ == "__main__":
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Server berjalan di http://{HOST}:{PORT}  (hentikan dengan Ctrl+C)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer dihentikan.")
    finally:
        server.server_close()
