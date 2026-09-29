from __future__ import annotations

import os
import shutil
import subprocess
import uuid
from pathlib import Path
from flask import Flask, render_template, request, send_from_directory, jsonify
from werkzeug.utils import secure_filename

BASE = Path(__file__).resolve().parent
UPLOADS = BASE / "uploads"
OUTPUTS = BASE / "outputs"
UPLOADS.mkdir(exist_ok=True)
OUTPUTS.mkdir(exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 150 * 1024 * 1024

ALLOWED = {"png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"}


def allowed(name: str) -> bool:
    return "." in name and name.rsplit(".", 1)[1].lower() in ALLOWED


def save_upload(file, folder: Path) -> Path:
    if not file or not file.filename or not allowed(file.filename):
        raise ValueError("صيغة الصورة غير مدعومة.")
    folder.mkdir(parents=True, exist_ok=True)
    ext = file.filename.rsplit(".", 1)[1].lower()
    path = folder / f"{uuid.uuid4().hex}.{ext}"
    file.save(path)
    return path


def facefusion_python() -> Path | None:
    configured = os.environ.get("FACEFUSION_DIR", "").strip()
    if not configured:
        return None
    root = Path(configured)
    candidates = [root / "facefusion.py", root / "facefusion" / "facefusion.py"]
    return next((p for p in candidates if p.exists()), None)


def run_faceswap(source: Path, target: Path, out: Path, model: str, order: str) -> None:
    entry = facefusion_python()
    if not entry:
        raise RuntimeError("محرك FaceFusion غير مضبوط. اضبط FACEFUSION_DIR داخل الـCodespace أولًا.")
    python = shutil.which("python") or shutil.which("python3")
    if not python:
        raise RuntimeError("Python غير متاح داخل بيئة التشغيل.")
    cmd = [
        python, str(entry), "headless-run",
        "-s", str(source), "-t", str(target), "-o", str(out),
        "--processors", "face_swapper",
        "--face-swapper-model", model,
        "--face-selector-mode", "reference",
        "--face-selector-order", order,
        "--output-image-quality", "95",
    ]
    proc = subprocess.run(cmd, cwd=str(entry.parent), capture_output=True, text=True, timeout=900)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()[-3000:]
        raise RuntimeError(detail or "فشل محرك تبديل الوجوه.")


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/faceswap")
def faceswap():
    try:
        reference = request.files.get("reference")
        targets = request.files.getlist("targets")
        if not reference or not targets:
            return jsonify(error="اختر صورة مرجعية وصورة مستهدفة واحدة على الأقل."), 400
        if len(targets) > 8:
            return jsonify(error="الحد الأقصى 8 صور مستهدفة في العملية الواحدة."), 400

        job = UPLOADS / uuid.uuid4().hex
        out_dir = OUTPUTS / uuid.uuid4().hex
        job.mkdir(parents=True)
        out_dir.mkdir(parents=True)
        source = save_upload(reference, job)
        model = request.form.get("model", "inswapper_128_fp16")
        order = request.form.get("order", "left-right")
        results = []
        for target in targets:
            target_path = save_upload(target, job)
            out = out_dir / f"{target_path.stem}_faceswap.png"
            run_faceswap(source, target_path, out, model, order)
            results.append({"name": out.name, "url": f"/files/{out_dir.name}/{out.name}"})
        return jsonify(results=results)
    except Exception as exc:
        return jsonify(error=str(exc)), 500


@app.post("/api/describe")
def describe():
    # Intentionally lightweight: the UI is ready for a local vision model integration.
    files = request.files.getlist("images")
    if not files:
        return jsonify(error="اختر صورة واحدة على الأقل."), 400
    return jsonify(message="واجهة الوصف جاهزة، لكن لم يتم تفعيل نموذج رؤية محلي بعد. يمكن ربط Ollama/llava أو نموذج رؤية آخر داخل Codespaces لاحقًا.")


@app.post("/api/enhance")
def enhance():
    return jsonify(error="محرك Real-ESRGAN غير موصول بعد. اضبط مسار المحرك المحلي أولًا."), 501


@app.get("/files/<folder>/<name>")
def files(folder: str, name: str):
    safe_folder = Path(folder).name
    safe_name = secure_filename(name)
    return send_from_directory(OUTPUTS / safe_folder, safe_name, as_attachment=False)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "7860")), debug=False)
