from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import uuid
from pathlib import Path
from urllib import request as urlrequest

from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

BASE = Path(__file__).resolve().parent
UPLOADS = BASE / "uploads"
OUTPUTS = BASE / "outputs"
UPLOADS.mkdir(exist_ok=True)
OUTPUTS.mkdir(exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 300 * 1024 * 1024

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


def find_facefusion() -> tuple[Path, Path] | None:
    configured = os.environ.get("FACEFUSION_DIR", "").strip()
    candidates: list[Path] = []
    if configured:
        root = Path(configured)
        candidates += [root / "facefusion.py", root / "facefusion" / "facefusion.py"]
    candidates += [
        BASE / "engines" / "facefusion" / "facefusion.py",
        Path.home() / "facefusion" / "facefusion.py",
    ]
    for entry in candidates:
        if entry.exists():
            return entry, entry.parent
    return None


def run_command(cmd: list[str], cwd: Path, timeout: int = 3600) -> None:
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(detail[-6000:] or "المحرك لم يُرجع نتيجة صالحة.")


def facefusion_faceswap(
    source: Path,
    target: Path,
    output: Path,
    *,
    model: str,
    order: str,
    reference_position: int,
    quality: str,
    provider: str,
) -> None:
    resolved = find_facefusion()
    if not resolved:
        raise RuntimeError("FaceFusion غير مثبت. افتح إعداد المحركات وشغّل إعداد FaceFusion أولًا.")

    entry, cwd = resolved
    conda = shutil.which("conda")
    python = shutil.which("python") or shutil.which("python3")
    if conda:
        runner = [conda, "run", "--no-capture-output", "-n", "facefusion", "python", str(entry)]
    elif python:
        runner = [python, str(entry)]
    else:
        raise RuntimeError("Python/Conda غير متاحان لتشغيل FaceFusion.")

    processors = ["face_swapper"]
    if quality in {"high", "ultra"}:
        processors.append("face_enhancer")
    if quality == "ultra":
        processors.append("frame_enhancer")

    cmd = [*runner, "headless-run", "-s", str(source), "-t", str(target), "-o", str(output)]
    cmd += ["--processors", *processors]
    cmd += [
        "--face-swapper-model", model,
        "--face-swapper-pixel-boost", "512x512" if quality in {"high", "ultra"} else "256x256",
        "--face-selector-mode", "reference",
        "--face-selector-order", order,
        "--reference-face-position", str(max(0, reference_position)),
        "--reference-face-distance", "0.45",
        "--face-detector-size", "1024x1024" if quality == "ultra" else "640x640",
        "--output-image-quality", "100",
        "--execution-thread-count", "8",
    ]

    if quality in {"high", "ultra"}:
        cmd += [
            "--face-enhancer-model", "gpen_bfr_2048",
            "--face-enhancer-blend", "86",
            "--face-enhancer-weight", "0.8",
        ]

    if quality == "ultra":
        cmd += [
            "--frame-enhancer-model", "real_esrgan_x4_fp16",
            "--frame-enhancer-blend", "82",
            "--output-image-scale", "4",
        ]

    if provider in {"cpu", "cuda", "directml", "rocm", "tensorrt", "openvino"}:
        cmd += ["--execution-providers", provider]

    run_command(cmd, cwd)


def run_realesrgan(image: Path, output: Path, scale: float, model: str) -> None:
    script = os.environ.get("REALESRGAN_SCRIPT", "").strip()
    if not script:
        raise RuntimeError("Real-ESRGAN غير مضبوط. اضبط REALESRGAN_SCRIPT أو استخدم FaceFusion Ultra.")

    python = shutil.which("python") or shutil.which("python3")
    if not python:
        raise RuntimeError("Python غير متاح.")
    cmd = [
        python, script,
        "-n", model,
        "-i", str(image),
        "-o", str(output.parent),
        "-s", str(scale),
        "--face_enhance",
    ]
    run_command(cmd, Path(script).parent)


def ollama_describe(image: Path, instruction: str) -> str:
    base_url = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
    model = os.environ.get("OLLAMA_VISION_MODEL", "llava:latest")
    payload = {
        "model": model,
        "prompt": instruction or "اوصف الصورة بالعربية بدقة: المشهد، الأشخاص، الوضعيات، الملابس، الإضاءة، المنظور، الخلفية والتفاصيل المرئية فقط.",
        "images": [base64.b64encode(image.read_bytes()).decode("ascii")],
        "stream": False,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urlrequest.Request(base_url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlrequest.urlopen(req, timeout=180) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        raise RuntimeError("نموذج الرؤية المحلي غير متاح: " + str(exc)) from exc
    return str(result.get("response") or "").strip()


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health():
    ff = find_facefusion()
    return jsonify(
        facefusion=bool(ff),
        realesrgan=bool(os.environ.get("REALESRGAN_SCRIPT")),
        ollama=bool(os.environ.get("OLLAMA_URL")),
        comfyui=bool(os.environ.get("COMFYUI_URL")),
    )


@app.post("/api/faceswap")
def faceswap():
    try:
        reference = request.files.get("reference")
        targets = [f for f in request.files.getlist("targets") if f and f.filename]
        if not reference or not targets:
            return jsonify(error="اختر صورة مرجعية وصورة مستهدفة واحدة على الأقل."), 400
        if len(targets) > 8:
            return jsonify(error="الحد الأقصى 8 صور مستهدفة في العملية الواحدة."), 400

        quality = request.form.get("quality", "ultra")
        if quality not in {"standard", "high", "ultra"}:
            quality = "ultra"

        model = request.form.get("model", "hyperswap_1a_256")
        order = request.form.get("order", "left-right")
        provider = request.form.get("provider", "auto")
        reference_position = int(request.form.get("reference_position", "0"))

        job = UPLOADS / uuid.uuid4().hex
        out_dir = OUTPUTS / uuid.uuid4().hex
        job.mkdir(parents=True)
        out_dir.mkdir(parents=True)
        source = save_upload(reference, job)

        results = []
        for target in targets:
            target_path = save_upload(target, job)
            output = out_dir / f"{target_path.stem}_studio.png"
            facefusion_faceswap(
                source, target_path, output,
                model=model, order=order,
                reference_position=reference_position,
                quality=quality, provider=provider,
            )
            results.append({"name": output.name, "url": f"/files/{out_dir.name}/{output.name}"})

        return jsonify(results=results, quality=quality)
    except Exception as exc:
        return jsonify(error=str(exc)), 500


@app.post("/api/enhance")
def enhance():
    try:
        files = [f for f in request.files.getlist("images") if f and f.filename]
        if not files:
            return jsonify(error="اختر صورة واحدة على الأقل."), 400
        if len(files) > 8:
            return jsonify(error="الحد الأقصى 8 صور في العملية الواحدة."), 400

        quality = request.form.get("quality", "ultra")
        scale = {"standard": 2.0, "high": 4.0, "ultra": 4.0}.get(quality, 4.0)
        model = request.form.get("model", "RealESRGAN_x4plus")

        job = UPLOADS / uuid.uuid4().hex
        out_dir = OUTPUTS / uuid.uuid4().hex
        job.mkdir(parents=True)
        out_dir.mkdir(parents=True)
        results = []

        for file in files:
            src = save_upload(file, job)
            output = out_dir / f"{src.stem}_enhanced.png"
            if os.environ.get("REALESRGAN_SCRIPT"):
                run_realesrgan(src, output, scale, model)
            else:
                resolved = find_facefusion()
                if not resolved:
                    raise RuntimeError("لم يتم تثبيت Real-ESRGAN أو FaceFusion.")
                entry, cwd = resolved
                python = shutil.which("python") or shutil.which("python3")
                cmd = [
                    python, str(entry), "headless-run",
                    "-t", str(src), "-o", str(output),
                    "--processors", "frame_enhancer",
                    "--frame-enhancer-model", "real_esrgan_x4_fp16" if quality != "standard" else "real_esrgan_x2_fp16",
                    "--frame-enhancer-blend", "88",
                    "--output-image-quality", "100",
                    "--output-image-scale", str(scale),
                ]
                run_command(cmd, cwd)
            results.append({"name": output.name, "url": f"/files/{out_dir.name}/{output.name}"})

        return jsonify(results=results, quality=quality, scale=scale)
    except Exception as exc:
        return jsonify(error=str(exc)), 500


@app.post("/api/describe")
def describe():
    try:
        files = [f for f in request.files.getlist("images") if f and f.filename]
        if not files:
            return jsonify(error="اختر صورة واحدة على الأقل."), 400
        if len(files) > 8:
            return jsonify(error="الحد الأقصى 8 صور."), 400

        job = UPLOADS / uuid.uuid4().hex
        job.mkdir(parents=True)
        instruction = request.form.get("instruction", "")
        descriptions = []
        for file in files:
            src = save_upload(file, job)
            descriptions.append({"name": file.filename, "description": ollama_describe(src, instruction)})
        return jsonify(results=descriptions)
    except Exception as exc:
        return jsonify(error=str(exc)), 500


def classify_command(text: str) -> str:
    t = (text or "").strip().lower()
    if any(k in t for k in ["تبديل الوجه", "تبديل الوجوه", "face swap", "faceswap", "بدل الوجه"]):
        return "faceswap"
    if any(k in t for k in ["رفع الجودة", "جودة فائقة", "تحسين الجودة", "تكبير", "super resolution", "enhance", "تحسين الصورة"]):
        return "enhance"
    if any(k in t for k in ["وصف الصورة", "صف الصورة", "حلل الصورة", "وصف", "describe", "caption"]):
        return "describe"
    if any(k in t for k in ["نقل الوضعية", "انقل الوضعية", "pose transfer", "pose"]):
        return "pose"
    return "edit"


@app.post("/api/command")
def command():
    try:
        instruction = request.form.get("instruction", "").strip()
        files = [f for f in request.files.getlist("images") if f and f.filename]
        if not instruction:
            return jsonify(error="اكتب طلبك أولًا."), 400
        if not files:
            return jsonify(error="ارفع صورة واحدة على الأقل مع الطلب."), 400
        if len(files) > 8:
            return jsonify(error="الحد الأقصى 8 صور في العملية الواحدة."), 400

        operation = classify_command(instruction)
        if operation == "enhance":
            quality = "ultra" if any(k in instruction.lower() for k in ["فائق", "فائقة", "ultra", "8x", "8×"]) else "high"
            job = UPLOADS / uuid.uuid4().hex
            out_dir = OUTPUTS / uuid.uuid4().hex
            job.mkdir(parents=True)
            out_dir.mkdir(parents=True)
            results = []
            for file in files:
                src = save_upload(file, job)
                output = out_dir / f"{src.stem}_command.png"
                resolved = find_facefusion()
                if not resolved:
                    raise RuntimeError("محرك تحسين الجودة غير مثبت بعد.")
                entry, cwd = resolved
                python = shutil.which("python") or shutil.which("python3")
                cmd = [
                    python, str(entry), "headless-run",
                    "-t", str(src), "-o", str(output),
                    "--processors", "frame_enhancer",
                    "--frame-enhancer-model", "real_esrgan_x4_fp16" if quality == "ultra" else "real_esrgan_x2_fp16",
                    "--frame-enhancer-blend", "88",
                    "--output-image-quality", "100",
                    "--output-image-scale", "4" if quality == "ultra" else "2",
                ]
                run_command(cmd, cwd)
                results.append({"name": output.name, "url": f"/files/{out_dir.name}/{output.name}"})
            return jsonify(operation=operation, results=results, quality=quality)

        if operation == "describe":
            job = UPLOADS / uuid.uuid4().hex
            job.mkdir(parents=True)
            results = []
            for file in files:
                src = save_upload(file, job)
                results.append({"name": file.filename, "description": ollama_describe(src, instruction)})
            return jsonify(operation=operation, results=results)

        if operation == "faceswap":
            if len(files) < 2:
                return jsonify(error="لتبديل الوجه: ارفع أولًا الصورة المرجعية ثم صورة/صور الهدف."), 400
            quality = "ultra" if any(k in instruction.lower() for k in ["فائق", "فائقة", "ultra"]) else "high"
            job = UPLOADS / uuid.uuid4().hex
            out_dir = OUTPUTS / uuid.uuid4().hex
            job.mkdir(parents=True)
            out_dir.mkdir(parents=True)
            source = save_upload(files[0], job)
            results = []
            for target in files[1:]:
                target_path = save_upload(target, job)
                output = out_dir / f"{target_path.stem}_command.png"
                facefusion_faceswap(
                    source, target_path, output,
                    model="hyperswap_1a_256", order="left-right",
                    reference_position=0, quality=quality, provider="auto",
                )
                results.append({"name": output.name, "url": f"/files/{out_dir.name}/{output.name}"})
            return jsonify(operation=operation, results=results, quality=quality)

        if operation == "pose":
            return jsonify(
                operation=operation,
                status="not_configured",
                error="محرك نقل الوضعية لم يتم ربطه بعد؛ لم أضع تنفيذًا وهميًا.",
            ), 503

        return jsonify(
            operation=operation,
            status="not_configured",
            error="المحرر التوليدي الحر لم يُربط بمحرك ComfyUI بعد؛ الطلب حُفظ كأمر حر دون تنفيذ وهمي.",
        ), 503


@app.get("/files/<folder>/<name>")
def files(folder: str, name: str):
    return send_from_directory(OUTPUTS / Path(folder).name, secure_filename(name), as_attachment=False)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "7860")), debug=False)
