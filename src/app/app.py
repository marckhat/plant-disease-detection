"""Verdis — premium demo dashboard for plant disease detection."""

from __future__ import annotations

import base64
import io
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from flask import Flask, jsonify, render_template, request, send_from_directory

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD, get_eval_transforms
from src.models.model_factory import get_model

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

# ── Model state (loaded once at startup) ────────────────────────────────────
_model = None
_class_names: list[str] = []
# Force CPU — keeps the demo safe even when the GPU is busy.
_device = torch.device("cpu")
_transform = get_eval_transforms(224)

MODELS_DIR = _ROOT / "results" / "models"
METRICS_DIR = _ROOT / "results" / "metrics"
CONFUSION_DIR = _ROOT / "results" / "confusion_matrices"

CHECKPOINT = MODELS_DIR / "best_resnet50_joint_all.pt"
METRICS_FILE_RESNET = METRICS_DIR / "eval_resnet50_plantdoc_test.json"
METRICS_FILE_EFFNET = METRICS_DIR / "eval_efficientnet_b0_plantdoc_test.json"

# ── Curated sample gallery (served from static/samples/) ────────────────────
SAMPLES = [
    {"file": "1_tomato_healthy.jpg",          "label": "Tomato (healthy)",       "expected": "Tomato__healthy",                              "difficulty": "easy"},
    {"file": "2_grape_black_rot.jpg",         "label": "Grape — Black Rot",      "expected": "Grape__Black_rot",                             "difficulty": "easy"},
    {"file": "3_corn_common_rust.jpg",        "label": "Corn — Common Rust",     "expected": "Corn_(maize)__Common_rust_",                   "difficulty": "medium"},
    {"file": "4_strawberry_healthy.jpg",      "label": "Strawberry (healthy)",   "expected": "Strawberry__healthy",                          "difficulty": "easy"},
    {"file": "5_potato_late_blight_HARD.jpg", "label": "Potato — Late Blight",   "expected": "Potato__Late_blight",                          "difficulty": "hard"},
]

DISEASE_INFO: dict[str, dict] = {
    "Apple__Apple_scab":                                  {"common": "Apple Scab",                  "treatment": "Apply fungicides (captan, myclobutanil) at bud break. Remove infected leaves. Improve air circulation.",                "severity": "medium"},
    "Apple__Cedar_apple_rust":                            {"common": "Cedar Apple Rust",            "treatment": "Apply fungicides when orange spots appear. Remove nearby cedar trees if possible.",                                     "severity": "medium"},
    "Apple__healthy":                                     {"common": "Healthy Apple",               "treatment": "No treatment needed.",                                                                                                  "severity": "none"},
    "Blueberry__healthy":                                 {"common": "Healthy Blueberry",           "treatment": "No treatment needed.",                                                                                                  "severity": "none"},
    "Cherry_(including_sour)__healthy":                   {"common": "Healthy Cherry",              "treatment": "No treatment needed.",                                                                                                  "severity": "none"},
    "Corn_(maize)__Cercospora_leaf_spot Gray_leaf_spot":  {"common": "Corn Gray Leaf Spot",         "treatment": "Use resistant hybrids. Apply fungicides (azoxystrobin, propiconazole). Rotate crops.",                                  "severity": "high"},
    "Corn_(maize)__Common_rust_":                         {"common": "Corn Common Rust",            "treatment": "Plant resistant varieties. Apply fungicides early if severe. Monitor regularly.",                                       "severity": "medium"},
    "Corn_(maize)__Northern_Leaf_Blight":                 {"common": "Northern Corn Leaf Blight",   "treatment": "Use resistant hybrids. Apply fungicides at early tasseling stage.",                                                     "severity": "high"},
    "Grape__Black_rot":                                   {"common": "Grape Black Rot",             "treatment": "Remove mummified fruit. Apply fungicides (mancozeb, myclobutanil) from bud break.",                                     "severity": "high"},
    "Grape__healthy":                                     {"common": "Healthy Grape",               "treatment": "No treatment needed.",                                                                                                  "severity": "none"},
    "Peach__healthy":                                     {"common": "Healthy Peach",               "treatment": "No treatment needed.",                                                                                                  "severity": "none"},
    "Pepper,_bell__Bacterial_spot":                       {"common": "Pepper Bacterial Spot",       "treatment": "Use copper-based bactericides. Remove infected plants. Avoid overhead irrigation.",                                    "severity": "high"},
    "Pepper,_bell__healthy":                              {"common": "Healthy Bell Pepper",         "treatment": "No treatment needed.",                                                                                                  "severity": "none"},
    "Potato__Early_blight":                               {"common": "Potato Early Blight",         "treatment": "Apply fungicides (chlorothalonil, mancozeb). Remove lower infected leaves. Ensure proper nutrition.",                  "severity": "medium"},
    "Potato__Late_blight":                                {"common": "Potato Late Blight",          "treatment": "Apply fungicides immediately (metalaxyl, cymoxanil). Remove and destroy infected plants. This disease spreads rapidly.", "severity": "high"},
    "Raspberry__healthy":                                 {"common": "Healthy Raspberry",           "treatment": "No treatment needed.",                                                                                                  "severity": "none"},
    "Soybean__healthy":                                   {"common": "Healthy Soybean",             "treatment": "No treatment needed.",                                                                                                  "severity": "none"},
    "Squash__Powdery_mildew":                             {"common": "Squash Powdery Mildew",       "treatment": "Apply sulfur or potassium bicarbonate sprays. Improve air circulation. Avoid overhead watering.",                       "severity": "medium"},
    "Strawberry__healthy":                                {"common": "Healthy Strawberry",          "treatment": "No treatment needed.",                                                                                                  "severity": "none"},
    "Tomato__Bacterial_spot":                             {"common": "Tomato Bacterial Spot",       "treatment": "Use copper-based bactericides. Remove infected leaves. Avoid working with wet plants.",                                  "severity": "high"},
    "Tomato__Early_blight":                               {"common": "Tomato Early Blight",         "treatment": "Apply fungicides (chlorothalonil). Remove lower leaves. Mulch around plants.",                                          "severity": "medium"},
    "Tomato__Late_blight":                                {"common": "Tomato Late Blight",          "treatment": "Apply fungicides immediately. Remove infected plants. Avoid overhead irrigation.",                                       "severity": "high"},
    "Tomato__Leaf_Mold":                                  {"common": "Tomato Leaf Mold",            "treatment": "Improve ventilation in greenhouses. Apply fungicides. Reduce humidity.",                                               "severity": "medium"},
    "Tomato__Septoria_leaf_spot":                         {"common": "Tomato Septoria Leaf Spot",   "treatment": "Apply fungicides (chlorothalonil, mancozeb). Remove infected leaves. Avoid wetting foliage.",                            "severity": "medium"},
    "Tomato__Tomato_Yellow_Leaf_Curl_Virus":              {"common": "Tomato Yellow Leaf Curl Virus","treatment": "Control whitefly vectors with insecticides. Use resistant varieties. Remove and destroy infected plants.",                "severity": "high"},
    "Tomato__Tomato_mosaic_virus":                        {"common": "Tomato Mosaic Virus",         "treatment": "No cure. Remove infected plants. Disinfect tools. Use virus-free seed.",                                                "severity": "high"},
    "Tomato__healthy":                                    {"common": "Healthy Tomato",              "treatment": "No treatment needed.",                                                                                                  "severity": "none"},
}


def pretty_label(class_name: str) -> str:
    """'Tomato__Bacterial_spot' → 'Tomato › Bacterial spot'."""
    if "__" in class_name:
        plant, disease = class_name.split("__", 1)
    else:
        plant, disease = class_name, ""
    plant = plant.replace("_(maize)", "").replace("_(including_sour)", "").replace("_", " ").strip()
    disease = disease.replace("_", " ").strip()
    disease = re.sub(r"\s+", " ", disease).strip()
    if not disease:
        return plant
    return f"{plant} › {disease}"


def load_model() -> None:
    global _model, _class_names
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")
    ckpt = torch.load(CHECKPOINT, map_location="cpu")
    _class_names = ckpt.get("class_names") or sorted(DISEASE_INFO.keys())

    if METRICS_FILE_RESNET.exists():
        with METRICS_FILE_RESNET.open() as f:
            _class_names = json.load(f)["class_names"]

    _model = get_model("resnet50", num_classes=len(_class_names), pretrained=False)
    _model.load_state_dict(ckpt["model_state_dict"])
    _model.eval()
    _model.to(_device)
    print(f"Model loaded — {len(_class_names)} classes on {_device}")


def predict_image(img: Image.Image) -> list[dict]:
    tensor = _transform(img).unsqueeze(0).to(_device)
    with torch.no_grad():
        logits = _model(tensor)
        probs = F.softmax(logits, dim=1)[0]
    top5_probs, top5_idx = probs.topk(5)
    out = []
    for p, i in zip(top5_probs.tolist(), top5_idx.tolist()):
        name = _class_names[i]
        info = DISEASE_INFO.get(name, {"common": name, "treatment": "N/A", "severity": "unknown"})
        out.append({
            "label": name,
            "pretty": pretty_label(name),
            "probability": round(float(p), 4),
            "info": info,
        })
    return out


# ── Grad-CAM ────────────────────────────────────────────────────────────────
def gradcam_overlay(img: Image.Image, target_class: int | None = None) -> tuple[str, int]:
    """Run Grad-CAM on the loaded ResNet50 and return base64 PNG of the overlay."""
    _model.eval()
    rgb = img.convert("RGB").resize((224, 224))
    tensor = _transform(rgb).unsqueeze(0).to(_device)
    tensor.requires_grad_(True)

    activations: dict = {}
    gradients: dict = {}

    def fwd_hook(_m, _i, out):
        activations["v"] = out

    def bwd_hook(_m, _g_in, g_out):
        gradients["v"] = g_out[0]

    layer = _model.layer4  # final ResNet50 conv block
    h1 = layer.register_forward_hook(fwd_hook)
    h2 = layer.register_full_backward_hook(bwd_hook)
    try:
        logits = _model(tensor)
        target_idx = int(logits.argmax(dim=1).item()) if target_class is None else int(target_class)
        _model.zero_grad()
        logits[0, target_idx].backward()

        acts = activations["v"].detach()[0]      # [C, H, W]
        grads = gradients["v"].detach()[0]       # [C, H, W]
        weights = grads.mean(dim=(1, 2))         # [C]
        cam = torch.relu((weights[:, None, None] * acts).sum(dim=0))
        cam = cam.cpu().numpy()
    finally:
        h1.remove()
        h2.remove()

    if cam.max() <= 0:
        cam = np.zeros_like(cam)
    else:
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)
    cam_arr = np.asarray(cam_img) / 255.0

    # Custom "ember" colormap — better on dark UI than jet/viridis
    base = np.asarray(rgb, dtype=np.float32) / 255.0
    r = np.clip(cam_arr * 1.4,                0, 1)
    g = np.clip(cam_arr * 0.6 - 0.15,         0, 1)
    b = np.clip(cam_arr * 0.15 - 0.05,        0, 1)
    heat = np.stack([r, g, b], axis=-1)
    alpha = (cam_arr ** 0.7)[..., None] * 0.65
    blended = (1 - alpha) * base + alpha * heat
    out_img = Image.fromarray((blended * 255).clip(0, 255).astype(np.uint8))

    buf = io.BytesIO()
    out_img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii"), target_idx


# ── Helpers ────────────────────────────────────────────────────────────────
def _read_metrics(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def _per_class_from_report(report) -> list[dict]:
    """Convert sklearn classification_report (dict or text) into a list of per-class rows."""
    SKIP = {"accuracy", "macro avg", "weighted avg"}
    rows: list[dict] = []
    if isinstance(report, dict):
        for cls, vals in report.items():
            if cls in SKIP or not isinstance(vals, dict):
                continue
            rows.append({
                "class": cls,
                "pretty": pretty_label(cls),
                "precision": float(vals.get("precision", 0)),
                "recall":    float(vals.get("recall", 0)),
                "f1":        float(vals.get("f1-score", 0)),
                "support":   int(vals.get("support", 0)),
            })
        return rows
    # Fallback: text format
    for line in str(report).splitlines():
        line = line.rstrip()
        if not line.strip() or line.strip().startswith(tuple(SKIP) + ("precision",)):
            continue
        m = re.match(r"^\s*(.+?)\s{2,}([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s*$", line)
        if not m:
            continue
        cls, prec, rec, f1, support = m.groups()
        rows.append({
            "class": cls.strip(),
            "pretty": pretty_label(cls.strip()),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "support": int(support),
        })
    return rows


# ── Routes ──────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/samples/<path:filename>")
def samples_file(filename: str):
    return send_from_directory(_ROOT / "src" / "app" / "static" / "samples", filename)


@app.route("/api/samples")
def api_samples():
    items = []
    for s in SAMPLES:
        info = DISEASE_INFO.get(s["expected"], {"common": s["expected"], "severity": "none"})
        items.append({
            "file": s["file"],
            "url": f"/static/samples/{s['file']}",
            "label": s["label"],
            "expected_class": s["expected"],
            "expected_pretty": pretty_label(s["expected"]),
            "difficulty": s["difficulty"],
            "common": info.get("common"),
        })
    return jsonify({"samples": items})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    try:
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
    except Exception:
        return jsonify({"error": "Invalid image file"}), 400
    return jsonify({"predictions": predict_image(img)})


@app.route("/api/gradcam", methods=["POST"])
def api_gradcam():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    file = request.files["image"]
    try:
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
    except Exception:
        return jsonify({"error": "Invalid image file"}), 400
    overlay_b64, target_idx = gradcam_overlay(img)
    return jsonify({
        "overlay": f"data:image/png;base64,{overlay_b64}",
        "target_class": _class_names[target_idx],
        "target_pretty": pretty_label(_class_names[target_idx]),
    })


@app.route("/api/metrics")
def api_metrics():
    d = _read_metrics(METRICS_FILE_RESNET) or {}
    return jsonify({
        "accuracy":        round(d.get("accuracy", 0), 4),
        "precision_macro": round(d.get("precision_macro", 0), 4),
        "recall_macro":    round(d.get("recall_macro", 0), 4),
        "f1_macro":        round(d.get("f1_macro", 0), 4),
        "num_samples":     d.get("num_samples"),
        "split":           d.get("split"),
    })


@app.route("/api/metrics-all")
def api_metrics_all():
    out = {}
    for key, path in (("resnet50", METRICS_FILE_RESNET), ("efficientnet_b0", METRICS_FILE_EFFNET)):
        d = _read_metrics(path)
        if d is None:
            continue
        out[key] = {
            "accuracy":        round(d["accuracy"], 4),
            "precision_macro": round(d["precision_macro"], 4),
            "recall_macro":    round(d["recall_macro"], 4),
            "f1_macro":        round(d["f1_macro"], 4),
            "num_samples":     d["num_samples"],
            "split":           d["split"],
        }
    return jsonify(out)


@app.route("/api/per-class")
def api_per_class():
    out: dict = {}
    for key, path in (("resnet50", METRICS_FILE_RESNET), ("efficientnet_b0", METRICS_FILE_EFFNET)):
        d = _read_metrics(path)
        if d is None or "classification_report" not in d:
            continue
        out[key] = _per_class_from_report(d["classification_report"])
    return jsonify(out)


if __name__ == "__main__":
    load_model()
    app.run(debug=False, host="0.0.0.0", port=5000)
