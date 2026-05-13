"""Flask dashboard for plant disease detection."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from flask import Flask, jsonify, render_template, request

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.transforms import get_eval_transforms
from src.models.model_factory import get_model

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload limit

# ── Model state (loaded once at startup) ────────────────────────────────────
_model = None
_class_names: list[str] = []
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_transform = get_eval_transforms(224)

CHECKPOINT = _ROOT / "results" / "models" / "best_resnet50_joint_aug.pt"
METRICS_FILE = _ROOT / "results" / "metrics" / "eval_resnet50_plantdoc_test.json"
HISTORY_FILE = _ROOT / "results" / "models" / "history_resnet50_joint_aug.json"

DISEASE_INFO: dict[str, dict] = {
    "Apple__Apple_scab": {
        "common": "Apple Scab",
        "treatment": "Apply fungicides (captan, myclobutanil) at bud break. Remove infected leaves. Improve air circulation.",
        "severity": "medium",
    },
    "Apple__Cedar_apple_rust": {
        "common": "Cedar Apple Rust",
        "treatment": "Apply fungicides when orange spots appear. Remove nearby cedar trees if possible.",
        "severity": "medium",
    },
    "Apple__healthy": {"common": "Healthy Apple", "treatment": "No treatment needed.", "severity": "none"},
    "Blueberry__healthy": {"common": "Healthy Blueberry", "treatment": "No treatment needed.", "severity": "none"},
    "Cherry_(including_sour)__healthy": {"common": "Healthy Cherry", "treatment": "No treatment needed.", "severity": "none"},
    "Corn_(maize)__Cercospora_leaf_spot Gray_leaf_spot": {
        "common": "Corn Gray Leaf Spot",
        "treatment": "Use resistant hybrids. Apply fungicides (azoxystrobin, propiconazole). Rotate crops.",
        "severity": "high",
    },
    "Corn_(maize)__Common_rust_": {
        "common": "Corn Common Rust",
        "treatment": "Plant resistant varieties. Apply fungicides early if severe. Monitor regularly.",
        "severity": "medium",
    },
    "Corn_(maize)__Northern_Leaf_Blight": {
        "common": "Northern Corn Leaf Blight",
        "treatment": "Use resistant hybrids. Apply fungicides at early tasseling stage.",
        "severity": "high",
    },
    "Grape__Black_rot": {
        "common": "Grape Black Rot",
        "treatment": "Remove mummified fruit. Apply fungicides (mancozeb, myclobutanil) from bud break.",
        "severity": "high",
    },
    "Grape__healthy": {"common": "Healthy Grape", "treatment": "No treatment needed.", "severity": "none"},
    "Peach__healthy": {"common": "Healthy Peach", "treatment": "No treatment needed.", "severity": "none"},
    "Pepper,_bell__Bacterial_spot": {
        "common": "Pepper Bacterial Spot",
        "treatment": "Use copper-based bactericides. Remove infected plants. Avoid overhead irrigation.",
        "severity": "high",
    },
    "Pepper,_bell__healthy": {"common": "Healthy Bell Pepper", "treatment": "No treatment needed.", "severity": "none"},
    "Potato__Early_blight": {
        "common": "Potato Early Blight",
        "treatment": "Apply fungicides (chlorothalonil, mancozeb). Remove lower infected leaves. Ensure proper nutrition.",
        "severity": "medium",
    },
    "Potato__Late_blight": {
        "common": "Potato Late Blight",
        "treatment": "Apply fungicides immediately (metalaxyl, cymoxanil). Remove and destroy infected plants. This disease spreads rapidly.",
        "severity": "high",
    },
    "Raspberry__healthy": {"common": "Healthy Raspberry", "treatment": "No treatment needed.", "severity": "none"},
    "Soybean__healthy": {"common": "Healthy Soybean", "treatment": "No treatment needed.", "severity": "none"},
    "Squash__Powdery_mildew": {
        "common": "Squash Powdery Mildew",
        "treatment": "Apply sulfur or potassium bicarbonate sprays. Improve air circulation. Avoid overhead watering.",
        "severity": "medium",
    },
    "Strawberry__healthy": {"common": "Healthy Strawberry", "treatment": "No treatment needed.", "severity": "none"},
    "Tomato__Bacterial_spot": {
        "common": "Tomato Bacterial Spot",
        "treatment": "Use copper-based bactericides. Remove infected leaves. Avoid working with wet plants.",
        "severity": "high",
    },
    "Tomato__Early_blight": {
        "common": "Tomato Early Blight",
        "treatment": "Apply fungicides (chlorothalonil). Remove lower leaves. Mulch around plants.",
        "severity": "medium",
    },
    "Tomato__Late_blight": {
        "common": "Tomato Late Blight",
        "treatment": "Apply fungicides immediately. Remove infected plants. Avoid overhead irrigation.",
        "severity": "high",
    },
    "Tomato__Leaf_Mold": {
        "common": "Tomato Leaf Mold",
        "treatment": "Improve ventilation in greenhouses. Apply fungicides. Reduce humidity.",
        "severity": "medium",
    },
    "Tomato__Septoria_leaf_spot": {
        "common": "Tomato Septoria Leaf Spot",
        "treatment": "Apply fungicides (chlorothalonil, mancozeb). Remove infected leaves. Avoid wetting foliage.",
        "severity": "medium",
    },
    "Tomato__Tomato_Yellow_Leaf_Curl_Virus": {
        "common": "Tomato Yellow Leaf Curl Virus",
        "treatment": "Control whitefly vectors with insecticides. Use resistant varieties. Remove and destroy infected plants.",
        "severity": "high",
    },
    "Tomato__Tomato_mosaic_virus": {
        "common": "Tomato Mosaic Virus",
        "treatment": "No cure. Remove infected plants. Disinfect tools. Use virus-free seed.",
        "severity": "high",
    },
    "Tomato__healthy": {"common": "Healthy Tomato", "treatment": "No treatment needed.", "severity": "none"},
}


def load_model() -> None:
    global _model, _class_names
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")
    ckpt = torch.load(CHECKPOINT, map_location="cpu")
    _class_names = ckpt.get("class_names") or sorted(DISEASE_INFO.keys())

    if METRICS_FILE.exists():
        with METRICS_FILE.open() as f:
            metrics = json.load(f)
        _class_names = metrics["class_names"]

    _model = get_model("resnet50", num_classes=len(_class_names), pretrained=False)
    _model.load_state_dict(ckpt["model_state_dict"])
    _model.eval()
    _model.to(_device)
    print(f"Model loaded — {len(_class_names)} classes on {_device}")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
    except Exception:
        return jsonify({"error": "Invalid image file"}), 400

    tensor = _transform(img).unsqueeze(0).to(_device)

    with torch.no_grad():
        logits = _model(tensor)
        probs = F.softmax(logits, dim=1)[0]

    top5_probs, top5_idx = probs.topk(5)
    top5 = [
        {
            "label": _class_names[i],
            "probability": round(float(p), 4),
            "info": DISEASE_INFO.get(_class_names[i], {"common": _class_names[i], "treatment": "N/A", "severity": "unknown"}),
        }
        for p, i in zip(top5_probs.tolist(), top5_idx.tolist())
    ]

    return jsonify({"predictions": top5})


@app.route("/api/metrics")
def metrics():
    result = {}
    if METRICS_FILE.exists():
        with METRICS_FILE.open() as f:
            data = json.load(f)
        result["eval"] = {
            "accuracy": round(data["accuracy"], 4),
            "precision_macro": round(data["precision_macro"], 4),
            "recall_macro": round(data["recall_macro"], 4),
            "f1_macro": round(data["f1_macro"], 4),
            "num_samples": data["num_samples"],
            "split": data["split"],
        }
    if HISTORY_FILE.exists():
        with HISTORY_FILE.open() as f:
            history = json.load(f)
        result["history"] = {
            "train_losses": history["train_losses"],
            "val_losses": history["val_losses"],
            "val_accuracies": history["val_accuracies"],
            "epochs_run": history["epochs_run"],
        }
    return jsonify(result)


if __name__ == "__main__":
    load_model()
    app.run(debug=False, host="0.0.0.0", port=5000)
