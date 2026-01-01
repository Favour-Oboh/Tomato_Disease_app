from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from img_prep import img_process

# ================= FLASK =================
app = Flask(__name__)

# ================= MODEL =================
class EfficientNet(nn.Module):
    def __init__(self, num_classes=3):
        super(EfficientNet, self).__init__()
        self.rgb_model = models.efficientnet_b3(
            weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1
        )
        self.rgb_model.classifier = nn.Identity()
        self.fc_out = nn.Linear(1536, num_classes)

    def forward(self, img_rgb):
        rgb_feat = self.rgb_model(img_rgb)
        return self.fc_out(rgb_feat)

device = torch.device("cpu")
model = EfficientNet().to(device)

model.load_state_dict(
    torch.load("https://github.com/Favour-Oboh/Tomato_Disease_app/blob/main/EfficientNetB3.pt", map_location=device)
)
model.eval()

classnames = [
    'Tomato_Bacterial_spot',
    'Tomato_Leaf_Mold',
    'Tomato_healthy'
]

# ================= PREDICTION =================
def prediction(tensor):
    tensor = tensor.to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)
        conf, pred = torch.max(probs, dim=1)

    return {
        "class": classnames[pred.item()],
        "confidence": round(conf.item() * 100, 2)
    }

# ================= ROUTES =================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files["image"]

    tensor = img_process(image_file)  
    result = prediction(tensor)

    return jsonify(result)

# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True)
