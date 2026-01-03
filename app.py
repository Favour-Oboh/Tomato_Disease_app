# import gradio as gr
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision.models as models
# from PIL import Image
# from img_prep import img_process  

# # ================= MODEL ARCHITECTURE =================
# class EfficientNet(nn.Module):
#     def __init__(self, num_classes=3):
#         super(EfficientNet, self).__init__()
#         self.rgb_model = models.efficientnet_b3(
#             weights=None # No need to download ImageNet weights again
#         )
#         self.rgb_model.classifier = nn.Identity()
#         self.fc_out = nn.Linear(1536, num_classes)

#     def forward(self, img_rgb):
#         rgb_feat = self.rgb_model(img_rgb)
#         return self.fc_out(rgb_feat)

# # ================= SETUP & LOAD =================
# device = torch.device("cpu")
# model = EfficientNet().to(device)

# # Using your preferred loading method
# model.load_state_dict(
#     torch.load("EfficientNetB3.pt", map_location=device, weights_only=True)
# )
# model.eval()

# classnames = [
#     'Tomato_Bacterial_spot',
#     'Tomato_Leaf_Mold',
#     'Tomato_healthy'
# ]

# # ================= PREDICTION FUNCTION =================
# def predict_image(img):
#     """
#     img: A PIL image provided by Gradio
#     """
#     if img is None:
#         return "Please upload an image."
   
#     # 1. Process the image using your existing helper
#     # Note: Ensure img_process can handle a PIL object or save it to a temp file
#     tensor = img_process(img)
   
#     # 2. Inference
#     with torch.no_grad():
#         logits = model(tensor)
#         probs = F.softmax(logits, dim=1)
#         conf, pred = torch.max(probs, dim=1)

#     # 3. Format result for Gradio
#     result_label = classnames[pred.item()]
#     confidence_score = round(conf.item() * 100, 2)
   
#     return f"Prediction: {result_label} ({confidence_score}%)"

# # ================= GRADIO UI =================
# # This replaces your Flask @app.route logic
# demo = gr.Interface(
#     fn=predict_image,
#     inputs=gr.Image(), # Accepts the image and converts to PIL for you
#     outputs="text",
#     title="Tomato Disease Classifier",
#     description="Upload a photo of a tomato leaf to identify diseases."
# )

# if __name__ == "__main__":
#     demo.launch(share = True)



import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image
from img_prep import img_process  # your preprocessing function

# ================= MODEL =================
class EfficientNet(nn.Module):
    def __init__(self, num_classes=3):
        super(EfficientNet, self).__init__()
        self.rgb_model = models.efficientnet_b3(weights=None)
        self.rgb_model.classifier = nn.Identity()
        self.fc_out = nn.Linear(1536, num_classes)

    def forward(self, img_rgb):
        rgb_feat = self.rgb_model(img_rgb)
        return self.fc_out(rgb_feat)

device = torch.device("cpu")
model = EfficientNet().to(device)
model.load_state_dict(torch.load("EfficientNetB3.pt", map_location=device))
model.eval()

classnames = [
    'Tomato_Bacterial_spot',
    'Tomato_Leaf_Mold',
    'Tomato_healthy'
]

# Recommendation mapping
recommendations = {
    "Tomato_Bacterial_spot": [
        "Remove infected leaves immediately",
        "Use copper-based sprays",
        "Avoid touching wet plants"
    ],
    "Tomato_Leaf_Mold": [
        "Improve air circulation around plants",
        "Avoid overhead watering",
        "Apply recommended fungicide"
    ],
    "Tomato_healthy": [
        "No disease detected",
        "Maintain proper watering and fertilization"
    ]
}

# ================= PREDICTION FUNCTION =================
def predict_image(img):
    if img is None:
        return "No image uploaded.", "", ""

    # Preprocess
    tensor = img_process(img)

    # Inference
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)
        conf, pred = torch.max(probs, dim=1)

    disease = classnames[pred.item()]
    confidence = round(conf.item() * 100, 2)

    # Prepare HTML for recommendations
    recs = recommendations.get(disease, ["Consult an agricultural expert"])
    rec_html = "<ul>"
    for item in recs:
        rec_html += f"<li>{item}</li>"
    rec_html += "</ul>"

    # Return values: prediction text, disease name, recommendation HTML
    return f"Prediction: {disease} ({confidence}%)", disease, rec_html

# ================= GRADIO INTERFACE =================
demo = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Textbox(label="Disease Name"),
        gr.HTML(label="Recommended Actions")
    ],
    title="Tomato Leaf Disease Classifier",
    description="Upload a tomato leaf image to detect disease and get treatment recommendations."
)

if __name__ == "__main__":
    demo.launch(share=True)
