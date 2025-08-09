import torch
from PIL import Image
from torchvision import transforms
import timm
import torch.nn as nn
import os
import io
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

# --- Config ---
NUM_CLASSES = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Transform (same as training) ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# --- Load Model ---
model = timm.create_model('mobilevit_s', pretrained=False, num_classes=NUM_CLASSES)
model.load_state_dict(torch.load("mobilevit_best_model.pth", map_location=device))
model = model.to(device)
model.eval()

# --- Class Names (Replace with your dataset folder names) ---
class_names = [
    "Bacterial Leaf Blight",
    "Brown Spot",
    "Leaf Smut",
    "Tungro",
    "Sheath Blight",
    "Narrow Brown Spot",
    "Blast",
    "False Smut",
    "Healthy",
    "Other"
]

# --- FastAPI app ---
app = FastAPI()

@app.get("/")
def root():
    return {"message": "Rice Disease Model API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read and process image
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)

        # Model prediction
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)[0]
            pred_idx = torch.argmax(probs).item()
            pred_class = class_names[pred_idx]
            confidence = float(probs[pred_idx].item() * 100)

        return JSONResponse({
            "predicted_class": pred_class,
            "confidence": round(confidence, 2)
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
