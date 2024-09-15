from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from PIL import Image, ImageDraw, ImageFont
import io
import numpy as np
from ultralytics import YOLO
from typing import List, Dict

app = FastAPI()

# Load YOLOv8 model
model = YOLO("/Users/mohammedhusain52/Desktop/Blockchain/PCB/runs/detect/train3/weights/best.pt")  # Replace with the path to your trained model

class PredictionResponse(BaseModel):
    boxes: List[Dict[str, float]]

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    confidence: float = Form(...)):

    try:
        # Read image data
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))

        # Convert image to numpy array
        image_np = np.array(image)

        # Run inference
        results = model(image_np, conf=confidence)

        # Extract bounding boxes and labels
        predictions = []
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            labels = result.names

            for box, score in zip(boxes, scores):
                predictions.append({
                    "class_id": int(box[0]),
                    "x_center": (box[0] + box[2]) / 2,
                    "y_center": (box[1] + box[3]) / 2,
                    "width": box[2] - box[0],
                    "height": box[3] - box[1],
                    "confidence": float(score)
                })

        return {"boxes": predictions}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/visualize")
async def visualize(
    file: UploadFile = File(...),
    confidence: float = Form(...)):

    try:
        # Read image data
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))

        # Convert image to numpy array
        image_np = np.array(image)

        # Run inference
        results = model(image_np, conf=confidence)

        # Draw bounding boxes on the image
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()  # You can load a different font if needed

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            labels = result.names

            for box, score in zip(boxes, scores):
                if score >= confidence:
                    x1, y1, x2, y2 = map(int, box)
                    draw.rectangle([x1, y1, x2, y2], outline="green", width=3)
                    label = f"{labels[int(box[0])]}: {score:.2f}"
                    draw.text((x1, y1 - 10), label, fill="green", font=font)

        # Convert image to bytes
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_bytes = buffered.getvalue()

        return StreamingResponse(io.BytesIO(img_bytes), media_type="image/jpeg")

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
