from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import numpy as np
import io
from PIL import Image
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO("best.pt")  

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    image_np = np.array(image)

    results = model(image_np)

    detections = []
    tumor_count = 1  

    for result in results:
        for box, cls, score in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
            x1, y1, x2, y2 = map(int, box[:4])
            class_id = int(cls)  
            confidence = float(score)  
            class_name = model.names[class_id]  

            if class_name.lower() == "tumor":
                label = f"Tumor {tumor_count} ({confidence:.2%})" 
                tumor_count += 1
            else:
                label = f"{class_name} ({confidence:.2%})" 

            detections.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2, "label": label
            })

    return JSONResponse(content={"detections": detections})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
