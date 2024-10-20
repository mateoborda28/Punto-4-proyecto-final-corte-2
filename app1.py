from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import shutil
from pathlib import Path
from PIL import Image
import torch
from ultralytics import YOLO
import json 
import matplotlib.pyplot as plt 
import cv2
import uvicorn

app = FastAPI()

# Directory to save uploaded files
UPLOAD_DIR = Path("uploads")

UPLOAD_DIR.mkdir(exist_ok=True)

# carga de modelo Yolo
model = YOLO('yolov8n.pt')  

@app.post("/predict-image/")
async def upload_image(file: UploadFile = File(...)):
  try:
    file_path = UPLOAD_DIR / file.filename
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    with open("atributos.json", 'r') as file:
        atributos = json.load(file)


    image = cv2.imread(str(file_path))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convertir de BGR a RGB para matplotlib

    # Realizar la detección de objetos
    results = model(image_rgb)
    detections = results[0].boxes.data.cpu().numpy()
    class_names = model.names
  

    for detection in detections:
     x1, y1, x2, y2, conf, cls = detection
     class_names = class_names[int(cls)]
     outputs = {
        "Clase" : class_names,
        "Confianza" : round(float(conf),3),
        "Coord1" : [round(float(x1), 4), round(float(y1), 4)],
        "Coord2" : [round(float(x2), 4), round(float(y2), 4)]
      }
     atributos.append(outputs)
     print(f"Clase: {class_names}, Confianza: {conf}, Coordenadas: ({x1}, {y1}, {x2}, {y2})")


    with open('atributos.json', 'w') as json_file:
       json.dump(atributos, json_file)
    return f"Proceso Exitoso"
  except Exception as e:

     return {"error": f"Hubo un error: {str(e)}"}
  finally:
     if file_path.exists():
        file_path.unlink()


@app.post("/upload-imagen")
async def upload_image(file: UploadFile = File(...)):
    try:
        file_path = UPLOAD_DIR / file.filename
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        image = cv2.imread(str(file_path))

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = model(image_rgb)
        resultado_imagen = results[0].plot()
        path1 = UPLOAD_DIR / f"DETECCION_{file.filename}"
        cv2.imwrite(str(path1), resultado_imagen)

        return FileResponse(path1, media_type="image/jpeg")
    except Exception as e:
        return {"error": f"Ocurrió un error: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)