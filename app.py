from fastapi import FastAPI
from fastapi.responses import JSONResponse
import base64
import uvicorn
import os
from data_downloader import Construct_Dataset
import seisbench.data as sbd
from data_filter import get_spectrogram_and_bbox
from yolo  import predict
from data_traces import get_trace
from typing import List, Optional
from pydantic import BaseModel

app = FastAPI()
class ImageResponse(BaseModel):
    filename: str
    content: Optional[str] = None  # Base64 content of the image
    error: Optional[str] = None  # Error message if applicable

class GetImageResponse(BaseModel):
    images: List[ImageResponse]

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

@app.get("/download_data")
def download_data(startdate: str, enddate: str):
    Construct_Dataset(startdate, enddate)
    data = sbd.WaveformDataset('Waveforms', cache='full')
    traces = data.get_waveforms()  # Replace with the correct method to get waveforms
    return {"data": len(data), "traces": len(traces)}

@app.get("/predict")
def predict_image():
    os.makedirs('data', exist_ok=True)
    os.makedirs('box', exist_ok=True)
    data = sbd.WaveformDataset('Waveforms', cache='full')
    results = []
    for number_trace in range(len(data)):
        text = get_spectrogram_and_bbox(number_trace, data)
        results.append((text, number_trace))
    predict(f"data")
    return results

def encode_image_to_base64(file_path: str) -> str:
    with open(file_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded

@app.get("/get_image", response_model=GetImageResponse)
def get_image(startdate: str, enddate: str):
    traces_index = get_trace(startdate, enddate)
    results = []
    for number_trace in traces_index:
        for path in [f"box/{number_trace}.png", f"pred/anno/{number_trace}.jpg"]:
            try:
                encoded_image = encode_image_to_base64(path)
                results.append({"filename": path, "content": encoded_image})
            except FileNotFoundError:
                results.append({"filename": path, "error": "File not found"})
    return JSONResponse(content={"images": results})

if __name__ == "__main__":
  # uvicorn.run(app, host="0.0.0.0", port=8000) # dev
    uvicorn.run(app, host="0.0.0.0", port=8000) # prod