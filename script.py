from fastapi import FastAPI
from fastapi.responses import FileResponse

import uvicorn
import os
from data_downloader import Construct_Dataset
import seisbench.data as sbd
from data_filter import get_spectrogram_and_bbox
from yolo  import predict
from data_traces import get_trace
app = FastAPI()

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


@app.get("/get_image")
def get_image(startdate: str, enddate: str):
    traces_index = get_trace(startdate, enddate)
    results = []
    for number_trace in traces_index:
        results.append(f"data/{number_trace}.png")
        results.append(f"pred/anno/{number_trace}.jpg")
    return FileResponse(results[0], media_type="image/png")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=800)