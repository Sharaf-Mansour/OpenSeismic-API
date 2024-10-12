from fastapi import FastAPI
import uvicorn

from data_downloader import Construct_Dataset
import seisbench.data as sbd
from data_filter import get_spectrogram_and_bbox
from yolo  import predict
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

@app.get("/download_data")
def download_data(startdate: str, enddate: str):
    Construct_Dataset(startdate, enddate)
    data = sbd.WaveformDataset('Waveforms', cache='full')
    results = []
    for number_trace in range(len(data)):
        text = get_spectrogram_and_bbox(number_trace, data)
        predict(f"data/{number_trace}.png")
        results.append((text, number_trace))
    
    return {"results": results}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)