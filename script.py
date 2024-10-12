from fastapi import FastAPI
import uvicorn

from data_downloader import write_data
import seisbench.data as sbd
from data_filter import get_spectrogram_and_bbox
from yolo  import predict
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

@app.get("/download_data")
def download_data(startdate: str, enddate: str):
    # write_data(startdate, enddate)
    number_trace = 2
    data = sbd.WaveformDataset('data_v2_denoised' , cache = 'full')
    text = get_spectrogram_and_bbox(number_trace,data)
    predict(f"{number_trace}.png")

 
    return {"yolo": text}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)