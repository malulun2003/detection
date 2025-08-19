import requests
from fastapi import FastAPI
import uvicorn
import smartphone_detection

app = FastAPI()

@app.get("/detect")
async def detect_objects(video_path, output_path, interval=10, driver=0):
    detections = smartphone_detection.sf_detection(video_path, output_path, interval=5, driver=1)
    return {"detections": detections}

if __name__ == "__main__":
    uvicorn.run(app, port=38080)
