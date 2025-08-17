import requests

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from os import environ

environ["TRANSFORMERS_OFFLINE"] = "1"


model_id = "IDEA-Research/grounding-dino-base"
device = "cpu"

processor = AutoProcessor.from_pretrained("models--IDEA-Research--grounding-dino-base")
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/detect")
async def detect_objects(image_name, prompt):
	image = Image.open(image_name)
	inputs = processor(images=image, text=[prompt], return_tensors="pt").to(device)
	with torch.no_grad():
		outputs = model(**inputs)
	results = processor.post_process_grounded_object_detection(
		outputs,
		inputs.input_ids,
		# box_threshold=0.4,
		text_threshold=0.3,
		target_sizes=[image.size[::-1]]
	)

	result = results[0]
	for box, score, labels in zip(result["boxes"], result["scores"], result["labels"]):
		box = [round(x, 2) for x in box.tolist()]
		print(f"Detected {labels} with confidence {round(score.item(), 3)} at location {box}")

	detections = []
	for box, score, labels in zip(result["boxes"], result["scores"], result["labels"]):
		box = [round(x, 2) for x in box.tolist()]
		detections.append({
			"label": labels,
			"confidence": round(score.item(), 3),
			"box": box
		})
	return {"detections": detections}

if __name__ == "__main__":
	uvicorn.run(app, port=38080)

