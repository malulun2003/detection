import requests

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from os import environ

environ["TRANSFORMERS_OFFLINE"] = "1"


model_id = "IDEA-Research/grounding-dino-base"
device = "cpu"

processor = AutoProcessor.from_pretrained("C:\\tool\\paligemma\\models--IDEA-Research--grounding-dino-tiny")
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

# image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(image_url, stream=True).raw)
image = Image.open("taxi1.jpg")  # Use your local image file here

# Check for cats and remote controls
text_labels = [["a cell phone", "People using cell phone"]]
text_labels = "People using cell phone"
text_labels = "People drive car"

inputs = processor(images=image, text=text_labels, return_tensors="pt").to(device)
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


import flet as ft

def main(page: ft.Page):
    page.title = "Grounding DINO Object Detection"
    page.vertical_alignment = ft.MainAxisAlignment.CENTER

    image_widget = ft.Image(src="taxi1.jpg", width=600, height=400)
    result_text = ft.Text("Detection results will be displayed here.", size=20)

    def update_results():
        result_text.value = "\n".join(
            f"Detected {labels} with confidence {round(score.item(), 3)} at location {box}"
            for box, score, labels in zip(result["boxes"], result["scores"], result["labels"])
        )
        page.update()

    update_results()

    page.add(image_widget, result_text)

if __name__ == "__main__":
    # ft.app(target=main, view=ft.WEB_BROWSER, port=8080, assets_dir="assets")
    ft.app(target=main)
