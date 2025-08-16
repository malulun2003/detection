import io
import requests
import numpy as np
import PIL
from PIL import Image
import re
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
from transformers.image_utils import load_image
import torch
import sys

# --- 公式ユーティリティ関数 ---

# Big Vision リポジトリのコードを Python の import パスに追加（事前にクローン済みの前提）
if "big_vision_repo" not in sys.path:
    sys.path.append("big_vision_repo")

def crop_and_resize(image, target_size):
    width, height = image.size
    source_size = min(image.size)
    left = width // 2 - source_size // 2
    top = height // 2 - source_size // 2
    right, bottom = left + source_size, top + source_size
    return image.resize(target_size, box=(left, top, right, bottom))

def read_image(url, target_size):
    contents = io.BytesIO(requests.get(url).content)
    image = Image.open(contents)
    image = crop_and_resize(image, target_size)
    image = np.array(image)
    if image.shape[2] == 4:
        image = image[:, :, :3]
    return image

def parse_bbox_and_labels(detokenized_output: str):
    matches = re.finditer(
        r'<loc(?P<y0>\d{4})><loc(?P<x0>\d{4})><loc(?P<y1>\d{4})><loc(?P<x1>\d{4})> (?P<label>.+?)( ;|$)',
        detokenized_output,
    )
    labels, boxes = [], []
    fmt = lambda x: float(x) / 1024.0
    for m in matches:
        d = m.groupdict()
        boxes.append([fmt(d['y0']), fmt(d['x0']), fmt(d['y1']), fmt(d['x1'])])
        labels.append(d['label'])
    return np.array(boxes), np.array(labels)

def display_boxes(image, boxes, labels, target_image_size):
    h, w = target_image_size
    fig, ax = plt.subplots()
    ax.imshow(image)
    for i in range(boxes.shape[0]):
        y, x, y2, x2 = boxes[i] * np.array([h, w, h, w])
        width = x2 - x
        height = y2 - y
        rect = patches.Rectangle((x, y), width, height,
                                 linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x, y, labels[i], color='red', fontsize=12, backgroundcolor='white')
    plt.axis("off")
    plt.show()

def display_segment_output(image, bounding_box, segment_mask, target_image_size):
    full_mask = np.zeros(target_image_size, dtype=np.uint8)
    target_width, target_height = target_image_size
    for bbox, mask in zip(bounding_box, segment_mask):
        y1, x1, y2, x2 = bbox
        x1 = int(x1 * target_width)
        y1 = int(y1 * target_height)
        x2 = int(x2 * target_width)
        y2 = int(y2 * target_height)
        if not isinstance(mask, np.ndarray):
            mask = np.array(mask.tolist())
        if mask.ndim == 3:
            mask = mask.squeeze(axis=-1)
        mask = Image.fromarray(mask)
        mask = mask.resize((x2 - x1, y2 - y1), resample=Image.NEAREST)
        mask = np.array(mask)
        binary_mask = (mask > 0.5).astype(np.uint8)
        full_mask[y1:y2, x1:x2] = np.maximum(full_mask[y1:y2, x1:x2], binary_mask)
    cmap = plt.get_cmap('jet')
    colored_mask = cmap(full_mask / 1.0)
    colored_mask = (colored_mask[:, :, :3] * 255).astype(np.uint8)
    if isinstance(image, Image.Image):
        image = np.array(image)
    blended_image = image.copy()
    mask_indices = full_mask > 0
    alpha = 0.5
    for c in range(3):
        blended_image[:, :, c] = np.where(mask_indices,
                                          (1 - alpha) * image[:, :, c] + alpha * colored_mask[:, :, c],
                                          image[:, :, c])
    fig, ax = plt.subplots()
    ax.imshow(blended_image)
    plt.axis("off")
    plt.show()

# セグメンテーション用パース関数（公式コードに準ずる）
import big_vision.evaluators.proj.paligemma.transfers.segmentation as segeval
reconstruct_masks = segeval.get_reconstruct_masks('oi')

# def parse_segments(detokenized_output: str) -> tuple[np.ndarray, np.ndarray]:
#     pattern = (
#         r'<loc(?P<y0>\d{4})><loc(?P<x0>\d{4})><loc(?P<y1>\d{4})><loc(?P<x1>\d{4})>' +
#         ''.join(f'<seg(?P<s{i}>\d{{3}})>' for i in range(16))
#     )
#     matches = re.finditer(pattern, detokenized_output)
#     boxes, segs = [], []
#     fmt_box = lambda x: float(x) / 1024.0
#     for m in matches:
#         d = m.groupdict()
#         boxes.append([fmt_box(d['y0']), fmt_box(d['x0']),
#                       fmt_box(d['y1']), fmt_box(d['x1'])])
#         segs.append([int(d[f's{i}']) for i in range(16)])
#     boxes = np.array(boxes)
#     segs = np.array(segs)
#     seg_masks = reconstruct_masks(segs)
#     return boxes, seg_masks

# --- Transformers を用いた PaliGemma の利用例 ---

# モデルとプロセッサの初期化（使用するモデル: 3B, 224×224）
model_id = "google/paligemma2-3b-mix-224"
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"
# image_transformers = load_image(url)
# target_size = (224, 224)
# image_transformers = load_image("texi1.jpg")
# image_np = read_image(url, target_size)

# image_local = Image.open("phone1.jpg")
# target_size = (885, 590)
image_local = Image.open("driver.jpg")
target_size = (1536, 1024)

model = PaliGemmaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="cpu"
).eval()
processor = PaliGemmaProcessor.from_pretrained(model_id)

# # --- 1. "answer" タスク ---
# prompt = "answer en where is the car standing?\n"
# model_inputs = processor(text=prompt, images=image_transformers, return_tensors="pt")
# model_inputs = model_inputs.to(torch.bfloat16).to(model.device)
# input_len = model_inputs["input_ids"].shape[-1]
# with torch.inference_mode():
#     generation = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
#     generation = generation[0][input_len:]
#     decoded_answer = processor.decode(generation, skip_special_tokens=True)
# print("Answer Output:", decoded_answer)

# --- 2. "detect" タスク ---
prompt = "detect driver and handle\n"
print("Detect Prompt:", prompt)
model_inputs = processor(text=prompt, images=image_local, return_tensors="pt")
model_inputs = model_inputs.to(torch.bfloat16).to(model.device)
input_len = model_inputs["input_ids"].shape[-1]
with torch.inference_mode():
    generation = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
    generation = generation[0][input_len:]
    decoded_detect = processor.decode(generation, skip_special_tokens=True)
print("Detect Output:", decoded_detect)
boxes, labels = parse_bbox_and_labels(decoded_detect)
print("Parsed Boxes:", boxes)
print("Parsed Labels:", labels)
display_boxes(image_local, boxes, labels, target_image_size=target_size)

# # --- 2-1. "segment" タスク ---
# prompt = "segment car\n"
# print("Segment Prompt:", prompt)
# model_inputs = processor(text=prompt, images=image_transformers, return_tensors="pt")
# model_inputs = model_inputs.to(torch.bfloat16).to(model.device)
# input_len = model_inputs["input_ids"].shape[-1]
# with torch.inference_mode():
#     generation = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
#     generation = generation[0][input_len:]
#     decoded_segment = processor.decode(generation, skip_special_tokens=True)
# print("Segment Output:", decoded_segment)
# boxes_seg, seg_masks = parse_segments(decoded_segment)
# print("Parsed Boxes (Segment):", boxes_seg)
# display_segment_output(image_np, boxes_seg, seg_masks, target_image_size=target_size)

# # --- 3. バッチプロンプト ---
# prompts = [
#     'answer en where is the car standing?\n',
#     'answer en what color is the car?\n',
#     'describe ja\n',
#     'detect car\n',
# ]
# images = [image_transformers] * len(prompts)
# batch_inputs = processor(
#     text=prompts,
#     images=images,
#     return_tensors="pt",
#     padding=True,
#     truncation=True
# )
# batch_inputs = batch_inputs.to(torch.bfloat16).to(model.device)
# batch_outputs = model.generate(
#     **batch_inputs,
#     max_new_tokens=100,
#     do_sample=False
# )
# for i, output in enumerate(batch_outputs):
#     inp_len = processor(text=prompts[i], images=image_transformers, return_tensors="pt", padding=True, truncation=True)["input_ids"].shape[-1]
#     decoded = processor.decode(output[inp_len:], skip_special_tokens=True)
#     print(f"Batch Output {i+1}: {decoded}")
