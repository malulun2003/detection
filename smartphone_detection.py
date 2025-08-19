import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import os
import cv2
import movie2image

# Ensure the environment variable is set for offline mode
os.environ["TRANSFORMERS_OFFLINE"] = "1"
# Load the model and processor
model_id = "IDEA-Research/grounding-dino-base"
device = "cpu"
processor = AutoProcessor.from_pretrained("models--IDEA-Research--grounding-dino-base")
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

def cv2pil(image):
    ''' 
    OpenCV型 -> PIL型
    arg:
        image: OpenCV型画像
    return:
        new_image: PIL型画像
    '''
    new_image = image.copy()
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
    new_image = Image.fromarray(new_image)
    return new_image

def sf_detection(video_path, out_path, interval=1, driver=0):
    '''
    smartphone_detection
    arg:
        video_path: 
        out_path:
        driver:
        interval:
    '''
    print("Processing video frames...")
    cap = cv2.VideoCapture(video_path)
    count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)

    print("out path>", out_path)
    os.makedirs(out_path, exist_ok=True)
    prompt = ['person', 'cell phone']

    detections = []

    for num in range(1, int(count), int(fps * interval)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, num)
        ret, frame = cap.read()
        if not ret:
            break
        image = cv2pil(frame)
        width, height = image.size
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=0.4,
            text_threshold=0.4,
            target_sizes=[image.size[::-1]]
        )
        result = results[0]
        for box, score, labels in zip(result["boxes"], result["scores"], result["labels"]):
            box = [round(x, 2) for x in box.tolist()]
            print(f"Detected {labels} with confidence {round(score.item(), 3)} at location {box}")
            if labels == "cell phone":
                col = (0, 255, 0)
            elif labels == "person":
                col = (0, 0, 255)
            else:
                col = (128, 128, 128)
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), col, 2)
        cv2.imwrite(f"{out_path}/processed_{num}.png", frame)

        # スマフォ使用しているかどうか
        # personが画像の左側にいること（ドライバー席）
        # person boxの中に cell phone box が被っていること
        for box, score, labels in zip(result["boxes"], result["scores"], result["labels"]):
            box = [round(x, 2) for x in box.tolist()]
            if labels == "person":
                # 左側より(driver=0)　右より(driver=1)
                flag = False
                print(box[0]+(box[2]-box[0])/2, width/2)
                if (driver == 0 and box[0]+(box[2]-box[0])/2 < width/2) or (driver == 1 and box[0]+(box[2]-box[0])/2 > width/2):
                    print("this is driver")
                    for _box, _score, _labels in zip(result["boxes"], result["scores"], result["labels"]):
                        _box = [round(x, 2) for x in _box.tolist()]
                        if _labels == "cell phone":
                            # person boxに被っているか
                            if (max(box[0], _box[0]) < min(box[2], _box[2])) and (max(box[1], _box[1]) < min(box[3], _box[3])):
                                print("overlap!")
                                flag = True
                if flag:
                    path = f"{out_path}/using_processed_{num}.png"
                    cv2.imwrite(path, frame)
                    detections.append({"frame":num, "path":path})
    return detections

if __name__ == "__main__":
    sf_detection('aa2.mp4', 'output', interval=5, driver=1)
