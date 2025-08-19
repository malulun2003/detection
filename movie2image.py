import cv2
import os

def save_all_frames(video_path, dir_path, basename, ext='png'):
    os.makedirs(dir_path, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    # print("width:{}, height:{}, count:{}, fps:{}".format(width,height,count,fps))
    for num in range(1, int(count), int(fps)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, num)
        cv2.imwrite(dir_path+"/"+basename+"{:0=3}".format(int((num-1)/int(fps)))+"."+ext, cap.read()[1])
        print(dir_path+"/"+basename+"{:0=3}".format(int((num-1)/int(fps)))+"."+ext)
    cap.release()
