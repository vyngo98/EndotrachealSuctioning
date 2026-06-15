import torch
from yolo_utils import *
import sys
import glob
import os

HOME = os.getcwd()
print(HOME)

# !git clone https://github.com/WongKinYiu/yolov7
# cd {HOME}/yolov7
# !pip install -r requirements.txt


sys.path.append(f"{HOME}/yolov7")


# Load YOLOv7
# %cd {HOME}/yolov7
# !wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt --quiet

POSE_MODEL_WEIGHTS_PATH = f"{HOME}/yolov7/yolov7-w6-pose.pt"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    weigths = torch.load(POSE_MODEL_WEIGHTS_PATH, map_location=device)
    pose_model = weigths["model"]
    _ = pose_model.float().eval()

    if torch.cuda.is_available():
        pose_model.half().to(device)

    list_files = glob.glob(FOLDER + '/Back_S*.MP4')
    list_files.sort()
    for file_name in list_files:
        print("Processing {} file".format(file_name))
        pose_estimation(file_name, pose_model, device)

main()