from typing import Generator
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
from yolo_utils.datasets import letterbox
from torchvision import transforms
import os
import time
from yolo_utils.general import non_max_suppression_kpt, non_max_suppression
from yolo_utils.plots import output_to_keypoint
from define import *


def generate_frames(video) -> Generator[np.ndarray, None, None]:
    # video = cv2.VideoCapture(video_file)

    while video.isOpened():
        success, frame = video.read()

        if not success:
            break

        yield frame

    video.release()


def plot_image(image: np.ndarray, size: int = 12) -> None:
    plt.figure(figsize=(size, size))
    plt.axis('off')
    plt.imshow(image[...,::-1])
    plt.show()

from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass(frozen=True)
class Point:
    x: float
    y: float

    @property
    def int_xy_tuple(self) -> Tuple[int, int]:
        return int(self.x), int(self.y)


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    width: float
    height: float

    @property
    def top_left(self) -> Point:
        return Point(x=self.x, y=self.y)

    @property
    def bottom_right(self) -> Point:
        return Point(x=self.x + self.width, y=self.y + self.height)

    @property
    def bottom_center(self) -> Point:
        return Point(x=self.x + self.width / 2, y=self.y + self.height)

@dataclass
class Detection:
    rect: Rect
    class_id: int
    confidence: float
    tracker_id: Optional[int] = None


@dataclass(frozen=True)
class Color:
    r: int
    g: int
    b: int

    @property
    def bgr_tuple(self) -> Tuple[int, int, int]:
        return self.b, self.g, self.r


COLOR = Color(r=255, g=255, b=255)

def pose_pre_process_frame(frame: np.ndarray, device: torch.device) -> torch.Tensor:
    image = letterbox(frame, POSE_IMAGE_SIZE, stride=STRIDE, auto=True)[0]
    image = transforms.ToTensor()(image)
    image = torch.tensor(np.array([image.numpy()]))

    if torch.cuda.is_available():
        image = image.half().to(device)

    return image

def detection_pre_process_frame(frame: np.ndarray, device: torch.device) -> torch.Tensor:
    img = letterbox(frame, DETECTION_IMAGE_SIZE, STRIDE, auto=True)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device).float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    return img


from dataclasses import dataclass

"""
usage example:

video_config = VideoConfig(
    fps=30,
    width=1920,
    height=1080)
video_writer = get_video_writer(
    target_video_path=TARGET_VIDEO_PATH,
    video_config=video_config)

for frame in frames:
    ...
    video_writer.write(frame)

video_writer.release()
"""


# stores information about output video file, width and height of the frame must be equal to input video
@dataclass(frozen=True)
class VideoConfig:
    fps: float
    width: int
    height: int


# create cv2.VideoWriter object that we can use to save output video
def get_video_writer(target_video_path: str, video_config: VideoConfig) -> cv2.VideoWriter:
    video_target_dir = os.path.dirname(os.path.abspath(target_video_path))
    os.makedirs(video_target_dir, exist_ok=True)
    return cv2.VideoWriter(
        target_video_path,
        fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
        fps=video_config.fps,
        frameSize=(video_config.width, video_config.height),
        isColor=True
    )



def draw_rect(image: np.ndarray, rect: Rect, color: Color, thickness: int = 2) -> np.ndarray:
    cv2.rectangle(image, rect.top_left.int_xy_tuple, rect.bottom_right.int_xy_tuple, color.bgr_tuple, thickness)
    return image

def clip_coords(boxes: np.ndarray, img_shape: Tuple[int, int]):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0] = np.clip(boxes[:, 0], 0, img_shape[1]) # x1
    boxes[:, 1] = np.clip(boxes[:, 1], 0, img_shape[0]) # y1
    boxes[:, 2] = np.clip(boxes[:, 2], 0, img_shape[1]) # x2
    boxes[:, 3] = np.clip(boxes[:, 3], 0, img_shape[0]) # y2


def detection_post_process_output(
    output: torch.tensor,
    confidence_trashold: float,
    iou_trashold: float,
    image_size: Tuple[int, int],
    scaled_image_size: Tuple[int, int]
) -> np.ndarray:
    output = non_max_suppression(
        prediction=output,
        conf_thres=confidence_trashold,
        iou_thres=iou_trashold
    )
    print(output)
    coords = output[0].detach().cpu().numpy()

    v_gain = scaled_image_size[0] / image_size[0]
    h_gain = scaled_image_size[1] / image_size[1]

    coords[:, 0] /= h_gain
    coords[:, 1] /= v_gain
    coords[:, 2] /= h_gain
    coords[:, 3] /= v_gain

    clip_coords(coords, image_size)
    return


def post_process_pose(pose: np.ndarray, image_size: Tuple, scaled_image_size: Tuple) -> np.ndarray:
    height, width = image_size
    scaled_height, scaled_width = scaled_image_size
    vertical_factor = height / scaled_height
    horizontal_factor = width / scaled_width
    result = pose.copy()
    for i in range(17):
        result[i * 3] = horizontal_factor * result[i * 3]
        result[i * 3 + 1] = vertical_factor * result[i * 3 + 1]
    return result


def pose_post_process_output(
    pose_model,
    output: torch.tensor,
    confidence_trashold: float,
    iou_trashold: float,
    image_size: Tuple[int, int],
    scaled_image_size: Tuple[int, int]
) -> np.ndarray:
    output = non_max_suppression_kpt(
        prediction=output,
        conf_thres=confidence_trashold,
        iou_thres=iou_trashold,
        nc=pose_model.yaml['nc'],
        nkpt=pose_model.yaml['nkpt'],
        kpt_label=True)

    with torch.no_grad():
        output = output_to_keypoint(output)

        for idx in range(output.shape[0]):
            output[idx, 7:] = post_process_pose(
                output[idx, 7:],
                image_size=image_size,
                scaled_image_size=scaled_image_size
            )

    return output

def validate_object(kpts, steps=3, thr=0.5):
    num_kpts = len(kpts) // steps
    conf = kpts[2::steps]
    x = kpts[::steps]
    y = kpts[1::steps]
    valid_keypoint = np.flatnonzero(conf >= thr)

def plot_skeleton_kpts(im, kpts, steps, orig_shape=None):
    #Plot the skeleton and keypointsfor coco datatset
    palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                        [230, 230, 0], [255, 153, 255], [153, 204, 255],
                        [255, 102, 255], [255, 51, 255], [102, 178, 255],
                        [51, 153, 255], [255, 153, 153], [255, 102, 102],
                        [255, 51, 51], [153, 255, 153], [102, 255, 102],
                        [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                        [255, 255, 255]])

    skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

    pose_limb_color = palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
    pose_kpt_color = palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
    radius = 5
    num_kpts = len(kpts) // steps

    for kid in range(num_kpts):
        r, g, b = pose_kpt_color[kid]
        x_coord, y_coord = kpts[steps * kid], kpts[steps * kid + 1]
        if not (x_coord % 640 == 0 or y_coord % 640 == 0):
            if steps == 3:
                conf = kpts[steps * kid + 2]
                if conf < 0.5:
                    continue
            cv2.circle(im, (int(x_coord), int(y_coord)), radius, (int(r), int(g), int(b)), -1)

    for sk_id, sk in enumerate(skeleton):
        r, g, b = pose_limb_color[sk_id]
        pos1 = (int(kpts[(sk[0]-1)*steps]), int(kpts[(sk[0]-1)*steps+1]))
        pos2 = (int(kpts[(sk[1]-1)*steps]), int(kpts[(sk[1]-1)*steps+1]))
        if steps == 3:
            conf1 = kpts[(sk[0]-1)*steps+2]
            conf2 = kpts[(sk[1]-1)*steps+2]
            if conf1<0.5 or conf2<0.5:
                continue
        if pos1[0]%640 == 0 or pos1[1]%640==0 or pos1[0]<0 or pos1[1]<0:
            continue
        if pos2[0] % 640 == 0 or pos2[1] % 640 == 0 or pos2[0]<0 or pos2[1]<0:
            continue
        cv2.line(im, pos1, pos2, (int(r), int(g), int(b)), thickness=2)


def calc_plot_status(kpts, steps):
    num_kpts = len(kpts) // steps
    plot_status = [1]*num_kpts

    for kid in range(num_kpts):
        x_coord, y_coord = kpts[steps * kid], kpts[steps * kid + 1]
        if not (x_coord % 640 == 0 or y_coord % 640 == 0):
            if steps == 3:
                conf = kpts[steps * kid + 2]
                if conf < 0.5:
                    kpts[steps * kid] = 0
                    kpts[steps * kid + 1] = 0
                    plot_status[kid] = 0
                    continue

    return plot_status, kpts

def calc_trunk_length(kpts, steps=3): # general 10
    # skeleton = [[12, 13], [6, 12],
    #             [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11]]
    skeleton = [[6, 7], [6, 12], [12, 13], [7, 13], [7, 9], [9, 11], [6, 8], [8, 10]]
    trunk_length = []
    for sk_id, sk in enumerate(skeleton):
        pos1 = (int(kpts[(sk[0]-1)*steps]), int(kpts[(sk[0]-1)*steps+1]))
        pos2 = (int(kpts[(sk[1]-1)*steps]), int(kpts[(sk[1]-1)*steps+1]))
        if steps == 3:
            conf1 = kpts[(sk[0]-1)*steps+2]
            conf2 = kpts[(sk[1]-1)*steps+2]
            if conf1<0.5 or conf2<0.5:
                trunk_length.append(0)
                continue
        if pos1[0]%640 == 0 or pos1[1]%640==0 or pos1[0]<0 or pos1[1]<0 or pos2[0] % 640 == 0 or pos2[1] % 640 == 0 or pos2[0]<0 or pos2[1]<0:
            trunk_length.append(0)
            continue

        skeleton_length = np.sqrt(abs(pos2[0] - pos1[0])**2 + abs(pos2[1] - pos1[1])**2)
        trunk_length.append(skeleton_length)
    if trunk_length[4] == 0 and trunk_length[6] > 0:
        trunk_length[4] = trunk_length[6]
    elif trunk_length[6] == 0 and trunk_length[4] > 0:
        trunk_length[6] = trunk_length[4]
    if trunk_length[5] == 0 and trunk_length[7] > 0:
        trunk_length[5] = trunk_length[7]
    elif trunk_length[7] == 0 and trunk_length[5] > 0:
        trunk_length[7] = trunk_length[5]
    total_length = sum(trunk_length)
    return total_length


def calc_diff_distance(previous_pose, pose, steps=3): # general 10
    num_kpts = len(pose) // steps
    sum_pose, sum_pose_y = 0, 0
    sum_pre_pose, sum_pre_pose_y = 0, 0
    number_pose = 0
    number_pre_pose = 0
    for kid in range(17):
        pose_x, pose_y = pose[steps * kid], pose[steps * kid + 1]
        pre_pose_x, pre_pose_y = previous_pose[steps * kid], previous_pose[steps * kid + 1]
        if pose_x > 0:
            sum_pose += pose_x
            sum_pose_y += pose_y
            number_pose += 1
        if pre_pose_x > 0:
            sum_pre_pose += pre_pose_x
            sum_pre_pose_y += pre_pose_y
            number_pre_pose += 1

    diff_distance_x, diff_distance_y = 5000, 5000
    if number_pose > 0 and number_pre_pose > 0:
        cg_pose_x = sum_pose / number_pose
        cg_pre_pose_x = sum_pre_pose / number_pre_pose
        diff_distance_x = abs(cg_pose_x - cg_pre_pose_x)
        cg_pose_y = sum_pose_y / number_pose
        cg_pre_pose_y = sum_pre_pose_y / number_pre_pose
        diff_distance_y = abs(cg_pose_y - cg_pre_pose_y)

    return diff_distance_x, diff_distance_y

def pose_annotate(image: np.ndarray, detections: np.ndarray, previous_detection: np.ndarray) -> np.ndarray: # general 11
    annotated_frame_final = image.copy()
    height, width = annotated_frame_final.shape[:2]
    plot_status_list = [1]*int(detections.shape[0])
    trunk_length_list = []
    for idx in range(detections.shape[0]):
        pose = detections[idx, 7:].T
        plot_kinect_status, pose = calc_plot_status(pose, 3)
        x = pose[::3]
        if len(np.flatnonzero(plot_kinect_status)) == 0:
            plot_status_list[idx] = 0
        trunk_length_list.append(calc_trunk_length(pose, steps=3))
    if len(np.flatnonzero(plot_status_list)) == 0:
        pose = np.array([0]*51)
        # print("There are no detected subject!")
    elif len(np.flatnonzero(plot_status_list)) == 1:
        if sum(previous_detection) == 0:
            pose = detections[int(np.flatnonzero(plot_status_list)), 7:]
            previous_detection = pose.copy()
        else:
            total_diff_distance_x, total_diff_distance_y = calc_diff_distance(previous_detection, detections[int(np.flatnonzero(plot_status_list)), 7:])
            if total_diff_distance_x < 300 and total_diff_distance_y < 180:
                pose = detections[int(np.flatnonzero(plot_status_list)), 7:]
                previous_detection = pose.copy()
            else:
                pose = np.array([0]*51)
    else:
        trunk_length_list = np.array(trunk_length_list)
        choose_id = np.flatnonzero(plot_status_list)
        if sum(previous_detection) == 0:
            choose_trunk_length_list = trunk_length_list[choose_id]
            choose_trunk_length_id = np.argmax(choose_trunk_length_list)
            pose = detections[int(choose_id[choose_trunk_length_id]), 7:]
            previous_detection = pose.copy()
        else:
            total_diff_distance = [calc_diff_distance(previous_detection, detections[i, 7:]) for i in choose_id]
            total_diff_distance_x = np.asarray(total_diff_distance)[:, 0]
            total_diff_distance_y = np.asarray(total_diff_distance)[:, 1]
            if np.min(total_diff_distance_x) > 680:
                pose = np.array([0]*51)
            else:
                choose_diff_distance_id = np.flatnonzero((total_diff_distance_x < 680) &  (total_diff_distance_y < 190))

                if len(choose_diff_distance_id) > 1:
                    if any(detections[choose_id[choose_diff_distance_id], 7:][:, 3*5] == 0) or any(detections[choose_id[choose_diff_distance_id], 7:][:, 3*6] == 0) or previous_detection[3*5] == 0 or previous_detection[3*6] == 0:
                        choose_diff_distance_id = np.argmin(total_diff_distance_x + total_diff_distance_y)
                        pose = detections[int(choose_id[choose_diff_distance_id]), 7:]
                        previous_detection = pose.copy()
                    else:
                        shoulder_diff_dist = []
                        for i in range(len(choose_diff_distance_id)):
                            shoulders_detection = sorted([[detections[int(choose_id[choose_diff_distance_id[i]]), 7:][3*5], detections[int(choose_id[choose_diff_distance_id[i]]), 7:][3*5+1]], [detections[int(choose_id[choose_diff_distance_id[i]]), 7:][3*6], detections[int(choose_id[choose_diff_distance_id[i]]), 7:][3*6+1]]])
                            shoulders_previous = sorted([[previous_detection[3*5], previous_detection[3*5+1]], [previous_detection[3*6], previous_detection[3*6 + 1]]])
                            right_shoulder_diff = np.sqrt(abs(shoulders_previous[0][0]-shoulders_detection[0][0])**2 + abs(shoulders_previous[0][1]-shoulders_detection[0][1])**2)
                            left_shoulder_diff = np.sqrt(abs(shoulders_previous[1][0]-shoulders_detection[1][0])**2 + abs(shoulders_previous[1][1]-shoulders_detection[1][1])**2)
                            sum_shoulder_diff = left_shoulder_diff + right_shoulder_diff
                            shoulder_diff_dist.append(sum_shoulder_diff)
                        # if np.min(np.array(shoulder_diff_dist)) < 100:
                        if np.min(np.array(shoulder_diff_dist)) < 120:
                            choose_diff_distance_id = int(choose_diff_distance_id[np.argmin(np.array(shoulder_diff_dist))])
                            pose = detections[int(choose_id[choose_diff_distance_id]), 7:]
                            previous_detection = pose.copy()

                        else:
                            pose = np.array([0]*51)
                elif len(choose_diff_distance_id) == 0:
                    choose_diff_distance_id = np.argmin(total_diff_distance_x + total_diff_distance_y)
                    pose = detections[int(choose_id[choose_diff_distance_id]), 7:]
                    previous_detection = pose.copy()

                else:
                    # choose_diff_distance_id = np.argmin(total_diff_distance_x + total_diff_distance_y)
                    pose = detections[int(choose_id[choose_diff_distance_id]), 7:]
                    previous_detection = pose.copy()
    plot_skeleton_kpts(annotated_frame_final, pose.T, 3)

    return annotated_frame_final, pose, previous_detection

def pose_estimation(video_path, pose_model, device):
    csv_path = CSV_FOLDER + "/" + os.path.basename(video_path)[:-4] + "_keypoint12.csv"
    # save_video_path = video_path[:-4] + "_yolo5.mp4"
    save_video_path = POSE_FOLDER + "/" + os.path.basename(video_path)[:-4] + "_yolo12.mp4"
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # Find OpenCV version
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

    if int(major_ver)  < 3 :
        fps_ = cap.get(cv2.cv.CV_CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps_))
    else :
        fps_ = cap.get(cv2.CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps_))

    # initiate video writer
    video_config = VideoConfig(
        fps=int(np.ceil(fps_)),
        width=frame_width,
        height=frame_height)

    video_writer = get_video_writer(
        target_video_path=save_video_path,
        video_config=video_config)
    print(save_video_path)

    # get fresh video frame generator
    frame_iterator = iter(generate_frames(cap))

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    alldata = []
    previous_pose = np.array([0]*51)
    start_time = time.time()
    for frame in tqdm(frame_iterator, total=total):
      annotated_frame = frame.copy()

      with torch.no_grad():
          image_size = frame.shape[:2]

          # pose
          pose_pre_processed_frame = pose_pre_process_frame(
              frame=frame,
              device=device)
          pose_scaled_image_size = tuple(pose_pre_processed_frame.size())[2:]

          pose_output = pose_model(pose_pre_processed_frame)[0].detach().cpu()
          pose_output = pose_post_process_output(
              pose_model,
              output=pose_output,
              confidence_trashold=CONFIDENCE_TRESHOLD,
              iou_trashold=IOU_TRESHOLD,
              image_size=image_size,
              scaled_image_size=pose_scaled_image_size
          )

          annotated_frame, pose, previous_pose = pose_annotate(
              image=annotated_frame, detections=pose_output, previous_detection=previous_pose)
          data_frame = {}
          for i in range(len(KEYPOINTS)):
            data_frame.update({(KEYPOINTS[i] + "_x") : pose[i*3]})
            data_frame.update({(KEYPOINTS[i] + "_y") : pose[i*3 + 1]})
            data_frame.update({(KEYPOINTS[i] + "_conf") : pose[i*3 + 2]})
          alldata.append(data_frame)

          # save video frame
          video_writer.write(annotated_frame)

    # close output video
    video_writer.release()
    df = pd.DataFrame(alldata)
    df.to_csv(csv_path)
    print("Pose estimation {} files successfully!".format(os.path.basename(video_path)))
    print("Process time: ", time.time() - start_time, "\n")
