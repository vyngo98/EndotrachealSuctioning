# Endotracheal Suctioning Activity Recognition

## Overview

This project focuses on activity recognition for Endotracheal Suctioning procedures using video-based pose estimation and machine learning.

The workflow consists of two main stages:

1. **Pose Estimation Extraction**

   * Extract human body keypoints from procedure videos using YOLOv7 Pose.
   * Save extracted keypoints as CSV files.
   * Generate visualization videos with detected skeleton keypoints.

2. **Activity Recognition**

   * Train an activity recognition model using the extracted pose keypoints.
   * Evaluate the trained model on labeled Endotracheal Suctioning activities.

![Video demo](videos/ES-demo.gif)
---

## Project Structure

```text
.
├── extract_pose_estimation.py
├── activity_recognition.py
├── yolo_utils.py
├── yolov7/
├── Endotracheal_Suctioning_Project.pdf
└── README.md
```

---

## Requirements

### Python

* Python 3.8+
* PyTorch
* OpenCV
* NumPy
* Pandas
* Scikit-learn

### YOLOv7 Pose

Clone YOLOv7 and download the pose estimation weights:

```bash
git clone https://github.com/WongKinYiu/yolov7
```

Download:

```text
yolov7-w6-pose.pt
```

and place it inside:

```text
yolov7/
```

---

# Step 1: Pose Estimation Extraction

Run:

```bash
python extract_pose_estimation.py
```

Before running the script, update the following variables:

### Input Video Folder

```python
FOLDER = "path_to_video_folder"
```

This folder should contain Endotracheal Suctioning videos.

Example:

```text
dataset/
├── Back_S01.mp4
├── Back_S02.mp4
└── ...
```

### Output CSV Folder

```python
CSV_FOLDER = "path_to_save_keypoints"
```

The extracted body keypoints will be saved as CSV files.

### Output Video Folder

```python
POSE_FOLDER = "path_to_save_pose_videos"
```

Videos with skeleton visualization will be saved here.

---

## Output

### Keypoint CSV

```text
keypoints/
├── Back_S01.csv
├── Back_S02.csv
└── ...
```

### Pose Visualization Videos

```text
pose_videos/
├── Back_S01_pose.mp4
├── Back_S02_pose.mp4
└── ...
```

---

# Step 2: Activity Recognition

Run:

```bash
python activity_recognition.py
```

Before training, modify the following paths:

### Keypoint Folder

```python
FOLDER_PATH = "path_to_keypoint_csv"
```

Folder containing pose estimation CSV files generated in Step 1.

### Ground Truth Annotation Folder

```python
ANN_PATH = "path_to_annotation_csv"
```

Folder containing activity labels and annotations.

### Model Save Path

```python
SAVE_MODEL_PATH = "path_to_save_model"
```

Location where the trained activity recognition model will be stored.

---

## Training Pipeline

1. Load extracted pose keypoints.
2. Load activity annotations.
3. Generate feature representations.
4. Train activity recognition model.
5. Evaluate model performance.
6. Save trained model.

---

## Dataset Format

### Keypoint CSV

Generated automatically from:

```text
extract_pose_estimation.py
```

### Annotation CSV

Example:

```csv
start_time,end_time,activity
0,10,Preparation
10,25,Suctioning
25,35,Cleaning
```

---

## Results

The trained model will be saved to:

```text
SAVE_MODEL_PATH
```

and can later be used for inference on new Endotracheal Suctioning videos.

---

## Additional Documentation

A project presentation is included in PDF format:

```text
Endotracheal_Suctioning_Project.pdf
```

The slides provide:

* Project motivation
* Dataset overview
* Pose estimation pipeline
* Activity recognition methodology
* Experimental results
* Future work

---

## Citation

If you use this project for research purposes, please cite the corresponding publication or thesis related to the Endotracheal Suctioning Activity Recognition project.
Hoang Anh Vy Ngo, Noriyo Colley, Shinji Ninomiya, Satoshi Kanai,
Shunsuke Komizunai, Atsushi Konno, Misuzu Nakamura, Sozo Inoue, Nurses' Skill Assessment in
Endotracheal Suctioning Using Video-based Activity Recognition, International Journal of Activity and
Behavior Computing, 2024, Volume 2024, Issue 2, Pages 1-24, Released on J-STAGE June 13, 2024,
Online ISSN 2759-2871, https://doi.org/10.60401/ijabc.20