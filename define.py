KEYPOINTS = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
             "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
             "right_knee", "left_ankle", "right_ankle"]

FOLDER = "/content/drive/MyDrive/STUDY MASTER/PROJECT/Hokkaido prj/Data/New Video/ESTE-SIM video/" # folder contains videos
CSV_FOLDER = "/content/drive/MyDrive/STUDY MASTER/PROJECT/Hokkaido prj/Data/New CSV/" # folder contains Ann csv files
POSE_FOLDER = "/content/drive/MyDrive/STUDY MASTER/PROJECT/Hokkaido prj/Data/New Video/Pose Video/" # folder to save pose video (optional)

DETECTION_IMAGE_SIZE = 1920
POSE_IMAGE_SIZE = 960
STRIDE = 64
CONFIDENCE_TRESHOLD = 0.25
IOU_TRESHOLD = 0.65

TRAIN_ID = [("N01T1"), ("N01T2"),
           ("N04T1"), ("N04T2"),
           ("N05T1"), ("N05T2"),
           ("N06T1"), ("N06T2"),
           ("N07T1"), ("N07T2"),

           ("S02T1"),
           ("S03T1"),
           ("S06T1"),
           ("S07T1"),
           ("S08T1"),
           ]

TEST_ID = ["N02T1",
           "N03T1",
           "S01T1",
           "S04T1",
           "S05T1"
           ]

ACTIVITY_DICT = {"Explanation to patient": 0,
                  "Auscultation": 1,
                  "Washing hands": 2,
                  'Checking cuff pressure': 3,
                  "Wearing apron": 4,
                  "Wearing goggles": 5,
                  "Wearing gloves": 6,
                  "Opening lids": 7,
                  'Preparing alcohol tissue': 8,
                  "Connecting suctioning catheter": 9,
                  "Pressing button": 10,
                  "Removal of airway": 11,
                  "Suctioning": 12,
                  "Refitting the airway": 13,
                  "Wiping catheter": 14,
                  'Washing catheter': 15,
                  'Disconnecting catheter': 16,
                  'Closing lids': 17,
                  "Putting back catheter": 18,
                  "Removal of gloves": 19,
                  "Removal of goggles": 20,
                  "Removal of apron": 21,
                  "Others": 22}

VERSION = "v3.0"
WINDOW_SIZE = 2 # seconds
OVERLAP_RATE = 0.5 * WINDOW_SIZE
FS = 60
SMOOTH_LEN = 3 # seconds
EXTENSION = "keypoint12"

TOTAL_CLASSESS = 23
N_SKIP_FRAMES = 10
LEARNING_RATE = 0.0001

USER_ID = ["N01T1", "N01T2",
           "N02T1",
           "N03T1",
           "N04T1", "N04T2",
           "N05T1", "N05T2",
           "N06T1", "N06T2",
           "N07T1", "N07T2",

           "S01T1",
           "S02T1",
           "S03T1",
           "S04T1",
           "S05T1",
           "S06T1",
           "S07T1",
           "S08T1",
           ]

