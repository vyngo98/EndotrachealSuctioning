import numpy as np
import pandas as pd
from copy import deepcopy
import scipy
from define import *

def remove_redundant_kp(kp_df):
  kp_df = kp_df.loc[:, ~kp_df.columns.str.contains(
      'conf|left_knee_x|left_knee_y|left_knee_conf|right_knee_x|right_knee_y|right_knee_conf|left_ankle_x|left_ankle_y|left_ankle_conf|right_ankle_x|right_ankle_y|right_ankle_conf', regex=True)]
  return kp_df

def convert_ann_to_int(ann_df):
  # Convert annotation to interger
  # Explanation to patient: 0
  # Auscultation: 1
  # Washing hands: 2
  # Checking cuff pressure: 3
  # Wearing apron: 4
  # Wearing goggles: 5
  # Wearing gloves: 6
  # Opening lids: 7
  # Preparing alcohol tissue: 8
  # Connecting suctioning catheter: 9
  # pressing button = touching a screen of ventilator: 10
  # Removal of airway: 11
  # Suctioning: 12
  # Refitting the airway: 13
  # Wiping cather: 14
  # Washing catheter: 15
  # Disconnecting catheter: 16
  # Closing lids: close lids: 17
  # Putting back catheter: turn off the machine, put back the catheter: 18
  # Removal of gloves: 19
  # Removal of goggles: 20
  # Removal of apron: 21
  # Others: 22

  ann_df['annotation'].iloc[np.flatnonzero(ann_df['annotation'].str.lower() == "explanation to patient")] = 0
  ann_df['annotation'].iloc[np.flatnonzero(ann_df['annotation'].str.lower() == "auscultation")] = 1
  ann_df['annotation'].iloc[np.flatnonzero(ann_df['annotation'].str.lower() == "washing hands")] = 2
  ann_df['annotation'].iloc[np.flatnonzero(ann_df['annotation'].str.lower() == "checking cuff pressure")] = 3
  ann_df['annotation'].iloc[np.flatnonzero(ann_df['annotation'].str.lower() == "wearing apron")] = 4
  ann_df['annotation'].iloc[np.flatnonzero(ann_df['annotation'].str.lower() == "wearing goggles")] = 5
  ann_df['annotation'].iloc[np.flatnonzero(ann_df['annotation'].str.lower() == "wearing gloves")] = 6
  ann_df['annotation'].iloc[np.flatnonzero((ann_df['annotation'].str.lower() == "opening lids")|(ann_df['annotation'].str.lower() == "open lids"))] = 7
  ann_df['annotation'].iloc[np.flatnonzero(ann_df['annotation'].str.lower() == "preparing alcohol tissue")] = 8
  ann_df['annotation'].iloc[np.flatnonzero((ann_df['annotation'].str.lower() == "connecting suctioning catheter")|(ann_df['annotation'].str.lower() == "connecting catheter"))] = 9
  ann_df['annotation'].iloc[np.flatnonzero(ann_df['annotation'].str.lower() == "pressing button")] = 10
  ann_df['annotation'].iloc[np.flatnonzero(ann_df['annotation'].str.lower() == "removal of airway")] = 11
  ann_df['annotation'].iloc[np.flatnonzero(ann_df['annotation'].str.lower() == "suctioning")] = 12
  ann_df['annotation'].iloc[np.flatnonzero((ann_df['annotation'].str.lower() == "refitting the airway")|(ann_df['annotation'].str.lower() == "refitting airway"))] = 13
  ann_df['annotation'].iloc[np.flatnonzero(ann_df['annotation'].str.lower() == "wiping catheter")] = 14
  ann_df['annotation'].iloc[np.flatnonzero((ann_df['annotation'].str.lower() == "washing catheter")|(ann_df['annotation'].str.lower() == "wash catheter"))] = 15
  ann_df['annotation'].iloc[np.flatnonzero(ann_df['annotation'].str.lower() == "disconnecting catheter")] = 16
  ann_df['annotation'].iloc[np.flatnonzero(ann_df['annotation'].str.lower() == "closing lids")] = 17
  ann_df['annotation'].iloc[np.flatnonzero((ann_df['annotation'].str.lower() == "putting back catheter")|(ann_df['annotation'].str.lower() == "putting back catheter removal"))] = 18
  ann_df['annotation'].iloc[np.flatnonzero(ann_df['annotation'].str.lower() == "removal of gloves")] = 19
  ann_df['annotation'].iloc[np.flatnonzero(ann_df['annotation'].str.lower() == "removal of goggles")] = 20
  ann_df['annotation'].iloc[np.flatnonzero(ann_df['annotation'].str.lower() == "removal of apron")] = 21
  ann_df['annotation'].iloc[np.flatnonzero(ann_df['annotation'].str.lower() == "others")] = 22

  return ann_df


def load_data(keypoint_csv, ann_csv):
  kp_df = pd.read_csv(keypoint_csv)
  kp_df = kp_df.loc[:, ~kp_df.columns.str.contains('^Unnamed')]
  kp_df = remove_redundant_kp(kp_df)

  ann_df = pd.read_csv(ann_csv)
  ann_df = ann_df.loc[:, ~ann_df.columns.str.contains('^Unnamed')]
  ann_df['annotation_str'] = ann_df['annotation'].copy()
  ann_df = convert_ann_to_int(ann_df)

  return kp_df, ann_df


def smooth_kp(kp_col):
  zero_idx = np.flatnonzero(kp_col == 0)
  split_idx = np.split(zero_idx, np.flatnonzero(np.diff(zero_idx) > 1) + 1)
  for each_split_idx in split_idx:
    if len(each_split_idx) == 0 or each_split_idx[0] == 0 or each_split_idx[-1] == (len(kp_col) - 1) or len(each_split_idx) > SMOOTH_LEN*FS:
      continue
    xp = [each_split_idx[0] - 1, each_split_idx[-1] + 1]
    fp = kp_col[xp]
    interp_kp = np.interp(each_split_idx, xp, fp)
    kp_col[each_split_idx] = interp_kp
  return kp_col

def segment(data, max_time, sub_window_size, stride_size):
    sub_windows = np.arange(sub_window_size)[None, :] + np.arange(0, max_time, stride_size)[:, None]

    row, col = np.where(sub_windows >= max_time)
    uniq_row = len(np.unique(row))

    if uniq_row > 0 and row[0] > 0:
        sub_windows = sub_windows[:-uniq_row, :]

    return data[sub_windows]

def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[result.size // 2:][:10]

def cal_shannon_entropy(x):
    pd_series = pd.Series(x)
    counts = pd_series.value_counts()
    entropy = scipy.stats.entropy(counts)

    return entropy

def cal_dominant_freq_ratio(data, fs):
    fourier = np.fft.fft(data)
    frequencies = np.fft.fftfreq(np.shape(data)[0], d=1/fs)
    magnitudes = abs(fourier[np.where(frequencies >= 0)])

    peak_magnitude = np.max(magnitudes, axis=0)
    dominant_freq_ratio = peak_magnitude / sum(magnitudes)

    energy = sum(magnitudes**2)
    return dominant_freq_ratio, energy

def extract_feature(data, fs):
    mean_ft = np.mean(data, axis=0)
    std_ft = np.std(data, axis=0)
    max_ft = np.max(data, axis=0)
    min_ft = np.min(data, axis=0)
    var_ft = np.var(data, axis=0)
    med_ft = np.median(data, axis=0)
    sum_ft = np.sum(data, axis=0)
    skew = scipy.stats.skew(data, axis=0)
    kurtosis = scipy.stats.kurtosis(data, axis=0)
    q25 = np.percentile(data, 25, axis=0)
    q75 = np.percentile(data, 75, axis=0)
    iqr = q75 - q25
    rms = np.sqrt(np.mean(data**2, axis=0))
    dominant_freq_ratio, energy = cal_dominant_freq_ratio(data, fs)
    autocorrelation = np.array([autocorr(data[:, x]) for x in range(np.shape(data)[1])]).T
    shannon_entropy = np.array([cal_shannon_entropy(data[:, x]) for x in range(np.shape(data)[1])])
    features = np.array([mean_ft, std_ft, max_ft, min_ft, var_ft, med_ft, sum_ft, skew, kurtosis, q25, q75, iqr, shannon_entropy, rms, dominant_freq_ratio, energy])
    features = np.concatenate([features, autocorrelation], axis=0).T.flatten()
    features = np.nan_to_num(features)
    return features

def cal_angle(a, b, c):
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return angle

def nan_helper(y):
        return np.isnan(y), lambda z: z.nonzero()[0]

def itpl_nan( y):
    nans, x = nan_helper(y)
    y[nans] = np.interp(x(nans), x(~nans), y[~nans])
    return y

def extract_joint_angles(kp_data, steps=2):
    # steps = 2 if kp_data is removed conf columns
    # steps = 3 if kp_data has conf columns
    left_elbow_shoulder_hip = np.asarray([cal_angle(kp_data[i, 7*steps:(7*steps+2)], kp_data[i, 5*steps:(5*steps+2)], kp_data[i, 11*steps:(11*steps+2)])
                                          for i in range(len(kp_data))])
    left_elbow_shoulder_hip = np.nan_to_num(left_elbow_shoulder_hip)
    right_elbow_shoulder_hip = np.asarray([cal_angle(kp_data[i, 8*steps:(8*steps+2)], kp_data[i, 6*steps:(6*steps+2)], kp_data[i, 12*steps:(12*steps+2)])
                                            for i in range(len(kp_data))])
    right_elbow_shoulder_hip = np.nan_to_num(right_elbow_shoulder_hip)
    left_wrist_elbow_shoulder = np.asarray([cal_angle(kp_data[i, 9*steps:(9*steps+2)], kp_data[i, 7*steps:(7*steps+2)], kp_data[i, 5*steps:(5*steps + 2)])
                                            for i in range(len(kp_data))])
    left_wrist_elbow_shoulder = np.nan_to_num(left_wrist_elbow_shoulder)
    right_wrist_elbow_shoulder = np.asarray([cal_angle(kp_data[i, 10*steps:(10*steps+2)], kp_data[i, 8*steps:(8*steps+2)], kp_data[i, 6*steps:(6*steps+2)])
                                              for i in range(len(kp_data))])
    right_wrist_elbow_shoulder = np.nan_to_num(right_wrist_elbow_shoulder)


    right_elbow_shoulder = np.asarray([cal_angle(kp_data[i, 8*steps:(8*steps+2)], kp_data[i, 6*steps:(6*steps+2)], kp_data[i, 5*steps:(5*steps+2)])
                                              for i in range(len(kp_data))])
    right_elbow_shoulder = np.nan_to_num(right_elbow_shoulder)
    left_elbow_shoulder = np.asarray([cal_angle(kp_data[i, 6*steps:(6*steps+2)], kp_data[i, 5*steps:(5*steps+2)], kp_data[i, 7*steps:(7*steps+2)])
                                              for i in range(len(kp_data))])
    left_elbow_shoulder = np.nan_to_num(left_elbow_shoulder)

    # left_ear_shoulder = np.asarray([cal_angle(kp_data[i, 3*steps:(3*steps+2)], kp_data[i, 5*steps:(5*steps+2)], kp_data[i, 6*steps:(6*steps+2)])
    #                                           for i in range(len(kp_data))])
    # left_ear_shoulder = np.nan_to_num(left_ear_shoulder)

    joint_angles = np.array([left_elbow_shoulder_hip,
                    right_elbow_shoulder_hip, left_wrist_elbow_shoulder, right_wrist_elbow_shoulder, right_elbow_shoulder, left_elbow_shoulder]).T

    return joint_angles

def extract_velocity(kp_data): # fast motion
    velocity = np.diff(kp_data, axis=0)
    # print(np.shape(kp_data))
    # print(np.shape(velocity))
    return velocity

def extract_slow_motion(kp_data): # slow motion
    # size of kp_data is [WINDOW_SIZE*FS, 13*2]
    # print("kp_data shape: ", np.shape(kp_data))
    # print("kp_data: ", kp_data)
    kp_data_roll = np.roll(kp_data, -2, axis=0)
    # print("kp_data_roll shape: ", np.shape(kp_data_roll))
    # print("kp_data_roll: ", kp_data_roll)
    slow_motion = kp_data_roll - kp_data
    slow_motion = slow_motion[:-2, :]
    # print("slow_motion shape: ", np.shape(slow_motion))
    return slow_motion

def extract_bone_length(kp_data, steps=2):
    skeleton = [[12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
    bone_length = []
    # print('Shape of kp_data: ', np.shape(kp_data))
    for sk_id, sk in enumerate(skeleton):
        pos1_x_arr = kp_data[:, (sk[0]-1)*steps]
        pos1_y_arr = kp_data[:, (sk[0]-1)*steps+1]

        pos2_x_arr = kp_data[:, (sk[1]-1)*steps]
        pos2_y_arr = kp_data[:, (sk[1]-1)*steps+1]

        skl = np.sqrt(abs(pos2_x_arr - pos1_x_arr)**2 + abs(pos2_y_arr - pos1_y_arr)**2)
        bone_length.append(skl)

    bone_length = np.array(bone_length).T
    # print("Shape of bone_length", np.shape(bone_length))
    return bone_length

def skip_frames(ws_seg):
  chosen_ind = np.arange(0, int(WINDOW_SIZE*FS), N_SKIP_FRAMES)
  chosen_ind = np.append(chosen_ind, int(WINDOW_SIZE * FS - 1))
  ws_seg = ws_seg[:, chosen_ind, :]
  return ws_seg

def normalize_kp(kp_data):
    for i in range(np.shape(kp_data)[0]):
        each_skeleton = kp_data[i, :].copy()
        x_list = each_skeleton[0::2]
        y_list = each_skeleton[1::2]
        x_new = (x_list - x_list[np.nonzero(x_list)].mean())/(max(x_list[np.nonzero(x_list)]) - min(x_list[np.nonzero(x_list)]))
        y_new = (y_list - y_list[np.nonzero(y_list)].mean())/(max(y_list[np.nonzero(y_list)]) - min(y_list[np.nonzero(y_list)]))
        kp_data[i, :][0::2] = x_new
        kp_data[i, :][1::2] = y_new
    return kp_data

def calc_torso(p1, p2, p3):
  # calculate distance from p3 to line p1 and p2
  d = np.linalg.norm(np.cross(p2-p1, p1-p3))/np.linalg.norm(p2-p1)
  return d

def normalize_bone_length(bone_length, kp_data, torso_thres=20):
    torso_dist = [calc_torso(np.array([kp_data[i][11*2], kp_data[i][11*2+1]]), np.array([kp_data[i][12*2], kp_data[i][12*2+1]]), np.array([kp_data[i][0], kp_data[i][1]])) for i in range(len(kp_data))]
    torso_dist = np.array(torso_dist).astype(np.float32)
    if len(np.flatnonzero(torso_dist<torso_thres))>0:
        print("There is very small value in torso_dist")
        torso_dist[torso_dist<torso_thres] = np.mean(torso_dist[~(torso_dist<torso_thres)])

    torso_dist_new = np.repeat(np.array(torso_dist)[:,None], np.shape(bone_length)[1], axis=1)
    bone_length_normalize = bone_length/torso_dist_new
    bone_length_normalize = np.array(bone_length_normalize).astype(np.float32)
    if len(np.flatnonzero(np.isinf(bone_length_normalize)))>0:
        print("There is infinite value in bone_length_normalize")
        bone_length_normalize[np.isinf(bone_length_normalize)] = np.mean(bone_length_normalize[~np.isinf(bone_length_normalize)])
    return bone_length_normalize


def load_data_comb(front_keypoint_csv, back_keypoint_csv, ann_csv):
  front_kp_df = pd.read_csv(front_keypoint_csv)
  front_kp_df = front_kp_df.loc[:, ~front_kp_df.columns.str.contains('^Unnamed')]
  front_kp_df = remove_redundant_kp(front_kp_df)

  back_kp_df = pd.read_csv(back_keypoint_csv)
  back_kp_df = back_kp_df.loc[:, ~back_kp_df.columns.str.contains('^Unnamed')]
  back_kp_df = remove_redundant_kp(back_kp_df)

  ann_df = pd.read_csv(ann_csv)
  ann_df = ann_df.loc[:, ~ann_df.columns.str.contains('^Unnamed')]
  ann_df['annotation_str'] = ann_df['annotation'].copy()
  ann_df = convert_ann_to_int(ann_df)

  # compare length of back and front view
  if len(front_kp_df) != len(back_kp_df):
      min_length = np.min([len(front_kp_df), len(back_kp_df)])
      front_kp_df = front_kp_df[: min_length, :]
      back_kp_df = back_kp_df[: min_length, :]

  return front_kp_df, back_kp_df, ann_df

def generate_data(user_id, ann_path, folder_path):
    # print(user_id)
    front_keypoint_csv = folder_path + "/Front_" + user_id + "_{}.csv".format(EXTENSION)
    back_keypoint_csv = folder_path + "/Back_" + user_id + "_{}.csv".format(EXTENSION)
    ann_csv = ann_path + "/Front_" + user_id + ".csv"
    all_data = []
    all_feature = []
    all_label = []
    front_kp_df, back_kp_df, ann_df = load_data_comb(front_keypoint_csv, back_keypoint_csv, ann_csv)
    for i, kp_name in enumerate(front_kp_df.columns):
        front_kp_df.iloc[:, i] = smooth_kp(np.array(front_kp_df.iloc[:, i]))
        back_kp_df.iloc[:, i] = smooth_kp(np.array(back_kp_df.iloc[:, i]))
    for i in range(len(ann_df)):
        front_seg = front_kp_df.loc[int(ann_df['start_time'][i]*FS): int(ann_df['stop_time'][i]*FS)]
        back_seg = back_kp_df.loc[int(ann_df['start_time'][i]*FS): int(ann_df['stop_time'][i]*FS)]
        seg_label = ann_df["annotation"].iloc[i]
        timestamp_seg = np.arange(int(ann_df['start_time'][i]*FS), int(ann_df['stop_time'][i]*FS)+1)
        if len(front_seg) > 0 and (len(front_seg) >= WINDOW_SIZE * FS):
            front_ws_seg = segment(np.array(front_seg), max_time=len(front_seg), sub_window_size=WINDOW_SIZE * FS, stride_size=int((WINDOW_SIZE - OVERLAP_RATE) * FS))
            front_ws_seg = skip_frames(front_ws_seg)

            back_ws_seg = segment(np.array(back_seg), max_time=len(back_seg), sub_window_size=WINDOW_SIZE * FS, stride_size=int((WINDOW_SIZE - OVERLAP_RATE) * FS))
            back_ws_seg = skip_frames(back_ws_seg)

            # front_ws_seg = np.asarray([normalize_kp(front_ws_seg[i]) for i in range(len(front_ws_seg))]) # Version 20
            # back_ws_seg = np.asarray([normalize_kp(back_ws_seg[i]) for i in range(len(back_ws_seg))]) # Version 20

            # print("Shape of ws_seg: ", np.array(ws_seg).shape)
            # kp_df.drop(np.arange(int(ann_df['start_time'][i]*FS), int(ann_df['stop_time'][i]*FS)) , inplace=True)

            front_feature_seg = [extract_feature(front_ws_seg[i], FS) for i in range(len(front_ws_seg))]
            back_feature_seg = [extract_feature(back_ws_seg[i], FS) for i in range(len(back_ws_seg))]
            # print("Shape of feature_seg: ", np.array(feature_seg).shape)

            timestamp_ws_seg = segment(np.array(timestamp_seg),  max_time=len(front_seg), sub_window_size=WINDOW_SIZE * FS, stride_size=int((WINDOW_SIZE - OVERLAP_RATE) * FS)) # version 22
            # print("shape of timestamp_ws_seg: ", np.shape(timestamp_ws_seg))
            timestamp_ws_seg = skip_frames(np.expand_dims(timestamp_ws_seg, axis=-1)) # version 22
            # print("shape of timestamp_ws_seg skip frame: ", np.shape(timestamp_ws_seg))
            timestamp_feature_seg = [extract_feature(timestamp_ws_seg[i], FS) for i in range(len(timestamp_ws_seg))] # version 22
            # print("shape of timestamp_feature_seg: ", np.shape(timestamp_feature_seg))

            front_joint_angles = extract_joint_angles(np.array(front_seg))
            front_joint_angles_seg = segment(front_joint_angles, max_time=len(front_seg), sub_window_size=WINDOW_SIZE * FS,
                                              stride_size=int((WINDOW_SIZE - OVERLAP_RATE) * FS))
            front_joint_angles_seg = skip_frames(front_joint_angles_seg) # version 20

            back_joint_angles = extract_joint_angles(np.array(back_seg))
            back_joint_angles_seg = segment(back_joint_angles, max_time=len(back_seg), sub_window_size=WINDOW_SIZE * FS,
                                              stride_size=int((WINDOW_SIZE - OVERLAP_RATE) * FS))
            back_joint_angles_seg = skip_frames(back_joint_angles_seg) # version 20
            # print("Shape of joint_angles_seg: ", np.array(joint_angles_seg).shape)
            # print("Shape of joint_angles_seg: ", np.shape(joint_angles_seg))

            front_feature_joint_angles_seg = [extract_feature(front_joint_angles_seg[i], FS) for i in
                                        range(len(front_joint_angles_seg))]

            back_feature_joint_angles_seg = [extract_feature(back_joint_angles_seg[i], FS) for i in
                                        range(len(back_joint_angles_seg))]
            # print("Shape of feature_joint_angles_seg: ", np.array(feature_joint_angles_seg).shape)

            # cg_seg = [extract_cg(ws_seg[i]) for i in range(len(ws_seg))]
            # feature_cg_seg = [extract_feature(cg_seg[i], FS) for i in range(len(cg_seg))]  # version 10

            # print("ws_seg: ", np.shape(ws_seg))
            # print("feature_seg: ", np.shape(feature_seg))
            # print("feature_joint_angles_seg: ", np.shape(feature_joint_angles_seg))
            front_bone_length_seg = [extract_bone_length(front_ws_seg[i]) for i in range(len(front_ws_seg))]
            # front_bone_length_seg = [normalize_bone_length(extract_bone_length(front_ws_seg[i]), front_ws_seg[i]) for i in range(len(front_ws_seg))] # version 20
            back_bone_length_seg = [extract_bone_length(back_ws_seg[i]) for i in range(len(back_ws_seg))]
            # back_bone_length_seg = [normalize_bone_length(extract_bone_length(back_ws_seg[i]), back_ws_seg[i]) for i in range(len(back_ws_seg))] # version 20
            # print("Shape of bone_length_seg: ", np.array(bone_length_seg).shape)

            # print("Shape of bone_length_seg: ", np.shape(bone_length_seg))
            front_feature_bone_length = [extract_feature(front_bone_length_seg[i], FS) for i in
                                        range(len(front_bone_length_seg))]
            back_feature_bone_length = [extract_feature(back_bone_length_seg[i], FS) for i in
                                        range(len(back_bone_length_seg))]
            # print("Shape of feature_bone_length: ", np.array(feature_bone_length).shape)

            front_feature_velocity_seg = [extract_feature(extract_velocity(front_ws_seg[i]), FS) for i in range(len(front_ws_seg))] # version 5 + 17
            back_feature_velocity_seg = [extract_feature(extract_velocity(back_ws_seg[i]), FS) for i in range(len(back_ws_seg))] # version 5 + 17

            front_feature_slow_motion = [extract_feature(extract_slow_motion(front_ws_seg[i]), FS) for i in
                                        range(len(front_ws_seg))] # version 18
            back_feature_slow_motion = [extract_feature(extract_slow_motion(back_ws_seg[i]), FS) for i in
                                        range(len(back_ws_seg))] # version 18

            feature_seg = np.concatenate([front_feature_seg, front_feature_joint_angles_seg, front_feature_velocity_seg, front_feature_bone_length, front_feature_slow_motion,
                                          back_feature_seg, back_feature_joint_angles_seg, back_feature_velocity_seg, back_feature_bone_length, back_feature_slow_motion, timestamp_feature_seg], axis=1) # version 22

            all_data.extend(front_ws_seg)
            all_label.extend([int(seg_label)]*len(front_ws_seg))
            all_feature.extend(feature_seg)

    return (all_data, all_label, all_feature)

def convert_ann_to_int_combined(ann_df):
  # Convert annotation to interger
  # Explanation to patient: 0
  # Auscultation: 1
  # Washing hands: 2
  # Checking cuff pressure: 3
  # Wearing apron: 4
  # Wearing goggles: 5
  # Wearing gloves: 6
  # Opening lids: 7
  # Preparing alcohol tissue: 8
  # Connecting suctioning catheter: 9
  # pressing button = touching a screen of ventilator: 10
  # Removal of airway: 11
  # Suctioning: 12
  # Refitting the airway: 13
  # Wiping cather: 14
  # Washing catheter: 15
  # Disconnecting catheter: 16
  # Closing lids: close lids: 17
  # Putting back catheter: turn off the machine, put back the catheter: 18
  # Removal of gloves: 19
  # Removal of goggles: 20
  # Removal of apron: 21
  # Others: 22

  ann_df['annotation'].iloc[np.flatnonzero(ann_df['annotation'].str.lower() == "explanation to patient")] = 0
  ann_df['annotation'].iloc[np.flatnonzero(ann_df['annotation'].str.lower() == "auscultation")] = 1
  ann_df['annotation'].iloc[np.flatnonzero(ann_df['annotation'].str.lower() == "washing hands")] = 2
  ann_df['annotation'].iloc[np.flatnonzero(ann_df['annotation'].str.lower() == "checking cuff pressure")] = 3
  ann_df['annotation'].iloc[np.flatnonzero(ann_df['annotation'].str.lower() == "wearing apron")] = 4
  ann_df['annotation'].iloc[np.flatnonzero(ann_df['annotation'].str.lower() == "wearing goggles")] = 4
  ann_df['annotation'].iloc[np.flatnonzero(ann_df['annotation'].str.lower() == "wearing gloves")] = 4
  ann_df['annotation'].iloc[np.flatnonzero((ann_df['annotation'].str.lower() == "opening lids")|(ann_df['annotation'].str.lower() == "open lids"))] = 5
  ann_df['annotation'].iloc[np.flatnonzero(ann_df['annotation'].str.lower() == "preparing alcohol tissue")] = 5
  ann_df['annotation'].iloc[np.flatnonzero((ann_df['annotation'].str.lower() == "connecting suctioning catheter")|(ann_df['annotation'].str.lower() == "connecting catheter"))] = 5
  ann_df['annotation'].iloc[np.flatnonzero(ann_df['annotation'].str.lower() == "pressing button")] = 6
  ann_df['annotation'].iloc[np.flatnonzero(ann_df['annotation'].str.lower() == "removal of airway")] = 7
  ann_df['annotation'].iloc[np.flatnonzero(ann_df['annotation'].str.lower() == "suctioning")] = 8
  ann_df['annotation'].iloc[np.flatnonzero((ann_df['annotation'].str.lower() == "refitting the airway")|(ann_df['annotation'].str.lower() == "refitting airway"))] = 9
  ann_df['annotation'].iloc[np.flatnonzero(ann_df['annotation'].str.lower() == "wiping catheter")] = 10
  ann_df['annotation'].iloc[np.flatnonzero((ann_df['annotation'].str.lower() == "washing catheter")|(ann_df['annotation'].str.lower() == "wash catheter"))] = 11
  ann_df['annotation'].iloc[np.flatnonzero(ann_df['annotation'].str.lower() == "disconnecting catheter")] = 12
  ann_df['annotation'].iloc[np.flatnonzero(ann_df['annotation'].str.lower() == "closing lids")] = 12
  ann_df['annotation'].iloc[np.flatnonzero((ann_df['annotation'].str.lower() == "putting back catheter")|(ann_df['annotation'].str.lower() == "putting back catheter removal"))] = 12
  ann_df['annotation'].iloc[np.flatnonzero(ann_df['annotation'].str.lower() == "removal of gloves")] = 13
  ann_df['annotation'].iloc[np.flatnonzero(ann_df['annotation'].str.lower() == "removal of goggles")] = 13
  ann_df['annotation'].iloc[np.flatnonzero(ann_df['annotation'].str.lower() == "removal of apron")] = 13
  ann_df['annotation'].iloc[np.flatnonzero(ann_df['annotation'].str.lower() == "others")] = 14

  return ann_df

def load_data_combined(keypoint_csv, ann_csv):
  kp_df = pd.read_csv(keypoint_csv)
  kp_df = kp_df.loc[:, ~kp_df.columns.str.contains('^Unnamed')]
  kp_df = remove_redundant_kp(kp_df)

  ann_df = pd.read_csv(ann_csv)
  ann_df = ann_df.loc[:, ~ann_df.columns.str.contains('^Unnamed')]
  ann_df['annotation_str'] = ann_df['annotation'].copy()
  ann_df = convert_ann_to_int_combined(ann_df)

  return kp_df, ann_df

def merge_consecutive_label(ann_df):
    ann_df = ann_df.drop(columns=['annotation_str'])
    ann_df['key'] = (ann_df['annotation'] != ann_df['annotation'].shift(1)).astype(int).cumsum()
    aggregation_functions = {'start_time': 'min', 'stop_time': 'max'}
    ann_df = ann_df.groupby(['key', 'annotation'], as_index=False).aggregate(aggregation_functions).reindex(columns=ann_df.columns)
    ann_df = ann_df.drop(columns=['key'])
    return ann_df

def post_processing(prediction):
    # remove short segments (= 4s)
    # segments = np.split(prediction, np.flatnonzero(np.diff(prediction) != 0) + 1)
    # short_activities = [1, 3, 5]
    # short_activities = [7, 10, 11, 13, 17, 18]
    # len_segments = [len(segments[i]) for i in range(len(segments))]
    # for i, segment in enumerate(segments):
    #     # if (segment[0] in short_activities) or len(segment) > 1:
    #     if (segment[0] in short_activities) or len(segment) >= 2:
    #         continue
    #     if i == 0:
    #         segments[i] = np.array([segments[i+1][0]]*len(segments[i]))
    #     elif i == len(segments) - 1:
    #         segments[i] = np.array([segments[i-1][0]]*len(segments[i]))
    #     else:
    #         chosen_id = np.argmax([len(segments[i-1]), len(segments[i+1])])
    #         if chosen_id == 0:
    #             segments[i] = np.array([segments[i-1][0]]*len(segments[i]))
    #         if chosen_id == 1:
    #             segments[i] = np.array([segments[i+1][0]]*len(segments[i]))
    # processed_prediction = np.array([segments]).flatten()
    # processed_prediction = list(np.concatenate(segments).flat)
    processed_prediction = np.array(prediction)
    # region if remove airway + suctioning + remove airway => remove airway + suctioning + refit airway
    remove_airway_id = np.flatnonzero(processed_prediction == ACTIVITY_DICT["Removal of airway"])
    suctioning_id = np.flatnonzero(processed_prediction == ACTIVITY_DICT["Suctioning"])
    # print("remove_airway_id: ", remove_airway_id)
    # print("suctioning_id: ", suctioning_id)
    if len(suctioning_id) > 0:
        suctioning_group = np.split(suctioning_id, np.flatnonzero(np.diff(suctioning_id) > 1) + 1)

        if len(suctioning_group) > 0:
            false_id = remove_airway_id[np.flatnonzero(remove_airway_id > suctioning_group[0][-1])]
            # print("false_id: ", false_id)
            # print(processed_prediction)
            # print(processed_prediction[false_id])
            # print(ACTIVITY_DICT["Refitting the airway"])
            processed_prediction[false_id] = ACTIVITY_DICT["Refitting the airway"]

    # region if wearing apron + > 6 seconds different activity + wearing apron => wearing apron + > 6 seconds different activity + remove apron
    wearing_apron_id = np.flatnonzero(np.array(processed_prediction) == ACTIVITY_DICT["Wearing apron"])
    # print("wearing_apron_id before: ", wearing_apron_id)
    # print("removal_of_apron_id before: ", np.flatnonzero(np.array(processed_prediction) == ACTIVITY_DICT["Removal of apron"]))
    if len(wearing_apron_id) > 0:
        wearing_apron_group = np.split(wearing_apron_id, np.flatnonzero(np.diff(wearing_apron_id) > 1) + 1)
        for count, group in enumerate(wearing_apron_group):
            if count > 0 and group[0] - wearing_apron_group[count - 1][-1] >= 3 and len(wearing_apron_group[count - 1]) >= 2:
                if processed_prediction[wearing_apron_group[count - 1][-1]] == ACTIVITY_DICT["Wearing apron"]:
                    processed_prediction[group] = ACTIVITY_DICT["Removal of apron"]
                else:
                    processed_prediction[group] = ACTIVITY_DICT["Wearing apron"]
    # print("wearing_apron_id after: ", np.flatnonzero(np.array(processed_prediction) == ACTIVITY_DICT["Wearing apron"]))
    # print("removal_of_apron_id after: ", np.flatnonzero(np.array(processed_prediction) == ACTIVITY_DICT["Removal of apron"]))

    # region if opening lids + > 6 seconds different activity + opening lids => opening lids + > 6 seconds different activity + closing lids
    opening_lids_id = np.flatnonzero(np.array(processed_prediction) == ACTIVITY_DICT["Opening lids"])
    # print("opening_lids_id before: ", opening_lids_id)
    # print("closing_lids_id before: ", np.flatnonzero(np.array(processed_prediction) == ACTIVITY_DICT["Closing lids"]))
    if len(opening_lids_id) > 0:
        opening_lids_group = np.split(opening_lids_id, np.flatnonzero(np.diff(opening_lids_id) > 1) + 1)
        for count, group in enumerate(opening_lids_group):
            if count > 0 and group[0] - opening_lids_group[count - 1][-1] >= 3:
                if processed_prediction[opening_lids_group[count - 1][-1]] == ACTIVITY_DICT["Opening lids"]:
                    processed_prediction[group] = ACTIVITY_DICT["Closing lids"]
                else:
                    processed_prediction[group] = ACTIVITY_DICT["Opening lids"]
    # print("opening_lids_id after: ", np.flatnonzero(np.array(processed_prediction) == ACTIVITY_DICT["Opening lids"]))
    # print("closing_lids_id after: ", np.flatnonzero(np.array(processed_prediction) == ACTIVITY_DICT["Closing lids"]))

    # remove short segments (= 4s)
    segments = np.split(processed_prediction, np.flatnonzero(np.diff(processed_prediction) != 0) + 1)
    # print("segments: ", segments)
    # short_activities = [2, 7, 10, 11, 13, 14, 16, 17, 18]
    short_activities = [0, 2, 5, 7, 10, 11, 13, 14, 15, 16, 17, 20]
    len_segments = [len(segments[i]) for i in range(len(segments))]
    for i, segment in enumerate(segments):
        # if (segment[0] in short_activities) or len(segment) > 1:
        if (segment[0] in short_activities) or len(segment) >= 2:
            continue
        # print("change segment: ", segments[i])
        if i == 0:
            segments[i] = np.array([segments[i+1][0]]*len(segments[i]))
        elif i == len(segments) - 1:
            segments[i] = np.array([segments[i-1][0]]*len(segments[i]))
        else:
            chosen_id = np.argmax([len(segments[i-1]), len(segments[i+1])])
            if chosen_id == 0:
                segments[i] = np.array([segments[i-1][0]]*len(segments[i]))
            if chosen_id == 1:
                segments[i] = np.array([segments[i+1][0]]*len(segments[i]))
        # print("segment changed to: ", segments[i])
    processed_prediction = np.array([segments]).flatten()
    processed_prediction = list(np.concatenate(segments).flat)
    processed_prediction = np.array(processed_prediction)

    # segments = np.split(processed_prediction, np.flatnonzero(np.diff(processed_prediction) != 0) + 1)
    # short_activities = [1, 3, 5]
    # len_segments = [len(segments[i]) for i in range(len(segments))]
    # for i, segment in enumerate(segments):
    #     if (segment[0] in short_activities) or len(segment) > 2:
    #         continue
    #     if i == 0:
    #         segments[i] = np.array([segments[i+1][0]]*len(segments[i]))
    #     elif i == len(segments) - 1:
    #         segments[i] = np.array([segments[i-1][0]]*len(segments[i]))
    #     else:
    #         chosen_id = np.argmax([len(segments[i-1]), len(segments[i+1])])
    #         if chosen_id == 0:
    #             segments[i] = np.array([segments[i-1][0]]*len(segments[i]))
    #         if chosen_id == 1:
    #             segments[i] = np.array([segments[i+1][0]]*len(segments[i]))
    # processed_prediction = list(np.concatenate(segments).flat)
    return processed_prediction


def convert_detail_2_long_act(prediction):
  prediction_cp = deepcopy(prediction)
  # 4, 5, 6 => wearing PPE
  prediction_cp[np.flatnonzero((prediction_cp == 5)|(prediction_cp == 6))] = 4
  # 7, 8, 9 => catheter preparation 5
  prediction_cp[np.flatnonzero((prediction_cp == 7)|(prediction_cp == 8)|(prediction_cp == 9))] = 5
  # 10, 11, 12, 13, 14, 15 =>  6, 7, 8, 9, 10, 11
  prediction_cp[np.flatnonzero((prediction_cp == 10)|(prediction_cp == 11)|(prediction_cp == 12)|(prediction_cp == 13)|(prediction_cp == 14)|(prediction_cp == 15))] -= 4
  # 16, 17, 18 => catheter disinfection 12
  prediction_cp[np.flatnonzero((prediction_cp == 16)|(prediction_cp == 17)|(prediction_cp == 18))] = 12
  # 19, 20, 21 => removal of PPE 13
  prediction_cp[np.flatnonzero((prediction_cp == 19)|(prediction_cp == 20)|(prediction_cp == 21))] = 13
  # 22 => 14
  prediction_cp[np.flatnonzero(prediction_cp == 22)] = 14
  return prediction_cp