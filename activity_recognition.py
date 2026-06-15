import numpy as np
import datetime
import joblib
from collections import Counter
from imblearn.over_sampling import SMOTE
from activity_recognition_utils import *
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from multiprocessing import Pool
import warnings
warnings.simplefilter('ignore')



def baseline_combined_1_view(FOLDER_PATH = "/content/drive/MyDrive/STUDY MASTER/PROJECT/Hokkaido prj/Data/New CSV",
    ANN_PATH = "/content/drive/MyDrive/STUDY MASTER/PROJECT/Hokkaido prj/Data/Ann_CSV",
    SAVE_MODEL_PATH = "/content/drive/MyDrive/STUDY MASTER/PROJECT/Hokkaido prj/New_Model"):

    all_data = []
    all_label = []
    all_feature = []

    all_data_test = []
    all_label_test = []
    all_feature_test = []

    for user_id in USER_ID:
        keypoint_csv = FOLDER_PATH + "/Front_" + user_id + "_{}.csv".format(EXTENSION)
        ann_csv = ANN_PATH + "/Front_" + user_id + ".csv"
        kp_df, ann_df = load_data_combined(keypoint_csv, ann_csv)
        # ann_df = merge_consecutive_label(ann_df)
        for i, kp_name in enumerate(kp_df.columns):
            kp_df.iloc[:, i] = smooth_kp(np.array(kp_df.iloc[:, i]))
        for i in range(len(ann_df)):
            seg = kp_df.loc[int(ann_df['start_time'][i] * FS): int(ann_df['stop_time'][i] * FS)]
            seg_label = ann_df["annotation"].iloc[i]
            timestamp_seg = np.arange(int(ann_df['start_time'][i] * FS), int(ann_df['stop_time'][i] * FS) + 1)

            if len(seg) > 0 and (len(seg) >= WINDOW_SIZE * FS):
                ws_seg = segment(np.array(seg), max_time=len(seg), sub_window_size=WINDOW_SIZE * FS,
                                 stride_size=int((WINDOW_SIZE - OVERLAP_RATE) * FS))
                ws_seg = skip_frames(ws_seg)  # version 20

                # print("Shape of ws_seg: ", np.array(ws_seg).shape)
                # kp_df.drop(np.arange(int(ann_df['start_time'][i]*FS), int(ann_df['stop_time'][i]*FS)) , inplace=True)

                feature_seg = [extract_feature(ws_seg[i], FS) for i in range(len(ws_seg))]
                # print("Shape of feature_seg: ", np.array(feature_seg).shape)

                joint_angles = extract_joint_angles(np.array(seg))
                joint_angles_seg = segment(joint_angles, max_time=len(seg), sub_window_size=WINDOW_SIZE * FS,
                                           stride_size=int((WINDOW_SIZE - OVERLAP_RATE) * FS))
                joint_angles_seg = skip_frames(joint_angles_seg)  # version 20
                # print("Shape of joint_angles_seg: ", np.array(joint_angles_seg).shape)
                # print("Shape of joint_angles_seg: ", np.shape(joint_angles_seg))

                feature_joint_angles_seg = [extract_feature(joint_angles_seg[i], FS) for i in
                                            range(len(joint_angles_seg))]
                # print("Shape of feature_joint_angles_seg: ", np.array(feature_joint_angles_seg).shape)

                # cg_seg = [extract_cg(ws_seg[i]) for i in range(len(ws_seg))]
                # feature_cg_seg = [extract_feature(cg_seg[i], FS) for i in range(len(cg_seg))]  # version 10

                # print("ws_seg: ", np.shape(ws_seg))
                # print("feature_seg: ", np.shape(feature_seg))
                # print("feature_joint_angles_seg: ", np.shape(feature_joint_angles_seg))
                bone_length_seg = [extract_bone_length(ws_seg[i]) for i in range(len(ws_seg))]
                # print("Shape of bone_length_seg: ", np.array(bone_length_seg).shape)

                # print("Shape of bone_length_seg: ", np.shape(bone_length_seg))
                feature_bone_length = [extract_feature(bone_length_seg[i], FS) for i in
                                       range(len(bone_length_seg))]
                # print("Shape of feature_bone_length: ", np.array(feature_bone_length).shape)

                velocity_seg = [extract_velocity(ws_seg[i]) for i in range(len(ws_seg))]
                # print("Shape of velocity_seg: ", np.shape(velocity_seg))
                feature_velocity_seg = [extract_feature(extract_velocity(ws_seg[i]), FS) for i in
                                        range(len(ws_seg))]  # version 5
                # print("Shape of feature_velocity_seg: ", np.array(feature_velocity_seg).shape)

                slow_motion_seg = [extract_slow_motion(ws_seg[i]) for i in range(len(ws_seg))]  # version 18
                # print("Shape of slow_motion_seg: ", np.shape(slow_motion_seg))
                feature_slow_motion = [extract_feature(slow_motion_seg[i], FS) for i in
                                       range(len(slow_motion_seg))]  # version 18

                timestamp_ws_seg = segment(np.array(timestamp_seg), max_time=len(seg), sub_window_size=WINDOW_SIZE * FS,
                                           stride_size=int((WINDOW_SIZE - OVERLAP_RATE) * FS))  # version 22
                # print("shape of timestamp_ws_seg: ", np.shape(timestamp_ws_seg))
                timestamp_ws_seg = skip_frames(np.expand_dims(timestamp_ws_seg, axis=-1))  # version 22
                # print("shape of timestamp_ws_seg skip frame: ", np.shape(timestamp_ws_seg))
                timestamp_feature_seg = [extract_feature(timestamp_ws_seg[i], FS) for i in
                                         range(len(timestamp_ws_seg))]  # version 22
                # print("shape of timestamp_feature_seg: ", np.shape(timestamp_feature_seg))

                # feature_seg = np.concatenate([feature_seg, joint_angles_seg.reshape(joint_angles_seg.shape[0], -1)], axis=1) # version 3, 7
                # feature_seg = np.concatenate([feature_seg, joint_angles_seg.reshape(joint_angles_seg.shape[0], -1), feature_joint_angles_seg, feature_velocity_seg], axis=1) # version 8
                # feature_seg = np.concatenate([feature_seg, feature_joint_angles_seg], axis=1) # version 4
                # feature_seg = np.concatenate([feature_seg, feature_joint_angles_seg, feature_velocity_seg], axis=1) # version 5
                # feature_seg = np.concatenate([feature_seg, feature_joint_angles_seg, feature_velocity_seg, feature_bone_length, feature_slow_motion], axis=1) # version 19, 20
                # print("Shape of feature_seg: ", np.array(feature_seg.shape))
                # feature_seg = np.concatenate([feature_seg, feature_velocity_seg], axis=1) # version 13
                # feature_seg = np.concatenate([feature_seg, feature_cg_seg, feature_joint_angles_seg, feature_velocity_seg], axis=1) # version 10
                # get selected features
                # feature_seg = feature_seg[:, selected_ft_ind]

                # New data
                feature_seg = np.concatenate(
                    [feature_seg, feature_joint_angles_seg, feature_velocity_seg, feature_bone_length,
                     feature_slow_motion, timestamp_feature_seg], axis=1)  # version 1, 2
                # feature_seg = np.concatenate([feature_seg, feature_joint_angles_seg, feature_bone_length], axis=1) # version 3
                # feature_seg = np.concatenate([feature_seg, feature_joint_angles_seg, feature_velocity_seg, feature_bone_length, feature_slow_motion], axis=1) # version 4
                # feature_seg = feature_seg[:, chose_feature_id] # version 4 and 5

                if user_id in TEST_ID:
                    all_data_test.extend(ws_seg)
                    all_label_test.extend([int(seg_label)] * len(ws_seg))
                    all_feature_test.extend(feature_seg)
                else:
                    all_data.extend(ws_seg)
                    all_label.extend([int(seg_label)] * len(ws_seg))
                    all_feature.extend(feature_seg)

    print("Data shape: ", np.shape(all_data))
    print("Label shape: ", np.shape(all_label))
    print("Feature shape: ", np.shape(all_feature), "\n")

    print("Data test shape: ", np.shape(all_data_test))
    print("Label test shape: ", np.shape(all_label_test))
    print("Feature test shape: ", np.shape(all_feature_test))

    # Splitting and Training data
    print("Total samples of training data: {}".format(len(all_feature)))
    print("Total samples of testing data: {}".format(len(all_feature_test)))

    model_ml = RandomForestClassifier(n_estimators=500, n_jobs=-1)
    model_ml.fit(all_feature, all_label)

    joblib.dump(model_ml, "{}/randomforest_{}.joblib".format(SAVE_MODEL_PATH, VERSION))

    # Evaluating
    y_predict = model_ml.predict(all_feature_test)
    print(classification_report(all_label_test, y_predict))
    cm = confusion_matrix(all_label_test, y_predict, labels=np.unique(all_label_test))
    cm_norm = cm / np.sum(cm, axis=1, keepdims=True)
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', annot_kws={"fontsize": 6},
                yticklabels=np.unique(all_label_test), xticklabels=np.unique(all_label_test))
    plt.show()


def baseline_without_skipframe(FOLDER_PATH = "/content/drive/MyDrive/STUDY MASTER/PROJECT/Hokkaido prj/Data/New CSV",
    ANN_PATH = "/content/drive/MyDrive/STUDY MASTER/PROJECT/Hokkaido prj/Data/Ann_CSV",
    SAVE_MODEL_PATH = "/content/drive/MyDrive/STUDY MASTER/PROJECT/Hokkaido prj/New_Model"):

    all_data = []
    all_label = []
    all_feature = []

    all_data_test = []
    all_label_test = []
    all_feature_test = []

    for user_id in USER_ID:
        keypoint_csv = FOLDER_PATH + "/Front_" + user_id + "_{}.csv".format(EXTENSION)
        ann_csv = ANN_PATH + "/Front_" + user_id + ".csv"
        kp_df, ann_df = load_data_combined(keypoint_csv, ann_csv)
        # ann_df = merge_consecutive_label(ann_df)
        for i, kp_name in enumerate(kp_df.columns):
            kp_df.iloc[:, i] = smooth_kp(np.array(kp_df.iloc[:, i]))
        for i in range(len(ann_df)):
            seg = kp_df.loc[int(ann_df['start_time'][i] * FS): int(ann_df['stop_time'][i] * FS)]
            seg_label = ann_df["annotation"].iloc[i]
            timestamp_seg = np.arange(int(ann_df['start_time'][i] * FS), int(ann_df['stop_time'][i] * FS) + 1)

            if len(seg) > 0 and (len(seg) >= WINDOW_SIZE * FS):
                ws_seg = segment(np.array(seg), max_time=len(seg), sub_window_size=WINDOW_SIZE * FS,
                                 stride_size=int((WINDOW_SIZE - OVERLAP_RATE) * FS))
                # ws_seg = skip_frames(ws_seg) # version 20

                # print("Shape of ws_seg: ", np.array(ws_seg).shape)
                # kp_df.drop(np.arange(int(ann_df['start_time'][i]*FS), int(ann_df['stop_time'][i]*FS)) , inplace=True)

                feature_seg = [extract_feature(ws_seg[i], FS) for i in range(len(ws_seg))]
                # print("Shape of feature_seg: ", np.array(feature_seg).shape)

                joint_angles = extract_joint_angles(np.array(seg))
                joint_angles_seg = segment(joint_angles, max_time=len(seg), sub_window_size=WINDOW_SIZE * FS,
                                           stride_size=int((WINDOW_SIZE - OVERLAP_RATE) * FS))
                # joint_angles_seg = skip_frames(joint_angles_seg) # version 20
                # print("Shape of joint_angles_seg: ", np.array(joint_angles_seg).shape)
                # print("Shape of joint_angles_seg: ", np.shape(joint_angles_seg))

                feature_joint_angles_seg = [extract_feature(joint_angles_seg[i], FS) for i in
                                            range(len(joint_angles_seg))]
                # print("Shape of feature_joint_angles_seg: ", np.array(feature_joint_angles_seg).shape)

                # cg_seg = [extract_cg(ws_seg[i]) for i in range(len(ws_seg))]
                # feature_cg_seg = [extract_feature(cg_seg[i], FS) for i in range(len(cg_seg))]  # version 10

                # print("ws_seg: ", np.shape(ws_seg))
                # print("feature_seg: ", np.shape(feature_seg))
                # print("feature_joint_angles_seg: ", np.shape(feature_joint_angles_seg))
                bone_length_seg = [extract_bone_length(ws_seg[i]) for i in range(len(ws_seg))]
                # print("Shape of bone_length_seg: ", np.array(bone_length_seg).shape)

                # print("Shape of bone_length_seg: ", np.shape(bone_length_seg))
                feature_bone_length = [extract_feature(bone_length_seg[i], FS) for i in
                                       range(len(bone_length_seg))]
                # print("Shape of feature_bone_length: ", np.array(feature_bone_length).shape)

                velocity_seg = [extract_velocity(ws_seg[i]) for i in range(len(ws_seg))]
                # print("Shape of velocity_seg: ", np.shape(velocity_seg))
                feature_velocity_seg = [extract_feature(extract_velocity(ws_seg[i]), FS) for i in
                                        range(len(ws_seg))]  # version 5
                # print("Shape of feature_velocity_seg: ", np.array(feature_velocity_seg).shape)

                slow_motion_seg = [extract_slow_motion(ws_seg[i]) for i in range(len(ws_seg))]  # version 18
                # print("Shape of slow_motion_seg: ", np.shape(slow_motion_seg))
                feature_slow_motion = [extract_feature(slow_motion_seg[i], FS) for i in
                                       range(len(slow_motion_seg))]  # version 18

                timestamp_ws_seg = segment(np.array(timestamp_seg), max_time=len(seg), sub_window_size=WINDOW_SIZE * FS,
                                           stride_size=int((WINDOW_SIZE - OVERLAP_RATE) * FS))  # version 22
                # print("shape of timestamp_ws_seg: ", np.shape(timestamp_ws_seg))
                # timestamp_ws_seg = skip_frames(np.expand_dims(timestamp_ws_seg, axis=-1)) # version 22
                # print("shape of timestamp_ws_seg skip frame: ", np.shape(timestamp_ws_seg))
                timestamp_ws_seg = np.expand_dims(timestamp_ws_seg, axis=-1)
                timestamp_feature_seg = [extract_feature(timestamp_ws_seg[i], FS) for i in
                                         range(len(timestamp_ws_seg))]  # version 22
                # print("shape of timestamp_feature_seg: ", np.shape(timestamp_feature_seg))

                # feature_seg = np.concatenate([feature_seg, joint_angles_seg.reshape(joint_angles_seg.shape[0], -1)], axis=1) # version 3, 7
                # feature_seg = np.concatenate([feature_seg, joint_angles_seg.reshape(joint_angles_seg.shape[0], -1), feature_joint_angles_seg, feature_velocity_seg], axis=1) # version 8
                # feature_seg = np.concatenate([feature_seg, feature_joint_angles_seg], axis=1) # version 4
                # feature_seg = np.concatenate([feature_seg, feature_joint_angles_seg, feature_velocity_seg], axis=1) # version 5
                # feature_seg = np.concatenate([feature_seg, feature_joint_angles_seg, feature_velocity_seg, feature_bone_length, feature_slow_motion], axis=1) # version 19, 20
                # print("Shape of feature_seg: ", np.array(feature_seg.shape))
                # feature_seg = np.concatenate([feature_seg, feature_velocity_seg], axis=1) # version 13
                # feature_seg = np.concatenate([feature_seg, feature_cg_seg, feature_joint_angles_seg, feature_velocity_seg], axis=1) # version 10
                # get selected features
                # feature_seg = feature_seg[:, selected_ft_ind]

                # New data
                feature_seg = np.concatenate(
                    [feature_seg, feature_joint_angles_seg, feature_velocity_seg, feature_bone_length,
                     feature_slow_motion, timestamp_feature_seg], axis=1)  # version 1, 2
                # feature_seg = np.concatenate([feature_seg, feature_joint_angles_seg, feature_bone_length], axis=1) # version 3
                # feature_seg = np.concatenate([feature_seg, feature_joint_angles_seg, feature_velocity_seg, feature_bone_length, feature_slow_motion], axis=1) # version 4
                # feature_seg = feature_seg[:, chose_feature_id] # version 4 and 5

                if user_id in TEST_ID:
                    all_data_test.extend(ws_seg)
                    all_label_test.extend([int(seg_label)] * len(ws_seg))
                    all_feature_test.extend(feature_seg)
                else:
                    all_data.extend(ws_seg)
                    all_label.extend([int(seg_label)] * len(ws_seg))
                    all_feature.extend(feature_seg)

    print("Data shape: ", np.shape(all_data))
    print("Label shape: ", np.shape(all_label))
    print("Feature shape: ", np.shape(all_feature), "\n")

    print("Data test shape: ", np.shape(all_data_test))
    print("Label test shape: ", np.shape(all_label_test))
    print("Feature test shape: ", np.shape(all_feature_test))

    # Splitting and Training data
    print("Total samples of training data: {}".format(len(all_feature)))
    print("Total samples of testing data: {}".format(len(all_feature_test)))

    model_ml = RandomForestClassifier(n_estimators=500, n_jobs=-1)
    model_ml.fit(all_feature, all_label)

    joblib.dump(model_ml, "{}/randomforest_{}.joblib".format(SAVE_MODEL_PATH, VERSION))

    # Evaluating
    y_predict = model_ml.predict(all_feature_test)
    print(classification_report(all_label_test, y_predict))
    cm = confusion_matrix(all_label_test, y_predict, labels=np.unique(all_label_test))
    cm_norm = cm / np.sum(cm, axis=1, keepdims=True)
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', annot_kws={"fontsize": 6},
                yticklabels=np.unique(all_label_test), xticklabels=np.unique(all_label_test))
    plt.show()

def train(
    FOLDER_PATH = "/content/drive/MyDrive/STUDY MASTER/PROJECT/Hokkaido prj/Data/New CSV",
    ANN_PATH = "/content/drive/MyDrive/STUDY MASTER/PROJECT/Hokkaido prj/Data/Ann_CSV",
    SAVE_MODEL_PATH = "/content/drive/MyDrive/STUDY MASTER/PROJECT/Hokkaido prj/New_Model"
):
    train_data_list = [
        (user_id, ANN_PATH, FOLDER_PATH)
        for user_id in USER_ID
    ]

    test_data_list = [
        (user_id, ANN_PATH, FOLDER_PATH)
        for user_id in TEST_ID
    ]

    result = []
    # Training set
    pool = Pool(processes=10)
    res = pool.map(generate_data, train_data_list)
    result.extend(res)
    pool.close()
    pool.join()
    all_data, all_label, all_feature = [], [], []
    for i in range(len(result)):
        all_data.extend(result[i][0])
        all_label.extend(result[i][1])
        all_feature.extend(result[i][2])

    # Test set
    result_test = []
    pool = Pool(processes=10)
    res = pool.map(generate_data, test_data_list)
    result_test.extend(res)
    pool.close()
    pool.join()
    all_data_test, all_label_test, all_feature_test = [], [], []
    for i in range(len(result_test)):
        all_data_test.extend(result_test[i][0])
        all_label_test.extend(result_test[i][1])
        all_feature_test.extend(result_test[i][2])
    print("Shape of all data: ", np.shape(all_data))
    print("Shape of all label: ", np.shape(all_label))
    print("Shape of all feature: ", np.shape(all_feature))
    print("Shape of all data test: ", np.shape(all_data_test))
    print("Shape of all label test: ", np.shape(all_label_test))
    print("Shape of all feature test: ", np.shape(all_feature_test))

    print('Before SMOTE\n', sorted(Counter(all_label).items()))
    all_feature, all_label = SMOTE().fit_resample(all_feature, all_label)
    print('After SMOTE\n', sorted(Counter(all_label).items()))

    print("Total samples of training data: {}".format(len(all_feature)))
    print("Total samples of testing data: {}".format(len(all_feature_test)))

    model_ml = RandomForestClassifier(n_estimators=500, n_jobs=-1)
    model_ml.fit(all_feature, all_label)

    joblib.dump(model_ml, "{}/randomforest_{}.joblib".format(SAVE_MODEL_PATH, VERSION))

    # Evaluate
    # model_ml = joblib.load("{}/randomforest_{}.joblib".format(SAVE_MODEL_PATH, "v22"))

    # y_predict = model_ml.predict(all_feature_test)
    # print(classification_report(all_label_test, y_predict))
    # cm = confusion_matrix(all_label_test, y_predict, labels=np.unique(all_label_test))
    # cm_norm = cm / np.sum(cm, axis=1, keepdims=True)
    # plt.figure(figsize=(10, 9))
    # sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', annot_kws={"fontsize": 7.5},
    #             yticklabels=np.arange(TOTAL_CLASSESS), xticklabels=np.arange(TOTAL_CLASSESS))
    # # disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    # # disp.plot()
    # plt.show()

def test(
    FOLDER_PATH = "/content/drive/MyDrive/STUDY MASTER/PROJECT/Hokkaido prj/Data/New CSV",
    ANN_PATH = "/content/drive/MyDrive/STUDY MASTER/PROJECT/Hokkaido prj/Data/Ann_CSV",
    SAVE_MODEL_PATH = "/content/drive/MyDrive/STUDY MASTER/PROJECT/Hokkaido prj/New_Model"
):
    model_ml = joblib.load("{}/randomforest_{}.joblib".format(SAVE_MODEL_PATH, VERSION))
    WINDOW_SIZE = 2  # seconds


    total_prediction_micro_before = []
    total_prediction_micro_after = []
    total_label_micro = []

    total_prediction_macro_before = []
    total_prediction_macro_after = []
    total_label_macro = []

    for user_id in TEST_ID:
        # for user_id in ['S05T1']:
        print("----------------------------------------")
        print("Processing {}".format(user_id))
        front_keypoint_csv = FOLDER_PATH + "/Front_" + user_id + "_{}.csv".format(EXTENSION)
        back_keypoint_csv = FOLDER_PATH + "/Back_" + user_id + "_{}.csv".format(EXTENSION)
        ann_csv = ANN_PATH + "/Front_" + user_id + ".csv"
        front_kp_df, back_kp_df, ann_df = load_data_comb(front_keypoint_csv, back_keypoint_csv, ann_csv)
        timestamp_column = np.arange(len(front_kp_df))
        for i, kp_name in enumerate(front_kp_df.columns):
            front_kp_df.iloc[:, i] = smooth_kp(np.array(front_kp_df.iloc[:, i]))
            back_kp_df.iloc[:, i] = smooth_kp(np.array(back_kp_df.iloc[:, i]))
        front_ws_seg = segment(np.array(front_kp_df), max_time=len(front_kp_df), sub_window_size=WINDOW_SIZE * FS,
                               stride_size=int((WINDOW_SIZE - 0) * FS))
        front_ws_seg = skip_frames(front_ws_seg)

        back_ws_seg = segment(np.array(back_kp_df), max_time=len(back_kp_df), sub_window_size=WINDOW_SIZE * FS,
                              stride_size=int((WINDOW_SIZE - 0) * FS))
        back_ws_seg = skip_frames(back_ws_seg)

        front_feature_seg = [extract_feature(front_ws_seg[i], FS) for i in range(len(front_ws_seg))]
        back_feature_seg = [extract_feature(back_ws_seg[i], FS) for i in range(len(back_ws_seg))]
        # print("Shape of feature_seg: ", np.array(feature_seg).shape)

        timestamp_ws_seg = segment(np.array(timestamp_column), max_time=len(front_kp_df),
                                   sub_window_size=WINDOW_SIZE * FS,
                                   stride_size=int((WINDOW_SIZE - 0) * FS))  # version 22
        # print("shape of timestamp_ws_seg: ", np.shape(timestamp_ws_seg))
        timestamp_ws_seg = skip_frames(np.expand_dims(timestamp_ws_seg, axis=-1))  # version 22
        # print("shape of timestamp_ws_seg skip frame: ", np.shape(timestamp_ws_seg))
        timestamp_feature_seg = [extract_feature(timestamp_ws_seg[i], FS) for i in
                                 range(len(timestamp_ws_seg))]  # version 22
        # print("shape of timestamp_feature_seg: ", np.shape(timestamp_feature_seg))

        front_joint_angles = extract_joint_angles(np.array(front_kp_df))
        front_joint_angles_seg = segment(front_joint_angles, max_time=len(front_kp_df),
                                         sub_window_size=WINDOW_SIZE * FS,
                                         stride_size=int((WINDOW_SIZE - 0) * FS))
        front_joint_angles_seg = skip_frames(front_joint_angles_seg)  # version 20

        back_joint_angles = extract_joint_angles(np.array(back_kp_df))
        back_joint_angles_seg = segment(back_joint_angles, max_time=len(back_kp_df), sub_window_size=WINDOW_SIZE * FS,
                                        stride_size=int((WINDOW_SIZE - 0) * FS))
        back_joint_angles_seg = skip_frames(back_joint_angles_seg)  # version 20
        # print("Shape of joint_angles_seg: ", np.array(joint_angles_seg).shape)
        # print("Shape of joint_angles_seg: ", np.shape(joint_angles_seg))

        front_feature_joint_angles_seg = [extract_feature(front_joint_angles_seg[i], FS) for i in
                                          range(len(front_joint_angles_seg))]

        back_feature_joint_angles_seg = [extract_feature(back_joint_angles_seg[i], FS) for i in
                                         range(len(back_joint_angles_seg))]
        # print("Shape of feature_joint_angles_seg: ", np.array(feature_joint_angles_seg).shape)

        front_bone_length_seg = [extract_bone_length(front_ws_seg[i]) for i in range(len(front_ws_seg))]
        back_bone_length_seg = [extract_bone_length(back_ws_seg[i]) for i in range(len(back_ws_seg))]
        # print("Shape of bone_length_seg: ", np.array(bone_length_seg).shape)

        # print("Shape of bone_length_seg: ", np.shape(bone_length_seg))
        front_feature_bone_length = [extract_feature(front_bone_length_seg[i], FS) for i in
                                     range(len(front_bone_length_seg))]
        back_feature_bone_length = [extract_feature(back_bone_length_seg[i], FS) for i in
                                    range(len(back_bone_length_seg))]
        # print("Shape of feature_bone_length: ", np.array(feature_bone_length).shape)

        # print("Shape of velocity_seg: ", np.shape(velocity_seg))
        front_feature_velocity_seg = [extract_feature(extract_velocity(front_ws_seg[i]), FS) for i in
                                      range(len(front_ws_seg))]  # version 5
        back_feature_velocity_seg = [extract_feature(extract_velocity(back_ws_seg[i]), FS) for i in
                                     range(len(back_ws_seg))]  # version 5
        # print("Shape of feature_velocity_seg: ", np.array(feature_velocity_seg).shape)

        front_slow_motion_seg = [extract_slow_motion(front_ws_seg[i]) for i in range(len(front_ws_seg))]  # version 18
        back_slow_motion_seg = [extract_slow_motion(back_ws_seg[i]) for i in range(len(back_ws_seg))]  # version 18
        # print("Shape of slow_motion_seg: ", np.shape(slow_motion_seg))
        front_feature_slow_motion = [extract_feature(front_slow_motion_seg[i], FS) for i in
                                     range(len(front_slow_motion_seg))]  # version 18
        back_feature_slow_motion = [extract_feature(back_slow_motion_seg[i], FS) for i in
                                    range(len(back_slow_motion_seg))]  # version 18
        # print("Shape of feature_slow_motion: ", np.array(feature_slow_motion).shape)

        feature_seg = np.concatenate(
            [front_feature_seg, front_feature_joint_angles_seg, front_feature_velocity_seg, front_feature_bone_length,
             front_feature_slow_motion,
             back_feature_seg, back_feature_joint_angles_seg, back_feature_velocity_seg, back_feature_bone_length,
             back_feature_slow_motion, timestamp_feature_seg], axis=1)  # version 1, 2


        y_predict = model_ml.predict(feature_seg)

        start_time = np.array(ann_df['start_time'])
        stop_time = np.array(ann_df['stop_time'])
        ann = np.array(ann_df['annotation'])

        print("--------------Result in detailed activity before postprocessing--------------")
        # plot_prediction(y_predict, start_time, stop_time, ann, user_id, post_processing=False)
        dur = np.floor(stop_time - start_time)
        # print('dur: ', dur)
        all_label = [[ann[i]] * int(dur[i]) for i in range(len(ann))]
        all_label = sum(all_label, [])
        y_predict_repeat = np.repeat(y_predict, 2)
        if len(y_predict_repeat) < len(all_label):
            all_label = all_label[: len(y_predict_repeat)]
        total_prediction_micro_before.extend(y_predict_repeat)
        total_label_micro.extend(all_label)


        print("--------------Result in detailed activity after postprocessing--------------")
        y_predict_pp = post_processing(y_predict)
        # plot_prediction(y_predict_pp, start_time, stop_time, ann, user_id, post_processing=True)
        dur = np.floor(stop_time - start_time)
        # print('dur: ', dur)
        all_label = [[ann[i]] * int(dur[i]) for i in range(len(ann))]
        all_label = sum(all_label, [])
        y_predict_pp = np.repeat(y_predict_pp, 2)
        if len(y_predict_pp) < len(all_label):
            all_label = all_label[: len(y_predict_pp)]
        total_prediction_micro_after.extend(y_predict_pp)


        print("--------------Result in long activity before postprocessing--------------")
        y_predict_convert = np.repeat(y_predict, 2)
        y_predict_convert = convert_detail_2_long_act(y_predict_convert)
        all_label_convert = convert_detail_2_long_act(np.array(all_label))
        total_prediction_macro_before.extend(y_predict_convert)
        total_label_macro.extend(all_label_convert)

        print("--------------Result in long activity after postprocessing--------------")
        y_predict_convert = convert_detail_2_long_act(y_predict_pp)
        total_prediction_macro_after.extend(y_predict_convert)


    print("-------------- All Test Dataset: Result in micro activity before post-processing--------------")
    # report = classification_report(total_label_micro, total_prediction_micro_before, output_dict=True)
    print(classification_report(total_label_micro, total_prediction_micro_before))
    cm_norm = confusion_matrix(total_label_micro, total_prediction_micro_before, labels=np.unique(total_label_micro),
                               normalize='true')
    plt.figure(figsize=(10, 9))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', annot_kws={"fontsize": 7.5},
                yticklabels=np.unique(total_label_micro), xticklabels=np.unique(total_label_micro))
    plt.title("All Test Dataset - Detailed Activity before post-processing")
    plt.show()

    print("-------------- All Test Dataset: Result in micro activity after post-processing--------------")
    print(classification_report(total_label_micro, total_prediction_micro_after))
    cm_norm = confusion_matrix(total_label_micro, total_prediction_micro_after, labels=np.unique(total_label_micro),
                               normalize='true')
    plt.figure(figsize=(10, 9))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', annot_kws={"fontsize": 7.5},
                yticklabels=np.unique(total_label_micro), xticklabels=np.unique(total_label_micro))
    plt.title("All Test Dataset - Detailed Activity after post-processing")
    plt.show()

    print("-------------- All Test Dataset: Result in macro activity before post-processing --------------")
    print(classification_report(total_label_macro, total_prediction_macro_before))
    cm_long_norm = confusion_matrix(total_label_macro, total_prediction_macro_before,
                                    labels=np.unique(total_label_macro), normalize='true')
    # cm_norm_long = cm_long / np.sum(cm_long, axis=1, keepdims=True)
    plt.figure(figsize=(10, 9))
    sns.heatmap(cm_long_norm, annot=True, fmt='.2f', cmap='Blues', annot_kws={"fontsize": 7.5},
                yticklabels=np.unique(total_label_macro), xticklabels=np.unique(total_label_macro))
    plt.title("All Test Dataset - Combined Activity before post-processing")
    plt.show()

    print("-------------- All Test Dataset: Result in macro activity after post-processing --------------")
    # report = classification_report(total_label_macro, total_prediction_macro_before, output_dict=True)
    print(classification_report(total_label_macro, total_prediction_macro_after))
    cm_long_norm = confusion_matrix(total_label_macro, total_prediction_macro_after,
                                    labels=np.unique(total_label_macro), normalize='true')
    # cm_norm_long = cm_long / np.sum(cm_long, axis=1, keepdims=True)
    plt.figure(figsize=(10, 9))
    sns.heatmap(cm_long_norm, annot=True, fmt='.2f', cmap='Blues', annot_kws={"fontsize": 7.5},
                yticklabels=np.unique(total_label_macro), xticklabels=np.unique(total_label_macro))
    plt.title("All Test Dataset - Combined Activity after post-processing")
    plt.show()


if __name__ == "__main__":
    train()
    # test()