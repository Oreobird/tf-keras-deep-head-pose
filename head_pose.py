import tensorflow as tf
import os
import numpy as np
import cv2
import scipy.io as sio
import dlib
from imutils import face_utils

import datasets
import utils
import models

PROJECT_DIR = "E:/demo/python/head_pose/"

AFLW2000_DATA_DIR = 'E:/data/AFLW2000/'
AFLW2000_MODEL_FILE = PROJECT_DIR + 'model/aflw2000_model.h5'
AFLW2000_TEST_SAVE_DIR = 'E:/ml/data/aflw2000_test/'

BIWI_DATA_DIR = 'E:/ml/data/Biwi/kinect_head_pose_db/hpdb/'
BIWI_MODEL_FILE = PROJECT_DIR + 'model/biwi_model.h5'
BIWI_TEST_SAVE_DIR = 'E:/ml/data/biwi_test/'

face_landmark_path = PROJECT_DIR + 'model/shape_predictor_68_face_landmarks.dat'

BIN_NUM = 66
INPUT_SIZE = 64
BATCH_SIZE=16
EPOCHS=20

dataset = datasets.Biwi(BIWI_DATA_DIR, 'filename_list.txt', batch_size=BATCH_SIZE, input_size=INPUT_SIZE, ratio=0.95)

net = models.AlexNet(dataset, BIN_NUM, batch_size=BATCH_SIZE, input_size=INPUT_SIZE)

net.train(BIWI_MODEL_FILE, max_epoches=EPOCHS, load_weight=True)

# net.test(BIWI_TEST_SAVE_DIR)


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Unable to connect to camera.")
    exit(-1)
    
    
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(face_landmark_path)

frames = []
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        face_rects = detector(frame, 0)
        if len(face_rects) > 0:
            shape = predictor(frame, face_rects[0])
            shape = face_utils.shape_to_np(shape)

            face_crop = utils.crop_face_loosely(shape, frame, INPUT_SIZE)
            
            frames.append(face_crop)
            if len(frames) == 1:
                print(shape[30])
                pred_cont_yaw, pred_cont_pitch, pred_cont_roll = net.test_online(frames)
                
                cv2_img = utils.draw_axis(frame, pred_cont_yaw, pred_cont_pitch, pred_cont_roll, tdx=shape[30][0],
                                          tdy=shape[30][1], size=100)
                cv2.imshow("cv2_img", cv2_img)
                frames = []
                
            # cv2.waitKey(0)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break