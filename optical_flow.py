import cv2
import numpy as np
import os
import sys
from natsort import natsorted
dataset_path = "../gesture_recognition"
original_dataset_folder = os.path.join(dataset_path, "20bn-jester-v1")
target_dataset_folder = os.path.join(dataset_path, "20bn-jester-v1_optical_flow")


def images(folder_name=""):
    for image_name in sorted(os.listdir(folder_name)):
        image = cv2.imread(os.path.join(folder_name, image_name))
        yield image


def calculate_optical_flow(folder_name):
    target_folder = os.path.join(target_dataset_folder, folder_name)
    print("current:{}".format(folder_name))
    if not os.path.exists(target_folder):
        os.mkdir(target_folder)
    else:
        return
    itr = images(os.path.join(original_dataset_folder, folder_name))
    frame1 = next(itr)
    previous_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255
    count = 1
    while True:
        try:
            frame2 = next(itr)
            next_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(previous_frame, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            target = "{}/{}/{.4}.jpg".format(target_dataset_folder, folder_name, count)
            cv2.imwrite(target, rgb)
            count = count + 1
            previous_frame = next_frame
        except StopIteration:
            return


if __name__ == "__main__":
    for folder_name in natsorted(os.listdir(original_dataset_folder)):
        calculate_optical_flow(folder_name)