import os
import glob
import numpy as np
import torch

from PIL import Image
from data_parser import JpegDataset
from torchvision.transforms import *
from natsort import natsorted
IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG']
cache_path = "cache"


def default_loader(path):
    return Image.open(path).convert('RGB')


class VideoFolder(torch.utils.data.Dataset):

    def __init__(self, root, optical_flow_folder, csv_file_input, csv_file_labels, clip_size,
                 nclips, step_size, is_val=False, transform=None,
                 loader=default_loader, include_optical_flow=False, is_label=True):
        self.dataset_object = JpegDataset(csv_file_input, csv_file_labels, root, optical_flow_folder, is_label)

        self.csv_data = self.dataset_object.csv_data
        self.classes = self.dataset_object.classes
        self.classes_dict = self.dataset_object.classes_dict
        self.root = root
        self.transform = transform
        self.loader = loader

        self.clip_size = clip_size
        self.nclips = nclips
        self.step_size = step_size
        self.is_val = is_val
        self.include_optical_flow = include_optical_flow
        self.is_label = is_label

    # try to add cache, don't process one item twice.
    def __getitem__(self, index):
        item = self.csv_data[index]
        img_paths = self.get_frame_names(item.path)
        optical_flow_paths = self.get_frame_names(item.optical_flow_path)
        imgs = []
        optical_flows = []

        for i in range(len(img_paths)):
            img = self.loader(img_paths[i])
            img = self.transform(img)
            imgs.append(torch.unsqueeze(img, 0))

            optical_flow = self.loader(optical_flow_paths[i])
            optical_flow = self.transform(optical_flow)
            optical_flows.append(torch.unsqueeze(optical_flow, 0))

        # for img_path in img_paths:
        #     img = self.loader(img_path)
        #     img = self.transform(img)
        #     imgs.append(torch.unsqueeze(img, 0))

        # format data to torch
        imgs = torch.cat(imgs)
        optical_flows =torch.cat(optical_flows)
        data = torch.cat((imgs, optical_flows), 1)
        data = data.permute(1, 0, 2, 3)
        
        return (data, self.classes_dict[item.label]) if self.is_label else (item[0], data)

    def __len__(self):
        return len(self.csv_data)

    def get_frame_names(self, path):
        frame_names = []
        for ext in IMG_EXTENSIONS:
            frame_names.extend(glob.glob(os.path.join(path, "*" + ext)))

        frame_names = list(natsorted(frame_names))
        num_frames = len(frame_names)

        # set number of necessary frames
        if self.nclips > -1:
            num_frames_necessary = self.clip_size * self.nclips * self.step_size
        else:
            num_frames_necessary = num_frames

        # pick frames
        offset = 0
        if num_frames_necessary > num_frames:
            # pad last frame if video is shorter than necessary
            frame_names += [frame_names[-1]] * (num_frames_necessary - num_frames)
        elif num_frames_necessary < num_frames:
            # If there are more frames, then sample starting offset
            diff = (num_frames - num_frames_necessary)
            # Temporal augmentation
            if not self.is_val:
                offset = np.random.randint(0, diff)
        frame_names = frame_names[offset:num_frames_necessary +
                                  offset:self.step_size]

        return frame_names


def get_frame_names(path):
    frame_names = []
    frame_names.extend(glob.glob(os.path.join('/home/wenjin/Documents/pycharmworkspace/20bn-jester-v1/'+path, "*" + '.jpg')))
    frame_names = list(natsorted(frame_names))
    num_frames = len(frame_names)
    num_frames_necessary = 36
    # pick frames
    offset = 0
    if num_frames_necessary > num_frames:
        # pad last frame if video is shorter than necessary
        frame_names += [frame_names[-1]] * (num_frames_necessary - num_frames)

    frame_names = frame_names[0:num_frames_necessary:2]
    return frame_names


if __name__ == '__main__':
    transform = Compose([
                        CenterCrop(84),
                        ToTensor(),
                        Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
                        ])
    loader = VideoFolder(root="D://pycharmproject//gesture_recognition//20bn-jester-v1",
                         optical_flow_folder="D://pycharmproject//gesture_recognition//20bn-jester-v1_optical_flow",
                         csv_file_input="./20bn-jester-v1/annotations/jester-v1-test.csv",
                         csv_file_labels="./20bn-jester-v1/annotations/jester-v1-labels.csv",
                         clip_size=18,
                         nclips=1,
                         step_size=2,
                         is_val=True,
                         transform=transform,
                         loader=default_loader)

    data_item = loader[0]

    # save_images_for_debug("input_images", data_item.unsqueeze(0))
    print(data_item)
    print(data_item.shape)

    train_loader = torch.utils.data.DataLoader(
        loader,
        batch_size=5, shuffle=False,
        num_workers=5, pin_memory=True)

