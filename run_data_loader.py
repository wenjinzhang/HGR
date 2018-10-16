import os
import glob
import numpy as np
import torch
from PIL import Image
from data_parser import JpegDataset
from torchvision.transforms import *
import json
from model import ConvColumn


# model.eval()
IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG']
with open('./configs/config2.json') as data_file:
    config = json.load(data_file)

class RunTimeDataSet(torch.utils.data.Dataset):

    dataset_object = JpegDataset(config['train_data_csv'], config['labels_csv'], config['train_data_folder'])

    # Arrya for imgs
    IMGS_Array = []

    def __init__(self):
        self.transform = Compose([
                        CenterCrop(84),
                        ToTensor(),
                        Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
                        ])
        self.clip_size = 18
        self.nclips = 1
        self.step_size = 2
        self.is_val = False

    def __getitem__(self, index):
        img_paths = self.get_frame_names('123292')
        imgs = []
        for img in self.IMGS_Array:
            # img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
            imgs.append(torch.unsqueeze(img, 0))

        # format data to torch
        data = torch.cat(imgs)
        data = data.permute(1, 0, 2, 3)
        return data

    def __len__(self):
        return 1


    def get_frame_names(self, path):
        frame_names = []
        frame_names.extend(
            glob.glob(os.path.join('/home/wenjin/Documents/pycharmworkspace/20bn-jester-v1/' + path, "*" + '.jpg')))
        frame_names = list(sorted(frame_names))
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
    # load config file
    with open('./configs/config2.json') as data_file:
        config = json.load(data_file)
    device = torch.device("cuda")

    # init model
    # create model
    model = ConvColumn(config['num_classes'])
    # multi GPU setting
    model = torch.nn.DataParallel(model).to(device)
    checkpoint = torch.load(config['checkpoint'])
    start_epoch = checkpoint['epoch']
    best_prec1 = checkpoint['best_prec1']
    model.load_state_dict(checkpoint['state_dict'])
    loader = RunTimeDataSet()
    # save_images_for_debug("input_images", data_item.unsqueeze(0))
    val_loader = torch.utils.data.DataLoader(
        loader,
        batch_size=1, shuffle=False,
        num_workers=1, pin_memory=True)

    with torch.no_grad():
        for i, input in enumerate(val_loader):
            input = input.to(device)
            # compute output
            output = model(input)
            label_number = np.argmax(output.detach().cpu().numpy()[0])
            print(label_number)
            label_name = loader.dataset_object.classes_dict[label_number]
            print("output={}==name==={}".format(label_number, label_name))






