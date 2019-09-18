import argparse
import os
import sys
import shutil
import json
import glob
import signal

import torch
import torch.nn as nn

from data_loader import VideoFolder
from model import ConvColumn
from torchvision.transforms import *
import numpy as np
import csv


config_path = "./configs/config.json"
files=open('test.csv','w', newline='')
writer=csv.writer(files, delimiter=';')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

best_prec1 = 0

# load config file
with open(config_path) as data_file:
    config = json.load(data_file)


def main():
    global best_prec1

    # set run output folder
    model_name = config["model_name"]
    output_dir = config["output_dir"]
    print("=> Output folder for this run -- {}".format(model_name))
    save_dir = os.path.join(output_dir, model_name)
    if not os.path.exists(save_dir):

        os.makedirs(save_dir)
        os.makedirs(os.path.join(save_dir, 'plots'))

    # adds a handler for Ctrl+C
    def signal_handler(signal, frame):
        """
        Remove the output dir, if you exit with Ctrl+C and
        if there are less then 3 files.
        It prevents the noise of experimental runs.
        """
        num_files = len(glob.glob(save_dir + "/*"))
        if num_files < 1:
            shutil.rmtree(save_dir)
        print('You pressed Ctrl+C!')
        sys.exit(0)
    # assign Ctrl+C signal handler
    signal.signal(signal.SIGINT, signal_handler)

    # create model
    model = ConvColumn(config['num_classes'])

    # multi GPU setting
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).to(device)

    # resume from a checkpoint
    if os.path.isfile(config['checkpoint']):
        print("=> loading checkpoint '{}'".format(config['checkpoint']))
        checkpoint = torch.load(config['checkpoint'])
        start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(config['checkpoint'], checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(
            config['checkpoint']))

    transform = Compose([
        CenterCrop(84),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])
    ])

    test_data = VideoFolder(root=config['val_data_folder'],
                            optical_flow_folder=config['optical_flow_folder'],
                            csv_file_input=config['val_data_csv'],
                            csv_file_labels=config['labels_csv'],
                            clip_size=config['clip_size'],
                            nclips=1,
                            step_size=config['step_size'],
                            is_val=True,
                            transform=transform,
                            )
    
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=config['batch_size'], shuffle=False,
        num_workers=config['num_workers'], pin_memory=True,
        drop_last=False)

    assert len(test_data.classes) == config["num_classes"]
    validate(test_loader, model, test_data.classes_dict)


def validate(val_loader, model, class_to_idx=None):
    # switch to evaluate mode
    model.eval()

    logits_matrix = []

    with torch.no_grad():
        for i, (folder, input) in enumerate(val_loader):
            input = input.to(device)
            # compute output
            output = model(input)
            output = output.detach().cpu().numpy()
            prediction = np.argmax(output, axis=1)
            prediction_lable = [class_to_idx[x] for x in prediction]
            result = np.vstack((folder, prediction_lable)).transpose()
            print(result)
            writer.writerows(result)
            logits_matrix.append(result)
            

        logits_matrix = np.concatenate(logits_matrix)
        print("-------------------------------------------")
        print(logits_matrix)
        print("-------------------------------------------\n")
    files.close()


if __name__ == '__main__':
    main()
