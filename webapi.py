from flask import Flask, render_template, request, jsonify
import base64
import time
from PIL import Image
from io import BytesIO
import json
from model import ConvColumn
from torchvision.transforms import *
import torch
import numpy as np
from data_parser import JpegDataset
from run_data_loader import RunTimeDataSet
import os
import glob
import torch.nn as nn

app = Flask('__HGR__')

IMGS_Array = []
# load config file
with open('./configs/config2.json') as data_file:
    config = json.load(data_file)
device = torch.device("cuda")

dataset_object = JpegDataset(config['train_data_csv'], config['labels_csv'], config['train_data_folder'])

label_dict = dataset_object.classes_dict

transform = Compose([
        CenterCrop(84),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])
    ])

loader = RunTimeDataSet()
    # save_images_for_debug("input_images", data_item.unsqueeze(0))
val_loader = torch.utils.data.DataLoader(
        loader,
        batch_size=1, shuffle=False,
        num_workers=1, pin_memory=True)

#init model
# create model
model = ConvColumn(config['num_classes'])
# multi GPU setting
model = torch.nn.DataParallel(model).to(device)
checkpoint = torch.load(config['checkpoint'])
start_epoch = checkpoint['epoch']
best_prec1 = checkpoint['best_prec1']
model.load_state_dict(checkpoint['state_dict'])
# model.eval()


def model_caculate(input):
    # compute the model
    with torch.no_grad():
        input = input.to(device)
        out = model(input)
        print(out.detach().cpu().numpy())
        label_number = np.argmax(out.detach().cpu().numpy())
        label = label_dict[label_number]
        print(label_number)
        print(label)
    return label, label_number


# input 18+2 shape()img for three input and we can output the high ;
def recognize(array_img):
    # normalize teh img;
    # pre-op data
    data = []
    for img in array_img:
        img = transform(img)
        data.append(torch.unsqueeze(img, 0))

    # data shape=(20,84,84,3)
    #input = [data[0:17], data[1:18], data[2: 19]]
    # input shape=(3,18,84,84,3)

    input = torch.cat(data)
    print(input.size())
    input = input.permute(1, 0, 2, 3)
    input = [input*15]
    input.resize_(15, 3, 18, 84, 84)
    label,label_number = model_caculate(input)
    return label, label_number



def get_frame_names(path):
    frame_names = []
    frame_names.extend(glob.glob(os.path.join('/home/wenjin/Documents/pycharmworkspace/20bn-jester-v1/'+path, "*" + '.jpg')))
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


def recognition2():
    with torch.no_grad():
        for i, input in enumerate(val_loader):
            input = input.to(device)
            # compute output
            output = model(input)
            label_number = np.argmax(output.detach().cpu().numpy()[0])
            print(label_number)
            label_name = loader.dataset_object.classes_dict[label_number]
            print("output={}==name==={}".format(label_number, label_name))
    

@app.route("/receive", methods=['GET', 'POST'])
def receive_img():
    data = request.get_json()
    imgdata = base64.b64decode(data['imageBase64'])
    file = open('images/{}.png'.format(len(IMGS_Array) % 18), 'wb')
    file.write(imgdata)
    file.close()
    # save imgs
    image_data = BytesIO(imgdata)
    imgdata = Image.open(image_data).convert('RGB')
    # insert images
    IMGS_Array.append(imgdata)
    result_data = {}
    if len(IMGS_Array) % 18 == 0:
        # recognize
        label, label_number = recognize(IMGS_Array)

        result_data['result'] = 'success'
        result_data['info'] = label
        del IMGS_Array[:]
    else:
        # cannot recognize
        result_data['result'] = 'fail'
        result_data['info'] = 'don\'t have enough numbers'
    return jsonify(result=result_data)


@app.route("/test/<foldname>", methods=['GET', 'POST'])
def test(foldname):
    fram_names = get_frame_names(''+foldname)
    imgs = []
    for path in fram_names:
        imgdata = Image.open(path).convert('RGB')
        imgdata = transform(imgdata)
        imgs.append(torch.unsqueeze(imgdata, 0))

    # format data to torch
    data = torch.cat(imgs)
    data = data.permute(1, 0, 2, 3)
    data.resize_(1, 3, 18, 84, 84)
    label, label_number = model_caculate(data)
    print(label)
    return 'label:{},<br>lable_number:{}'.format(label, label_number)


@app.route("/")
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, ssl_context='adhoc')










