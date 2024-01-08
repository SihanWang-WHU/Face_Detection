import numpy as np
import torch
import torchvision
import sys
sys.path.append("./detection")
import cv2
import random
import time
import datetime
from model import get_model_instance_segmentation

def random_color():
    b = random.randint(0, 255)
    g = random.randint(0, 255)
    r = random.randint(0, 255)

    return (b, g, r)


def toTensor(img):
    assert type(img) == np.ndarray, 'the img type is {}, but ndarry expected'.format(type(img))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    return img.float().div(255)  # 255也可以改为256

def process_video(video_path, model, device):
    cap = cv2.VideoCapture(video_path)
    bbox_list = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        bbox_info = PredictImg(frame, model, device)
        bbox_list.extend(bbox_info)

    cap.release()
    return bbox_list

def save_bbox_to_txt(bbox_list, file_path):
    with open(file_path, 'w') as f:
        for bbox in bbox_list:
            f.write(f"{bbox}\n")

def PredictImg(img, model, device):
    img_tensor = toTensor(img)
    prediction = model([img_tensor.to(device)])
    boxes = prediction[0]['boxes']
    labels = prediction[0]['labels']
    scores = prediction[0]['scores']

    bbox_info = []
    for idx in range(boxes.shape[0]):
        if scores[idx] >= 0.8:
            bbox = boxes[idx].cpu().detach().numpy().astype(int)
            label = labels[idx].item()
            bbox_info.append((label, bbox.tolist()))

    return bbox_info


if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_classes = 2
    model = get_model_instance_segmentation(num_classes)
    model.to(device)
    model.eval()
    model.load_state_dict(torch.load('model.pth'))

    start_time = time.time()
    bbox_list = process_video('test.mp4', model, device)
    save_bbox_to_txt(bbox_list, 'bboxes.txt')
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Processing time {}'.format(total_time_str))