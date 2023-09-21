# -*- coding: utf-8 -*-
# Excerpt from: scn_detecta_08Aug2023.ipynb

import numpy as np
import cv2
import torch
import torchvision
import glob as glob
import os
import time
# #from google.colab.patches import cv2_imshow
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# UPLOAD_FOLDER = 'mysite/static/uploads'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# classes: 0 index is reserved for background
CLASSES = [
    '__background__', 'infected', 'healthy'
]

NUM_CLASSES = len(CLASSES)

# this will help us create a different color for each class
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
# COLORS = ['red', 'yellow', 'green', 'pink', 'blue']

# Getting the Model Ready

def create_model(num_classes):

    # load Faster RCNN pre-trained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # define a new head for the detector with required number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

# img = cv2.imread(UPLOAD_FOLDER+'/'+filename)

# load the best model and trained weights
model = create_model(num_classes=NUM_CLASSES)
checkpoint = torch.load('/home/jango23/mysite/static/uploads/data/output/model50.pth', map_location=DEVICE)
# model.load_state_dict(checkpoint['model_state_dict'])
model.load_state_dict(checkpoint, strict=False)
model.to(DEVICE).eval()

# directory where all the test images are present
# DIR_TEST = '/home/jango23/mysite/static/uploads/data/test'
DIR_TEST = '/home/jango23/mysite/static/uploads'
test_images = glob.glob(f"{DIR_TEST}/*")
print(f"Test instances: {len(test_images)}")

# define the detection threshold...
# ... any detection having score below this will be discarded
detection_threshold = 0.35

# to count the total number of images iterated through
frame_count = 0
# to keep adding the FPS for each image
total_fps = 0
for i in range(len(test_images)):
    # get the image file name for saving output later on
    image_name = test_images[i].split(os.path.sep)[-1].split('.')[0]
    image = cv2.imread(test_images[i])
    orig_image = image.copy()
    # BGR to RGB
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)
    # make the pixel range between 0 and 1
    image /= 255.0
    # bring color channels to front
    image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    # convert to tensor
    # image = torch.tensor(image, dtype=torch.float).cuda()
    image = torch.tensor(image, dtype=torch.float).cpu()
    # add batch dimension
    image = torch.unsqueeze(image, 0)
    start_time = time.time()
    with torch.no_grad():
        outputs = model(image.to(DEVICE))
    end_time = time.time()

    # get the current fps
    fps = 1 / (end_time - start_time)
    # add `fps` to `total_fps`
    total_fps += fps
    # increment frame count
    frame_count += 1
    # load all detection to CPU for further operations
    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
    # carry further only if there are detected boxes
    if len(outputs[0]['boxes']) != 0:
        boxes = outputs[0]['boxes'].data.numpy()
        scores = outputs[0]['scores'].data.numpy()
        # filter out boxes according to `detection_threshold`
        boxes = boxes[scores >= detection_threshold].astype(np.int32)
        draw_boxes = boxes.copy()
        # get all the predicted class names
        pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]

        # creating lists to count good and bad classes detected.
        class_healthy, class_infected = 0, 0

        # draw the bounding boxes and write the class name on top of it
        for j, box in enumerate(draw_boxes):
            class_name = pred_classes[j]

            # creating the bounding box in color
            color = COLORS[CLASSES.index(class_name)]
            cv2.rectangle(orig_image,
                          (int(box[0]), int(box[1])),
                          (int(box[2]), int(box[3])),
                          color, 2)
            cv2.putText(orig_image, class_name,
                        (int(box[0]), int(box[1] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color,
                        2, lineType=cv2.LINE_AA)

            # computing the value for classes detected
            if class_name == 'infected':
                class_infected += 1
            else:
                class_healthy += 1

        cv2.imwrite(f"/home/jango23/mysite/static/uploads/{image_name}_detected.jpg", orig_image)
    #print(f"{image_name} done...")
    #print(f"{len(boxes)} Detections made.\nHealthy: {class_healthy}\nInfected: {class_infected}")
    #print('-' * 50)

# print('TEST PREDICTIONS COMPLETE')
# cv2.destroyAllWindows()
# calculate and print the average FPS
avg_fps = total_fps / frame_count
# print(f"Average FPS: {avg_fps:.3f} | Detection_threshold = {detection_threshold}")

