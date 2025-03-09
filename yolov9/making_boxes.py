import os
import cv2
from PIL import Image
from tqdm import tqdm
import torch
import json
import os
import argparse
import random
import sys

def handle_no_detection(directory_path, data_dict):
  
  directory_files = sorted(os.listdir(directory_path))
  box_list = {}

  for idx, imageDir in enumerate(directory_files):
    try:
      path = os.path.join(directory_path, imageDir)
      if idx not in data_dict.keys():
        boxes = []
        img = Image.open(path)
        width, height = img.size
        x1 = 0
        y1 = 0
        boxes.append([[x1, y1, width, height], 'random_box', 0.51])
        box_list[idx] = boxes
      else:
        box_list[idx] = data_dict[idx]
    except Exception as e:
      print(e)

  return box_list

def get_merged_bounding_box(path, boxes_json, k=10):
  listClassNames = []        
  listCrop = []
  listProb = []

  for box, class_box, prob in boxes_json:
    listCrop.append(box)
    listClassNames.append(class_box)
    listProb.append(prob)

  img = Image.open(path)

  if len(listCrop) < 2:
    sorted_boxes = sorted(boxes_json, key=lambda x: x[2], reverse=True)[:2]
    for box, class_box, prob in sorted_boxes:
      listCrop.append(box)
      listClassNames.append(class_box)
      listProb.append(prob)
  
  ### change
  img = Image.open(path)
  width, height = img.size
  x1 = 0
  y1 = 0
  listCrop.append([x1, y1, width, height])
  listClassNames.append('random_box')
  listProb.append(0.51)
  
  if len(listCrop) < 2:
    width, height = img.size
    for _ in range(2 - len(listCrop)):
      img = Image.open(path)
      width, height = img.size
      x1 = 0
      y1 = 0
      listCrop.append([x1, y1, width, height])
      listClassNames.append('random_box')
      listProb.append(0.51)

  boxes = []
  for index1,box1 in enumerate(listCrop):
    class1 = listClassNames[index1]
    for class2,box2,prob2 in zip(listClassNames[index1+1:], listCrop[index1+1:],listProb[index1+1:]):
      x1 = min(box1[0],box2[0])
      y1 = min(box1[1],box2[1])
      x2 = max(box1[2],box2[2])
      y2 = max(box1[3],box2[3])
      # remove box1, box2 if needed
      boxes.append(([x1,y1,x2,y2], box1, box2, class1, class2, listProb[index1]*prob2))

  boxes = sorted(boxes, key=lambda x: x[-1], reverse=True)
  return boxes[0:k]

def get_relation_boxes_in_json(directory_path, data_dict):
  
  directory_files = sorted(os.listdir(directory_path))
  box_list = {}

  for idx, imageDir in enumerate(directory_files):
    try:
      path = os.path.join(directory_path, imageDir)
      boxes = get_merged_bounding_box(path, data_dict[idx])  
      box_list[idx] = boxes
    except Exception as e:
      print(e)

  return box_list