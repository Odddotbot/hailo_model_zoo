'''Contains core functions for this project, which help to parse, evaluate, and compare. predictions from Hailo and Ultralytics models.'''

import pickle
import torch
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import numpy as np
import pandas as pd
from ultralytics.utils.metrics import ConfusionMatrix


def get_dict(tensor):
    return {
        'bboxes': tensor[:,:4],  
        'conf': tensor[:,4], 
        'cls': tensor[:,5]}

def get_annotations(filepath):
    with open(filepath, 'rb') as file:
        annotations = pickle.load(file)
    for annotation in annotations:
        annotation[:,:4] = annotation[:,:4] / 448 * 2144
        if len(annotation) > 0:
            annotation[:,4] = 1.0 # Always 1.0 confidence
    return annotations

def get_ultra_preds(filepath, resize=False):
    with open(filepath, 'rb') as file:
        predictions = pickle.load(file)
    for pred in predictions:
        if resize:
            pred[:,:4] = pred[:,:4] / 448 * 2144
    predictions = [pred.to('cpu') for pred in predictions]
    return predictions

def get_hailo_preds(filepath):
    with open(filepath, 'rb') as file:
        output_list = pickle.load(file)
    hailo_output = []
    for img in output_list:
        weed_detections = img[0]
        weed_detections = np.column_stack((weed_detections, np.zeros(shape=(weed_detections.shape[0]),dtype=np.float32)))
        crop_detections = img[1]
        crop_detections = np.column_stack((crop_detections, np.ones(shape=(crop_detections.shape[0]),dtype=np.float32)))
        img_detections = np.concatenate((weed_detections, crop_detections))
        img_detections[:,:4] = img_detections[:,:4]*2144
        img_detections = img_detections[:,[1,0,3,2,4,5]]
        img_detections = torch.tensor(img_detections)
        hailo_output.append(img_detections)
    return hailo_output

def get_image_paths(dataset_path):
    path = Path(dataset_path)
    return [path / img for img in sorted(os.listdir(path)) if img.endswith('.jpg')]

def visualize(img, preds_dict, refs_dict):
    classes = ["weed", "crop"]
    img = cv2.imread(img)
    fig, ax = plt.subplots(figsize=(14,14))
    ax.imshow(img)
    for bbox, conf, cls in zip(refs_dict['bboxes'], refs_dict['conf'], refs_dict['cls']):
        xmin, ymin, xmax, ymax = bbox
        w = (xmax - xmin)
        h = (ymax - ymin)
        color = 'b' if cls == 0 else 'g'
        rect = patches.Rectangle((xmin,ymin), w, h, linewidth=1, edgecolor=color, facecolor='none')
        ax.text(xmin, ymin + 1, f'Ref: {round(conf.item(),3)} {classes[int(cls)]}')
        ax.add_patch(rect)
    for bbox, conf, cls in zip(preds_dict['bboxes'], preds_dict['conf'], preds_dict['cls']):
        xmin, ymin, xmax, ymax = bbox
        w = (xmax - xmin)
        h = (ymax - ymin)
        color = 'r' if cls == 0 else 'y'
        rect = patches.Rectangle((xmin,ymin), w, h, linewidth=1, edgecolor=color, facecolor='none')
        ax.text(xmax, ymin + 1, f'Pred: {round(conf.item(),3)} {classes[int(cls)]}', ha='right')
        ax.add_patch(rect)
    ax.axis('off')
    plt.tight_layout()
    plt.show()

def evaluate(predictions, references, filepaths, vis_error_th=None):
    cm = ConfusionMatrix(['WeedBox', 'CropBox'])
    for i in range(len(filepaths)):
        preds, refs = predictions[i], references[i]
        preds, refs = get_dict(preds), get_dict(refs)
        matrix_before = cm.matrix.copy()
        cm.process_batch(preds, refs, conf=0.4, iou_thres=0.8)
        matrix_after = cm.matrix.copy()
        batch_mistakes = matrix_after - matrix_before
        batch_mistakes[0,0] = 0
        batch_mistakes[1,1] = 0
        # batch_mistakes[:,2] = 0
        # batch_mistakes[2,:] = 0
        if vis_error_th is not None and batch_mistakes.sum() > vis_error_th:
            print(batch_mistakes)
            visualize(filepaths[i], preds, refs)
    return cm

def calculate_metrics(cm):
    matrix = cm.matrix
    crop_tp = cm.matrix[1,1]
    crop_fp = cm.matrix[1,0] + cm.matrix[1,2]
    crop_fn = cm.matrix[0,1] + cm.matrix[2,1]
    weed_tp = cm.matrix[0,0]
    weed_fp = cm.matrix[0,1] + cm.matrix[0,2]
    weed_fn = cm.matrix[1,0] + cm.matrix[2,0]
    crop_precision = crop_tp / (crop_tp + crop_fp)
    weed_precision = weed_tp / (weed_tp + weed_fp)
    crop_recall = crop_tp / (crop_tp + crop_fn)
    weed_recall = weed_tp / (weed_tp + weed_fn)
    avg_precision = (crop_precision + weed_precision) / 2
    avg_recall = (crop_recall + weed_recall) / 2
    accuracy = (matrix[0][0] + matrix[1][1]) / matrix.sum()
    conf_mae = torch.stack(cm.conf_abs_errors).mean().item() if len(cm.conf_abs_errors) > 0 else 0
    results = pd.DataFrame({'Accuracy': [accuracy], 'Precision': [avg_precision], 'Recall': [avg_recall], 'Conf. MAE': [conf_mae]})
    return results

def save_results(cm, results_dir):
    results_dir = Path(results_dir) if type(results_dir) != Path else results_dir
    if not results_dir.exists():
        os.mkdir(results_dir)
    results = calculate_metrics(cm)
    results.to_csv(results_dir / 'results.csv')
    cm.plot(normalize=True, save_dir=results_dir)
    cm.plot(normalize=False, save_dir=results_dir)