# Copyright Odd.Bot 2025, Luuk Romeijn (luuk@odd.bot)

'''Validation of model performance during model conversion stages.'''

import os
import yaml

import cv2
from hailo_sdk_client import ClientRunner, InferenceContext
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import ultralytics
from ultralytics.utils.metrics import ConfusionMatrix
import torch


CLASSES = ['weed', 'crop']
SIMILARITY_METRICS = ['matched', 'class confused', 'missed', 'novel', 'accuracy']


def conversion_validation(pt_filepath: str, har_filepath: str, data_yaml: str, imgsize: tuple[int], ground_truth_src: str, hw_arch: str, nms_scores_th: float, nms_iou_th: float, similarity_th: float, vis_error_th: float, val_iou_th: float, results_dir: str) -> tuple[bool, str, float]:
    # TODO: Not sure if all the variable names are like super duper interintuitive

    # Loading validation data
    img_list = load_validation_data(data_yaml, imgsize)

    if ground_truth_src == 'sly':
        ground_truth = get_labels(data_yaml, imgsize)
    else: # ground_truth_src == 'pt'
        ground_truth = infer_ultralytics(pt_filepath, img_list, nms_scores_th, nms_iou_th)
    
    onnx_filepath = pt_filepath.replace('.pt', '.onnx')
    conversion_results = {
        'PyTorch': infer_ultralytics(pt_filepath, img_list, nms_scores_th, nms_iou_th),
        'ONNX': infer_ultralytics(onnx_filepath, img_list, nms_scores_th, nms_iou_th),
        'FP optimized': infer_hailo(har_filepath, img_list, nms_scores_th, nms_iou_th, imgsize, InferenceContext.SDK_FP_OPTIMIZED, hw_arch),
        'Quantized': infer_hailo(har_filepath, img_list, nms_scores_th, nms_iou_th, imgsize, InferenceContext.SDK_QUANTIZED, hw_arch),
        'Hardware (emulated)': infer_hailo(har_filepath, img_list, nms_scores_th, nms_iou_th, imgsize, InferenceContext.SDK_BIT_EXACT, hw_arch)
    }

    similarity_data = calculate_prediction_similarity(ground_truth, conversion_results, val_iou_th, vis_error_th, img_list, results_dir)
    plot_results(similarity_data, conversion_results.keys(), results_dir)
    for i, stage in enumerate(conversion_results):
        similarity = similarity_data[i,-1] / 100
        if similarity <= similarity_th:
            return_msg = f"FAILED conversion validation at '{stage}' stage, similarity to .pt model ({np.round(similarity, 4)}) below threshold ({similarity_th})."
            return False, return_msg
        
    return_msg = f"SUCCESS in conversion validation (similarity={np.round(similarity, 4)})."
    return True, return_msg, similarity


def load_validation_data(data_yaml: str, imgsize: int) -> np.ndarray:
    with open(data_yaml) as data_yaml_file:
        data_config = yaml.safe_load(data_yaml_file) # TODO: this actually has the number/names/order of classes
    val_data_dirs = data_config['val']
    val_data_dirs = [f"/datasets/{dir.split('/datasets/')[1]}" for dir in val_data_dirs] # only /datasets is mounted in Docker
    # img_paths = [f'{dir}/images/train/{image_path}' for dir in val_data_dirs for image_path in sorted(os.listdir(f'{dir}/images/train'))] # TODO remove this line (was for Germany testing)
    img_paths = [f'{dir}/images/val/{image_path}' for dir in val_data_dirs for image_path in sorted(os.listdir(f'{dir}/images/val'))]
    img_list = [cv2.imread(img_path) for img_path in img_paths]
    img_list = [cv2.resize(img, imgsize) for img in img_list]
    return img_list


def get_labels(data_yaml: str, imgsize: tuple[int]):
    with open(data_yaml) as data_yaml_file:
        data_config = yaml.safe_load(data_yaml_file)
    val_data_dirs = data_config['val']
    val_data_dirs = [f"/datasets/{dir.split('/datasets/')[1]}" for dir in val_data_dirs]
    label_paths = [f'{dir}/labels/val/{image_path}' for dir in val_data_dirs for image_path in sorted(os.listdir(f'{dir}/labels/val'))]
    label_list = []
    for label_path in label_paths:
        labels = np.loadtxt(label_path)
        confidence_placeholder = np.ones(len(labels))
        labels = np.column_stack((labels, confidence_placeholder))
        labels = labels[:,[1,2,3,4,5,0]]
        labels[:,[0,2]] = imgsize[0]*labels[:,[0,2]]
        labels[:,[1,3]] = imgsize[1]*labels[:,[1,3]]
        labels[:,0] = labels[:,0] - 0.5*labels[:,2] # xmin
        labels[:,2] = labels[:,0] + labels[:,2] # xmax
        labels[:,1] = labels[:,1] - 0.5*labels[:,3] # ymin
        labels[:,3] = labels[:,1] + labels[:,3] # ymax
        labels = torch.tensor(labels)
        label_list.append(labels)
    return label_list


def format_hailo_prediction(raw_result: list[np.ndarray], class_idx: int) -> np.ndarray:
    result = raw_result[class_idx].transpose()
    result = result[result.sum(axis=1) != 0]
    result = np.column_stack((result, class_idx*np.ones(shape=(result.shape[0]), dtype=np.float32)))
    return result


def infer_ultralytics(pt_or_onnx_filepath: str, img_list: list[np.ndarray], conf: float, iou: float) -> list[torch.Tensor]:
    model = ultralytics.models.yolo.model.YOLO(pt_or_onnx_filepath, task='detect')
    results = [model.predict(img, conf=conf, iou=iou) for img in img_list]
    results = [result[0].boxes.data.to('cpu') for result in results]
    return results


def infer_hailo(har_filepath: str, img_list: list[np.ndarray], conf: float, iou: float, imgsize: tuple[int], inference_context: InferenceContext, hw_arch: str) -> list[torch.Tensor]:
    
    imgs_array = np.stack(img_list)[:,:,:,[2,1,0]] # Switch BGR to RGB for Hailo
    img_width, img_height = imgsize
    runner = ClientRunner(har=har_filepath, hw_arch=hw_arch)
    runner._sdk_backend.nms_metadata.nms_config.nms_scores_th = conf
    runner._sdk_backend.nms_metadata.nms_config.nms_iou_th = iou

    with runner.infer_context(inference_context) as ctx:
        raw_results = runner.infer(ctx, imgs_array)

    results = []
    for raw_result in raw_results:
        weed_result = format_hailo_prediction(raw_result, 0)
        crop_result = format_hailo_prediction(raw_result, 1)
        result = np.concatenate((weed_result, crop_result))
        result = result[:,[1,0,3,2,4,5]]
        result[:,[0,2]] = result[:,[0,2]]*img_width
        result[:,[1,3]] = result[:,[1,3]]*img_height
        result = torch.tensor(result)
        results.append(result)

    return results


def get_dict(tensor: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        'bboxes': tensor[:,:4],  
        'conf': tensor[:,4], 
        'cls': tensor[:,5]}


def xyxy_to_xywh(xyxy_bbox: tuple[int]) -> tuple[int]:
    xmin, ymin, xmax, ymax = xyxy_bbox
    w = (xmax - xmin)
    h = (ymax - ymin)
    return xmin, w, ymin, h


def visualize(img: np.ndarray, preds_dict: dict[str, torch.Tensor], refs_dict: dict[str, torch.Tensor], filepath: str):
    fig, ax = plt.subplots(figsize=(14,14))
    ax.imshow(img)
    for bbox, conf, cls in zip(refs_dict['bboxes'], refs_dict['conf'], refs_dict['cls']):
        xmin, w, ymin, h = xyxy_to_xywh(bbox)
        color = 'g' if cls == 0 else 'y'
        rect = patches.Rectangle((xmin,ymin), w, h, linewidth=5, edgecolor=color, facecolor='none')
        ax.text(xmin, ymin + 1, f'Ref: {round(conf.item(),3)} {CLASSES[int(cls)]}')
        ax.add_patch(rect)
    for bbox, conf, cls in zip(preds_dict['bboxes'], preds_dict['conf'], preds_dict['cls']):
        xmin, w, ymin, h = xyxy_to_xywh(bbox)
        color = 'r' if cls == 0 else 'b'
        rect = patches.Rectangle((xmin,ymin), w, h, linewidth=1, edgecolor=color, facecolor='none')
        ax.text(xmin + w, ymin + 1, f'Pred: {round(conf.item(),3)} {CLASSES[int(cls)]}', ha='right')
        ax.add_patch(rect)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()


def evaluate(predictions: list[torch.Tensor], references: list[torch.Tensor], iou_th: float, vis_error_th: int=None, img_list: list[np.ndarray]=None, prefix: str=None) -> ConfusionMatrix:
    cm = ConfusionMatrix(CLASSES)
    for i in range(len(references)):
        preds, refs = predictions[i], references[i]
        preds, refs = get_dict(preds), get_dict(refs)
        matrix_before = cm.matrix.copy()
        cm.process_batch(preds, refs, conf=0.4, iou_thres=iou_th)
        matrix_after = cm.matrix.copy()
        batch_mistakes = matrix_after - matrix_before
        batch_mistakes[0,0] = 0
        batch_mistakes[1,1] = 0
        if vis_error_th is not None and batch_mistakes.sum() > vis_error_th:
            visualize(img_list[i], preds, refs, f'{prefix}.{i}.png')
    return cm


def calculate_prediction_similarity(true_results: list[torch.Tensor], stage_results_dict: dict[str, list[torch.Tensor]], iou_th: float, vis_error_th: int=None, img_list: list[np.ndarray]=None, results_dir: str=None) -> np.ndarray:
    similarity_data = np.empty((len(stage_results_dict), len(SIMILARITY_METRICS) + 1))
    for i, stage in enumerate(stage_results_dict):
        stage_results = stage_results_dict[stage]
        confusion_matrix = evaluate(stage_results, true_results, iou_th, vis_error_th, img_list, f'{results_dir}/mistake.{stage}').matrix
        matches = confusion_matrix[0,0] + confusion_matrix[1,1]
        mismatches = confusion_matrix[1,0] + confusion_matrix[0,1]
        missed = confusion_matrix[2].sum()
        novel = confusion_matrix[:,2].sum()
        accuracy = np.round((matches / confusion_matrix.sum())*100, 2)
        similarity_data[i] = (0, matches, mismatches, missed, novel, accuracy)
    return similarity_data


def plot_results(similarity_data, conversion_stages, results_dir):
    for i in range(1, similarity_data.shape[1]-1):
        bottom = similarity_data[:,:i].sum(axis=1)
        plt.bar(range(similarity_data.shape[0]), similarity_data[:,i], bottom=bottom, width=0.8, label=SIMILARITY_METRICS[i-1])
    xtick_labels = [f'{stage}\n{similarity_data[i,-1]}%' for i, stage in enumerate(conversion_stages)]
    plt.xticks(range(len(xtick_labels)), labels=xtick_labels)
    plt.xlabel('conversion stage')
    plt.ylabel('predictions w.r.t. PyTorch model')
    plt.legend(title='predictions')
    plt.tight_layout()
    plt.savefig(f'{results_dir}/degradation.png')