import cv2
from hailo_sdk_client import ClientRunner, InferenceContext
import numpy as np
import os
from pathlib import Path
import pickle
import ultralytics


# VARIABLES
val_dir = Path('/datasets/hailo_calibration/val/images_448')
model_dir = Path('/experiments/ca_2025_06_26/train/ca_20250626_all_XL/weights')
model_name = 'opt3'
results_dir = Path('results')


# CONSTANTS
pt_filepath = model_dir / 'best.pt'
onnx_filepath = model_dir / 'best.onnx'
imgs_list = [cv2.imread(val_dir / img_path) for img_path in sorted(os.listdir(val_dir)) if img_path.endswith('jpg')]
imgs_array = np.stack(imgs_list)


# HELPER FUNCTIONS
def pickle_and_save_results(filename, object):
    with open(results_dir / filename, 'wb') as file:
        pickle.dump(object, file)

def process_raw_hailo_results(raw_results):
    results = []
    for result in raw_results:
        weed_result = result[0].transpose()
        crop_result = result[1].transpose()
        weed_result = weed_result[weed_result.sum(axis=1) != 0]
        crop_result = crop_result[crop_result.sum(axis=1) != 0]
        results.append([weed_result, crop_result])
    return results


# INFERENCE & SAVING
# Evaluate PyTorch
model = ultralytics.models.yolo.model.YOLO(pt_filepath).to('cpu')
results = [result.boxes.data for result in model.predict(imgs_list)]
pickle_and_save_results('predictions.pt.pk', results)

# Evaluate ONNX
model = ultralytics.models.yolo.model.YOLO(onnx_filepath, task='detect')
results = [model.predict(img)[0].boxes.data for img in imgs_list]
pickle_and_save_results('predictions.onnx.pk', results)

# Evaluate Un-optimized HAR
runner = ClientRunner(har=f'{model_dir.__str__()}/nothing.har')
with runner.infer_context(InferenceContext.SDK_FP_OPTIMIZED) as ctx:
    raw_results = runner.infer(ctx, imgs_array)
results = process_raw_hailo_results(raw_results)
pickle_and_save_results('predictions.unoptimized_har.pk', results)

# Evaluate FP-optimized HAR
runner = ClientRunner(har=f'{model_dir.__str__()}/{model_name}.har')
with runner.infer_context(InferenceContext.SDK_FP_OPTIMIZED) as ctx:
    raw_results = runner.infer(ctx, imgs_array)
results = process_raw_hailo_results(raw_results)
pickle_and_save_results('predictions.fp_optimized_har.pk', results)

# Evaluate Quantized HAR
runner = ClientRunner(har=f'{model_dir.__str__()}/{model_name}.har')
with runner.infer_context(InferenceContext.SDK_QUANTIZED) as ctx:
    raw_results = runner.infer(ctx, imgs_array)
results = process_raw_hailo_results(raw_results)
pickle_and_save_results('predictions.quantized_har.pk', results)

# TODO: Evaluate Compiled HAR on Raspberry Pi (or elsewhere I guess)