'''Inference of the different models (run on GPU)'''

import cv2
from hailo_sdk_client import ClientRunner, InferenceContext
import numpy as np
import os
from pathlib import Path
import pickle
import ultralytics


# VARIABLES
val_dir = Path('/datasets/hailo_calibration/val/images')
model_dir = Path('/experiments/ca_2025_06_26/train/ca_20250626_all_XL/weights')
model_name = 'opt1' # 'opt2', 'opt3
results_dir = Path('results')
model_conf = 0.4
model_iou = 0.5


# CONSTANTS
pt_filepath = model_dir / 'best.pt'
onnx_filepath = model_dir / 'best.onnx'
imgs_list = [cv2.resize(cv2.imread(val_dir / img_path), (448, 448)) for img_path in sorted(os.listdir(val_dir)) if img_path.endswith('jpg')]
imgs_array = np.stack(imgs_list)[:,:,:,[2,1,0]] # Switch BGR to RGB for Hailo
output_layers = ['/model.22/cv2.0/cv2.0.2/Conv', '/model.22/cv3.0/cv3.0.2/Conv', '/model.22/cv2.1/cv2.1.2/Conv', '/model.22/cv3.1/cv3.1.2/Conv', '/model.22/cv2.2/cv2.2.2/Conv', '/model.22/cv3.2/cv3.2.2/Conv']
layer_outputs = {}

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

def get_layer_output(name):
    def hook(model, input, output):
        output = output.detach().cpu().numpy()
        output = np.transpose(output, (0, 2, 3, 1))
        layer_outputs[name] = output
    return hook


# INFERENCE & SAVING
# Evaluate PyTorch
model = ultralytics.models.yolo.model.YOLO(pt_filepath).to('cpu')
# model.model.model[22].cv2[0][2].register_forward_hook(get_layer_output('/model.22/cv2.0/cv2.0.2/Conv'))
# model.model.model[22].cv3[0][2].register_forward_hook(get_layer_output('/model.22/cv3.0/cv3.0.2/Conv'))
# model.model.model[22].cv2[1][2].register_forward_hook(get_layer_output('/model.22/cv2.1/cv2.1.2/Conv'))
# model.model.model[22].cv3[1][2].register_forward_hook(get_layer_output('/model.22/cv3.1/cv3.1.2/Conv'))
# model.model.model[22].cv2[2][2].register_forward_hook(get_layer_output('/model.22/cv2.2/cv2.2.2/Conv'))
# model.model.model[22].cv3[2][2].register_forward_hook(get_layer_output('/model.22/cv3.2/cv3.2.2/Conv'))
results = [result.boxes.data for result in model.predict(imgs_list, conf=model_conf, iou=model_iou)]
pickle_and_save_results(f'{model_name}.pt.pk', results)

# Evaluate ONNX
model = ultralytics.models.yolo.model.YOLO(onnx_filepath, task='detect')
results = [model.predict(img, conf=model_conf, iou=model_iou)[0].boxes.data for img in imgs_list]
pickle_and_save_results(f'{model_name}.onnx.pk', results)

# Evaluate parsed HAR
# runner = ClientRunner(har=f'{model_dir.__str__()}/opt100_basic_parsing.har')
# with runner.infer_context(InferenceContext.SDK_NATIVE) as ctx:
#     raw_results = runner.infer(ctx, imgs_array)
# for hailo_result, output_layer in zip(raw_results, output_layers):
#     ultra_result = layer_outputs[output_layer]
#     print(ultra_result.mean(), ultra_result.std())
#     print(hailo_result.mean(), hailo_result.std())
#     print(f"MAE({output_layer}):", np.abs(hailo_result - ultra_result).mean())
# NOTE: this still gives differences, maybe because hailo returns the result BEFORE activation?

# Evaluate Un-optimized HAR
runner = ClientRunner(har=f'{model_dir.__str__()}/nothing.har')
with runner.infer_context(InferenceContext.SDK_FP_OPTIMIZED) as ctx:
    raw_results = runner.infer(ctx, imgs_array)
results = process_raw_hailo_results(raw_results)
pickle_and_save_results(f'{model_name}.unoptimized_har.pk', results)

# Evaluate FP-optimized HAR
runner = ClientRunner(har=f'{model_dir.__str__()}/{model_name}.har')
with runner.infer_context(InferenceContext.SDK_FP_OPTIMIZED) as ctx:
    raw_results = runner.infer(ctx, imgs_array)
results = process_raw_hailo_results(raw_results)
pickle_and_save_results(f'{model_name}.fp_optimized_har.pk', results)

# Evaluate Quantized HAR
runner = ClientRunner(har=f'{model_dir.__str__()}/{model_name}.har')
with runner.infer_context(InferenceContext.SDK_QUANTIZED) as ctx:
    raw_results = runner.infer(ctx, imgs_array)
results = process_raw_hailo_results(raw_results)
pickle_and_save_results(f'{model_name}.quantized_har.pk', results)

# Evaluate Compiled HAR on Raspberry Pi