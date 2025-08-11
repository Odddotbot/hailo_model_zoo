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
imgs_list = [cv2.imread(val_dir / img_path) for img_path in sorted(os.listdir(val_dir)) if img_path.endswith('jpg')][:10]
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
# model = ultralytics.models.yolo.model.YOLO(pt_filepath).to('cpu')
# results = [result.boxes.data for result in model.predict(imgs_list)]
# pickle_and_save_results('predictions.pt.pk', results)

# Evaluate ONNX
# model = ultralytics.models.yolo.model.YOLO(onnx_filepath)
# results = [result.boxes.data for result in model.predict(imgs_list)]
# pickle_and_save_results('predictions.onnx.pk', results)

# Evaluate Un-optimized HAR
runner = ClientRunner(har=f'{model_dir.__str__()}/nothing.har')
with runner.infer_context(InferenceContext.SDK_FP_OPTIMIZED) as ctx:
    raw_results = runner.infer(ctx, imgs_array)
results = process_raw_hailo_results(raw_results)
pickle_and_save_results('predictions.unoptimized_har.pk', results)

# TODO: Evaluate FP-optimized HAR
runner = ClientRunner(har=model_dir / '')

# TODO: Evaluate Quantized HAR


# TODO: Evaluate Compiled HAR


exit()

results_dir = '/experiments/ca_2025_06_26/train/ca_20250626_all_XL/weights'
pt_filename = 'best.pt'
model_name = 'opt1'
calib_dataset = np.ones((20, 448, 448, 3)) # TODO
end_node_names = ['/model.22/cv2.0/cv2.0.2/Conv', '/model.22/cv3.0/cv3.0.2/Conv', '/model.22/cv2.1/cv2.1.2/Conv', '/model.22/cv3.1/cv3.1.2/Conv', '/model.22/cv2.2/cv2.2.2/Conv', '/model.22/cv3.2/cv3.2.2/Conv']


# Seem not to be able to specify end nodes...

# TODO: I do seem to be able to infer from the .onnx model tho!

# # But then we must find out how to use Hailo's post-processing stuff...


# model = ultralytics.models.yolo.model.YOLO(f'{results_dir}/{pt_filename}') # Should specify iou and conf here!
# for layer in model.model:
#     print(layer)
# # result = model.predict(calib_dataset)
# # result = model.predict(calib_dataset)
# exit()

# The output here should be EXACTLY the same as the rest of the model. 
# Kinda hard to quantify tho.
runner = ClientRunner(har=f'{results_dir}/{model_name}_fp_optimized.har')
with runner.infer_context(InferenceContext.SDK_QUANTIZED) as ctx:
    output = runner.infer(ctx, calib_dataset)

print([o.shape for o in output])

# TODO: Evaluate parsed .onnx file (<model_name>_parsed.har)
# TODO: Evaluate equalized & pruned .har file (<model_name>_fp_optimized.har)
# TODO: Evaluate quantized file .har file (<model_name>.hef)

# TODO::::::::::::::::::::::::::::
# Read all the processes according to the logs
# Note that some calibration stuff is also specified WITHIN the alls file


# Can we also run a compiled model?


# The models that I will have
# - .pt                                                       = always the same
# - .onnx                                                     = always the same
# - -100 .har (no optimization) (= SDK_NATIVE? = Onnx?)       = always the same
# - fp_optimzied .har (= SDK_FP_OPTIMIZED?)                   (for opt 1-4)
# - quantized .har (= SDK_QUANTIZED?)                         (for opt 1-4)
# - compiled .har (= SDK_HAILO_HW = .hef?)                    (for opt 1-4)

# What about the different levels of optimization?