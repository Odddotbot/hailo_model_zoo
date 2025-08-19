'''Comparison of models that were generated using different hyperparmater settings of the model conversion algorithm.'''

from evaluation.hailo_performance import *

annotations = get_annotations('/media/oddbot/USBVANLUUK/val/images/annotations.npy')
ultra_preds = get_ultra_preds('ultralytics_output_val.pk')
image_paths = get_image_paths('/media/oddbot/USBVANLUUK/val/images')

cm_ultra_vs_annotations = evaluate(ultra_preds, annotations, image_paths)
save_results(cm_ultra_vs_annotations, 'ultra-vs-annotations')

hailo_models = [
    'opt100_comp0',
    'opt0_comp0',
    'opt1_comp0',
    'opt2_comp0',
    'opt3_comp0',
    'opt4_comp0_nano',
    'opt3_comp0_nano',
    'opt3_comp0_carrots_256',
    'opt3_comp0_onions_256',
]

for hailo_model in hailo_models:
    hailo_preds = get_hailo_preds(f'/media/oddbot/USBVANLUUK/verified_models/{hailo_model}_val.pk')
    cm_vs_ultralytics = evaluate(hailo_preds, ultra_preds, image_paths)
    save_results(cm_vs_ultralytics, f'{hailo_model}-vs-ultralytics')
    cm_vs_annotations = evaluate(hailo_preds, annotations, image_paths)
    save_results(cm_vs_annotations, f'{hailo_model}-vs-annotations')
