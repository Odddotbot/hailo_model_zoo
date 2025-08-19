'''Performance comparison of the model at different stages in the conversion process.'''

from evaluation.hailo_performance import *
from pathlib import Path
import matplotlib.pyplot as plt


preds_dir = Path('conversion_stages')
results_dir = Path('conversion_stages')
conversion_stages = {
    'pt': 'PyTorch',
    'onnx': 'ONNX',
    'unoptimized_har': 'unoptimized',
    'fp_optimized_har': 'FP optimized',
    'quantized_har': 'quantized',
    'comp0_val': 'compiled (.hef)',
}


annotation =  get_annotations('/media/oddbot/USBVANLUUK/val/images/annotations.npy') 
image_paths = get_image_paths('/media/oddbot/USBVANLUUK/val/images')

results_df = pd.DataFrame(columns=['Optimization level', 'Conversion stage', 'Accuracy', 'Precision', 'Recall'])
for opt_level in range(1,4):
    for stage in conversion_stages:
        preds_filepath = preds_dir / f'opt{opt_level}.{stage}.pk'
        if stage in ['pt', 'onnx']:
            preds = get_ultra_preds(preds_filepath, resize=True)
            results = evaluate(preds, annotation, image_paths)
        else:
            preds = get_hailo_preds(preds_filepath)
            results = evaluate(preds, annotation, image_paths, 0)
        results = calculate_metrics(results)[['Accuracy', 'Precision', 'Recall']]
        results[['Optimization level', 'Conversion stage']] = [opt_level, stage]
        results_df = pd.concat((results_df, results), ignore_index=True)

fig, axs = plt.subplots(3,1, figsize=(9,6))
for opt_level in range(1,4):
    results_for_lvl = results_df[(results_df['Optimization level'] == opt_level)]
    for i, metric in enumerate(['Accuracy', 'Precision', 'Recall']):
        axs[i].grid(axis='y')
        axs[i].plot(range(len(results_for_lvl)), results_for_lvl[metric], label=f'lvl={opt_level}')
        axs[i].set_ylabel(metric)
        axs[i].set_xticks([])
        axs[i].set_ylim(0.8,1)
        axs[i].legend(title='Optimization', loc='lower left')
axs[2].set_xticks(range(len(results_for_lvl)), [conversion_stages[stage] for stage in results_for_lvl['Conversion stage']])
plt.tight_layout()
plt.savefig(preds_dir / 'performance.png')