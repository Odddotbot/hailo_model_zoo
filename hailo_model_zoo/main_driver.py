from pathlib import Path

# FOR validation stuff
import os
import yaml
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import ultralytics
from ultralytics.utils.metrics import ConfusionMatrix
import torch

try:
    from hailo_platform import HEF, PcieDevice

    HEF_EXISTS = True
except ModuleNotFoundError:
    HEF_EXISTS = False

from hailo_sdk_client import ClientRunner, InferenceContext
from hailo_sdk_client.exposed_definitions import States
from hailo_sdk_client.tools.profiler.react_report_generator import ReactReportGenerator
from hailo_sdk_common.logger.logger import DeprecationVersion
from hailo_sdk_common.targets.inference_targets import SdkFPOptimized, SdkPartialNumeric

from hailo_model_zoo.core.main_utils import (
    compile_model,
    get_hef_path,
    get_integrated_postprocessing,
    get_network_info,
    infer_model_tf1,
    infer_model_tf2,
    is_network_performance,
    optimize_full_precision_model,
    optimize_model,
    parse_model,
    prepare_calibration_data,
    resolve_alls_path,
)
from hailo_model_zoo.utils.hw_utils import DEVICE_NAMES, DEVICES, INFERENCE_TARGETS, TARGETS
from hailo_model_zoo.utils.logger import get_logger
import ultralytics # For YOLO .pt -> .onnx conversion


# TODO put this at a better place
CLASSES = ['weed', 'crop']
SIMILARITY_METRICS = ['matched', 'mismatched', 'missed', 'added', 'accuracy']


def _ensure_performance(model_name, model_script, performance, hw_arch, logger):
    if not performance and is_network_performance(model_name, hw_arch):
        # Check whether the model has a performance
        logger.info(f"Running {model_name} with default model script.\n\
                       To obtain maximum performance use --performance:\n\
                       hailomz <command> {model_name} --performance")
    if performance and model_script:
        if model_script.parent.name == "base":
            logger.info(f"Using base alls script found in {model_script} because there is no performance alls")
        elif model_script.parent.name == "generic":
            logger.info(f"Using generic alls script found in {model_script} because there is no specific hardware alls")
        elif model_script.parent.name == "performance" and is_network_performance(model_name, hw_arch):
            logger.info(f"Using performance alls script found in {model_script}")
        else:
            logger.info(f"Using alls script from {model_script}")


def _extract_model_script_path(networks_alls_script, model_script_path, hw_arch, performance):
    return (
        Path(model_script_path)
        if model_script_path
        else resolve_alls_path(networks_alls_script, hw_arch=hw_arch, performance=performance)
    )


def _ensure_compiled(runner, logger, args, network_info):
    if runner.state == States.COMPILED_MODEL or runner.hef:
        return
    logger.info("Compiling the model (without inference) ...")
    compile_model(
        runner,
        network_info,
        args.results_dir,
        allocator_script_filename=args.model_script_path,
        performance=args.performance,
    )


def _ensure_optimized(runner, logger, args, network_info):
    _ensure_parsed(runner, logger, network_info, args)

    integrated_postprocessing = get_integrated_postprocessing(network_info)
    if integrated_postprocessing and integrated_postprocessing.enabled and args.model_script_path is not None:
        raise ValueError(
            f"Network {network_info.network.network_name} joins several networks together\n"
            "and cannot get a user model script"
        )

    if runner.state != States.HAILO_MODEL:
        return
    model_script = _extract_model_script_path(
        network_info.paths.alls_script, args.model_script_path, args.hw_arch, args.performance
    )

    # Set optimization level and comrpression level as specified in arguments
    if args.optimization_level is not None or args.compression_level is not None:
        with open(model_script, 'r') as file:
            contents = file.readlines()
        contents = [line for line in contents if not line.startswith('model_optimization_flavor')]
        line_to_add = f"model_optimization_flavor(optimization_level={args.optimization_level}, compression_level={args.compression_level})\n"
        contents.append(line_to_add)
        model_script = model_script.with_stem(model_script.stem + '_tmp')
        with open(model_script , 'w') as file:
            file.writelines(contents)

    _ensure_performance(network_info.network.network_name, model_script, args.performance, args.hw_arch, logger)
    calib_feed_callback = prepare_calibration_data(
        runner, network_info, args.calib_path, logger, args.input_conversion, args.resize
    )
    model_name = args.output_name if args.output_name else network_info.network.network_name
    optimize_model(
        model_name,
        runner,
        calib_feed_callback,
        logger,
        network_info,
        args.results_dir,
        model_script,
        args.resize,
        args.input_conversion,
        args.classes,
        args.imgsize,
        args.nms_scores_th,
        args.nms_iou_th
    )


def _ensure_parsed(runner, logger, network_info, args):
    if runner.state != States.UNINITIALIZED:
        return

    # Use .pt file and convert to .onnx if provided in arguments
    if args.pt_filepath: 
        model = ultralytics.models.yolo.model.YOLO(args.pt_filepath)
        model.export(format="onnx", imgsz=args.imgsize, opset=11)
        ckpt_path = args.pt_filepath.replace(".pt", ".onnx")
    else:
        ckpt_path = args.ckpt_path

    model_name = args.output_name if args.output_name else network_info.network.network_name
    parse_model(runner, network_info, model_name, ckpt_path=ckpt_path, results_dir=args.results_dir, logger=logger)


def configure_hef_tf1(hef_path, target):
    hef = HEF(hef_path)
    network_groups = target.configure(hef)
    return network_groups


def configure_hef_tf2(runner, hef_path):
    if hef_path:
        runner.hef = hef_path
    return


def _ensure_runnable_state_tf1(args, logger, network_info, runner, target):
    _ensure_parsed(runner, logger, network_info, args)
    if isinstance(target, SdkFPOptimized) or (isinstance(target, PcieDevice) and args.hef_path is not None):
        if runner.state == States.HAILO_MODEL:
            calib_feed_callback = prepare_calibration_data(
                runner, network_info, args.calib_path, logger, args.input_conversion, args.resize
            )
            integrated_postprocessing = get_integrated_postprocessing(network_info)
            if integrated_postprocessing and integrated_postprocessing.enabled:
                runner.optimize_full_precision(calib_data=calib_feed_callback)
                return configure_hef_tf1(args.hef_path, target) if args.hef_path else None
            # We intentionally use base model script and assume its modifications
            # compatible to the performance model script
            model_script = _extract_model_script_path(
                network_info.paths.alls_script, args.model_script_path, args.hw_arch, performance=False
            )

            optimize_full_precision_model(
                runner, calib_feed_callback, logger, model_script, args.resize, args.input_conversion, args.classes
            )

        return configure_hef_tf1(args.hef_path, target) if args.hef_path else None

    if args.hef_path:
        return configure_hef_tf1(args.hef_path, target)

    _ensure_optimized(runner, logger, args, network_info)

    if isinstance(target, SdkPartialNumeric):
        return

    assert isinstance(target, PcieDevice)
    _ensure_compiled(runner, logger, args, network_info)
    return None


def _ensure_runnable_state_tf2(args, logger, network_info, runner, target):
    _ensure_parsed(runner, logger, network_info, args)
    if target == InferenceContext.SDK_FP_OPTIMIZED or (
        target == InferenceContext.SDK_HAILO_HW and args.hef_path is not None
    ):
        if runner.state != States.HAILO_MODEL:
            configure_hef_tf2(runner, args.hef_path)
            return

        # We intentionally use base model script and assume its modifications
        # compatible to the performance model script
        model_script = _extract_model_script_path(
            network_info.paths.alls_script, args.model_script_path, args.hw_arch, False
        )
        calib_feed_callback = prepare_calibration_data(
            runner, network_info, args.calib_path, logger, args.input_conversion, args.resize
        )
        optimize_full_precision_model(
            runner, calib_feed_callback, logger, model_script, args.resize, args.input_conversion, args.classes
        )
        configure_hef_tf2(runner, args.hef_path)

    else:
        configure_hef_tf2(runner, args.hef_path)

        _ensure_optimized(runner, logger, args, network_info)

        if target != InferenceContext.SDK_QUANTIZED:
            _ensure_compiled(runner, logger, args, network_info)

    return


def parse(args):
    logger = get_logger()
    nodes = [args.start_node_names, args.end_node_names]
    network_info = get_network_info(args.model_name, yaml_path=args.yaml_path, nodes=nodes)
    model_name = network_info.network.network_name
    logger.info(f"Start run for network {model_name} ...")

    logger.info("Initializing the runner...")
    runner = ClientRunner(hw_arch=args.hw_arch)
    parse_model(runner, network_info, ckpt_path=args.ckpt_path, results_dir=args.results_dir, logger=logger)


def optimize(args):
    logger = get_logger()
    nodes = [args.start_node_names, args.end_node_names]
    network_info = get_network_info(args.model_name, yaml_path=args.yaml_path, nodes=nodes)
    model_name = network_info.network.network_name
    logger.info(f"Start run for network {model_name} ...")

    if args.calib_path is None and network_info.quantization.calib_set is None:
        raise ValueError("Cannot run optimization without dataset. use --calib-path to provide external dataset.")

    logger.info(f"Initializing the {args.hw_arch} runner...")
    runner = ClientRunner(hw_arch=args.hw_arch, har=args.har_path)
    _ensure_parsed(runner, logger, network_info, args)

    model_script = _extract_model_script_path(
        network_info.paths.alls_script, args.model_script_path, args.hw_arch, args.performance
    )
    _ensure_performance(model_name, model_script, args.performance, args.hw_arch, logger)
    calib_feed_callback = prepare_calibration_data(
        runner, network_info, args.calib_path, logger, args.input_conversion, args.resize
    )
    optimize_model(
        runner,
        calib_feed_callback,
        logger,
        network_info,
        args.results_dir,
        model_script,
        args.resize,
        args.input_conversion,
        args.classes,
        args.imgsize,
        args.nms_scores_th,
        args.nms_iou_th
    )


def compile(args):
    logger = get_logger()
    nodes = [args.start_node_names, args.end_node_names]
    network_info = get_network_info(args.model_name, yaml_path=args.yaml_path, nodes=nodes, imgsize=args.imgsize)
    model_name = network_info.network.network_name
    logger.info(f"Start run for network {model_name} ...")

    logger.info(f"Initializing the {args.hw_arch} runner...")
    runner = ClientRunner(hw_arch=args.hw_arch, har=args.har_path)

    _ensure_optimized(runner, logger, args, network_info)

    model_script = _extract_model_script_path(
        network_info.paths.alls_script, args.model_script_path, args.hw_arch, args.performance
    )

    # Set optimization level and comrpression level as specified in arguments
    if args.optimization_level is not None or args.compression_level is not None:
        with open(model_script, 'r') as file:
            contents = file.readlines()
        contents = [line for line in contents if not line.startswith('model_optimization_flavor')]
        line_to_add = f"model_optimization_flavor(optimization_level={args.optimization_level}, compression_level={args.compression_level})\n"
        contents.append(line_to_add)
        model_script = model_script.with_stem(model_script.stem + '_tmp')
        with open(model_script , 'w') as file:
            file.writelines(contents)

    _ensure_performance(model_name, model_script, args.performance, args.hw_arch, logger)
    compile_model(runner, network_info, args.results_dir, model_script, performance=args.performance, output_name=args.output_name)

    model_name = args.output_name if args.output_name else network_info.network.network_name
    logger.info(f"HEF file written to {get_hef_path(args.results_dir, model_name)}")


def profile(args):
    logger = get_logger()
    logger.deprecation_warning(
        (
            "'profile' command is deprecated and will be removed in future release."
            " Please use 'hailo profiler' tool instead."
        ),
        DeprecationVersion.FUTURE,
    )
    nodes = [args.start_node_names, args.end_node_names]
    network_info = get_network_info(args.model_name, yaml_path=args.yaml_path, nodes=nodes)
    model_name = network_info.network.network_name
    logger.info(f"Start run for network {model_name} ...")

    logger.info(f"Initializing the {args.hw_arch} runner...")
    runner = ClientRunner(hw_arch=args.hw_arch, har=args.har_path)

    _ensure_parsed(runner, logger, network_info, args)

    if args.hef_path and runner.state == States.HAILO_MODEL:
        model_script = _extract_model_script_path(
            network_info.paths.alls_script, args.model_script_path, args.hw_arch, args.performance
        )
        _ensure_performance(model_name, model_script, args.performance, args.hw_arch, logger)
        calib_feed_callback = prepare_calibration_data(
            runner, network_info, args.calib_path, logger, args.input_conversion, args.resize
        )
        optimize_full_precision_model(
            runner, calib_feed_callback, logger, model_script, args.resize, args.input_conversion, args.classes
        )

    export = runner.profile(should_use_logical_layers=True, hef_filename=args.hef_path)
    outpath = args.results_dir / f"{model_name}.html"
    report_generator = ReactReportGenerator(export, outpath)
    csv_data = report_generator.create_report(should_open_web_browser=False)
    logger.info(f"Profiler report generated in {outpath}")

    return export["stats"], csv_data, export["latency_data"]


def evaluate(args):
    logger = get_logger()

    if args.target == "hardware" and not HEF_EXISTS:
        raise ModuleNotFoundError(
            f"HailoRT is not available, in case you want to run on {args.target} you should install HailoRT first"
        )

    if (args.hw_arch == ["hailo15h", "hailo15m"] and args.target == "hardware") and not args.use_service:
        raise ValueError("Evaluation of hw_arch hailo15h is currently not supported in the Hailo Model Zoo")

    if args.hef_path and not HEF_EXISTS:
        raise ModuleNotFoundError(
            "HailoRT is not available, in case you want to evaluate with hef you should install HailoRT first"
        )

    hardware_targets = set(DEVICE_NAMES)
    hardware_targets.update(["hardware"])
    if args.hef_path and args.target not in hardware_targets:
        raise ValueError(
            f"hef is not used when evaluating with {args.target}. use --target hardware for evaluating with a hef."
        )

    if args.video_outpath and not args.visualize_results:
        raise ValueError("The --video-output argument requires --visualize argument")

    nodes = [args.start_node_names, args.end_node_names]
    network_info = get_network_info(args.model_name, yaml_path=args.yaml_path, nodes=nodes)

    if args.data_path is None and network_info.evaluation.data_set is None:
        raise ValueError("Cannot run evaluation without dataset. use --data-path to provide external dataset.")

    if path := args.custom_infer_config:
        custom_infer_config = Path(path)
        if not custom_infer_config.is_file:
            raise ValueError(
                "The given path for '--custom-infer-file' is not a file, please provide a valid file for the argument"
            )
        if not args.target == "emulator":
            raise ValueError("custom_infer_config only works on target: emulator")

    model_name = network_info.network.network_name
    logger.info(f"Start run for network {model_name} ...")

    logger.info("Initializing the runner...")
    runner = ClientRunner(hw_arch=args.hw_arch, har=args.har_path)
    network_groups = None

    #  Enabling service for hailo15h
    if args.use_service:
        # This property will print a warning when set.
        runner.use_service = args.use_service

    logger.info(f"Chosen target is {args.target}")
    batch_size = args.batch_size or __get_batch_size(network_info, args.target)
    infer_type = network_info.evaluation.infer_type
    # legacy tf1 inference flow
    if infer_type not in [
        "runner_infer",
        "model_infer",
        "np_infer",
        "facenet_infer",
        "np_infer_lite",
        "model_infer_lite",
        "sd2_unet_infer",
    ]:
        hailo_target = TARGETS[args.target]
        with hailo_target() as target:
            network_groups = _ensure_runnable_state_tf1(args, logger, network_info, runner, target)
            return infer_model_tf1(
                runner,
                network_info,
                target,
                logger,
                args.eval_num_examples,
                args.data_path,
                batch_size,
                args.print_num_examples,
                args.visualize_results,
                args.video_outpath,
                dump_results=False,
                network_groups=network_groups,
            )

    else:
        # new tf2 inference flow
        target = INFERENCE_TARGETS[args.target]
        _ensure_runnable_state_tf2(args, logger, network_info, runner, target)

        device_info = DEVICES.get(args.target)
        # overrides nms score threshold if postprocess on-host
        nms_score_threshold = (
            network_info["postprocessing"].get("score_threshold", None)
            if network_info["postprocessing"]["hpp"] and not network_info["postprocessing"]["bbox_decoding_only"]
            else None
        )
        context = runner.infer_context(
            target,
            device_ids=device_info,
            nms_score_threshold=nms_score_threshold,
            custom_infer_config=args.custom_infer_config,
        )
        return infer_model_tf2(
            runner,
            network_info,
            context,
            logger,
            args.eval_num_examples,
            args.data_path,
            batch_size,
            args.print_num_examples,
            args.visualize_results,
            args.video_outpath,
            args.use_lite_inference,
            dump_results=False,
            input_conversion_args=args.input_conversion,
            resize_args=args.resize,
            show_results_per_class=args.show_results_per_class,
        )
    

def load_validation_data(data_yaml: str, imgsize: int) -> np.ndarray:
    with open(data_yaml) as data_yaml_file:
        data_config = yaml.safe_load(data_yaml_file) # NOTE: this actually has the number/names/order of classes
    val_data_dirs = data_config['val']
    val_data_dirs = [f"/datasets/{dir.split('/datasets/')[1]}" for dir in val_data_dirs] # only /datasets is mounted in Docker
    img_paths = [f'{dir}/images/val/{image_path}' for dir in val_data_dirs for image_path in sorted(os.listdir(f'{dir}/images/val'))]
    img_list = [cv2.imread(img_path) for img_path in img_paths]
    img_list = [cv2.resize(img, imgsize) for img in img_list]
    return img_list


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


def infer_hailo(har_filepath: str, img_list: list[np.ndarray], conf: float, iou: float, inference_context: InferenceContext) -> list[torch.Tensor]:
    
    imgs_array = np.stack(img_list)[:,:,:,[2,1,0]] # Switch BGR to RGB for Hailo
    img_width, img_height = imgs_array.shape[2], imgs_array.shape[1]
    runner = ClientRunner(har=har_filepath, hw_arch='hailo8')
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
        added = confusion_matrix[:,2].sum()
        accuracy = np.round((matches / confusion_matrix.sum())*100, 2)
        similarity_data[i] = (0, matches, mismatches, missed, added, accuracy)
    return similarity_data


def plot_results(similarity_data, conversion_stages, results_dir):
    for i in range(1, similarity_data.shape[1]-1):
        bottom = bottom=similarity_data[:,:i-1].sum(axis=1)
        plt.bar(range(similarity_data.shape[0]), similarity_data[:,i], bottom=bottom, width=0.8, label=SIMILARITY_METRICS[i-1])
    xtick_labels = [f'{stage}\n{similarity_data[i,-1]}%' for i, stage in enumerate(conversion_stages)]
    plt.xticks(range(len(xtick_labels)), labels=xtick_labels)
    plt.xlabel('conversion stage')
    plt.ylabel('predictions w.r.t. PyTorch model')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{results_dir}/degradation.png')


def validate(args):

    # TODO: Not sure if all the variable names are like super duper interintuitive

    # Loading validation data
    img_list = load_validation_data(args.data_yaml, args.imgsize)
    
    results_dir = "/".join(args.pt_filepath.split("/weights")[:-1]) + '/conversion'
    results_dir = 'conversion' # TODO remove this line
    Path(results_dir).mkdir(exist_ok=True)

    pt_results = infer_ultralytics(args.pt_filepath, img_list, args.nms_scores_th, args.nms_iou_th)
    onnx_filepath = args.pt_filepath.replace('.pt', '.onnx')
    conversion_results = {
        'ONNX': infer_ultralytics(onnx_filepath, img_list, args.nms_scores_th, args.nms_iou_th),
        'FP optimized': infer_hailo(args.har_filepath, img_list, args.nms_scores_th, args.nms_iou_th, InferenceContext.SDK_FP_OPTIMIZED),
        'Quantized': infer_hailo(args.har_filepath, img_list, args.nms_scores_th, args.nms_iou_th, InferenceContext.SDK_QUANTIZED),
        'Hardware (emulated)': infer_hailo(args.har_filepath, img_list, args.nms_scores_th, args.nms_iou_th, InferenceContext.SDK_BIT_EXACT)
    }

    similarity_data = calculate_prediction_similarity(pt_results, conversion_results, args.val_iou_th, args.vis_error_th, img_list, results_dir)
    plot_results(similarity_data, conversion_results.keys(), results_dir)
    print(similarity_data)
    for i, stage in enumerate(conversion_results):
        similarity = similarity_data[i,-1] / 100
        if similarity <= args.similarity_th:
            print(f"FAILED conversion validation at '{stage}' stage, similarity to .pt model ({np.round(similarity, 4)}) below threshold ({args.similarity_th}).")
            return
        
    print(f"SUCCESS in conversion validation.")

    # TODO: deleting the .har and .onnx files if conversion was succesfull?


def __get_batch_size(network_info, target):
    if target == "full_precision":
        return network_info.inference.full_precision_batch_size
    return network_info.inference.emulator_batch_size
