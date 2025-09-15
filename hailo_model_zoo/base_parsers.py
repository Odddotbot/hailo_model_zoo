import argparse
from pathlib import Path

from hailo_model_zoo.utils.cli_utils import OneResizeValueAction, add_model_name_arg
from hailo_model_zoo.utils.completions import (
    ALLS_COMPLETE,
    CKPT_COMPLETE,
    FILE_COMPLETE,
    HAR_COMPLETE,
    HEF_COMPLETE,
    TFRECORD_COMPLETE,
    YAML_COMPLETE,
)
from hailo_model_zoo.utils.constants import DEVICE_NAMES, TARGETS


def make_parsing_base():
    parsing_base_parser = argparse.ArgumentParser(add_help=False)
    config_group = parsing_base_parser.add_mutually_exclusive_group()
    add_model_name_arg(config_group, optional=True)
    parsing_base_parser.add_argument(
        "--hw-arch",
        type=str,
        metavar="",
        choices=["hailo8", "hailo8l", "hailo15h", "hailo15m", "hailo15l", "hailo10h", "hailo10h2", "mars"],
        help="Which hw arch to run: hailo8 / hailo8l/ hailo15h/ hailo15m / hailo10h. By default using hailo8.",
    )
    parsing_base_parser.add_argument(
        "--classes", type=int, metavar="", help="Number of classes for NMS configuration"
    )
    config_group.add_argument(
        "--yaml",
        type=str,
        default=None,
        dest="yaml_path",
        help=("Path to YAML for network configuration." "By default using the default configuration"),
    ).complete = YAML_COMPLETE
    parsing_base_parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        dest="ckpt_path",
        help=("Path to onnx or ckpt to use for parsing." " By default using the model cache location"),
    ).complete = CKPT_COMPLETE
    parsing_base_parser.add_argument(
        "--start-node-names",
        type=str,
        default="",
        nargs="+",
        help="List of names of the first nodes to parse.\nExample: --start-node-names <start_name1> <start_name2> ...",
    )
    parsing_base_parser.add_argument(
        "--end-node-names",
        type=str,
        default="",
        nargs="+",
        help="List of nodes that indicate the parsing end. The order determines the order of the outputs."
        "\nExample: --end-node-names <end_name1> <end_name2> ...",
    )
    parsing_base_parser.set_defaults(results_dir=Path("./"))
    return parsing_base_parser


def make_optimization_base():
    optimization_base_parser = argparse.ArgumentParser(add_help=False)
    mutually_exclusive_group = optimization_base_parser.add_mutually_exclusive_group()
    mutually_exclusive_group.add_argument(
        "--model-script",
        type=str,
        default=None,
        dest="model_script_path",
        help="Path to model script to use. By default using the model script specified"
        " in the network YAML configuration",
    ).complete = ALLS_COMPLETE
    mutually_exclusive_group.add_argument(
        "--performance",
        action="store_true",
        help="Enable flag for benchmark performance",
    )

    optimization_base_parser.add_argument(
        "--har", type=str, default=None, help="Use external har file", dest="har_path"
    ).complete = HAR_COMPLETE
    optimization_base_parser.add_argument(
        "--calib-path",
        type=Path,
        help="Path to external tfrecord for calibration or a directory containing \
            images in jpg or png format",
    ).complete = TFRECORD_COMPLETE
    optimization_base_parser.add_argument(
        "--resize",
        type=int,
        nargs="+",
        action=OneResizeValueAction,
        help="Add input resize from given [h,w]",
    )
    optimization_base_parser.add_argument(
        "--input-conversion",
        type=str,
        choices=["nv12_to_rgb", "yuy2_to_rgb", "rgbx_to_rgb"],
        help="Add input conversion from given type",
    )
    optimization_base_parser.add_argument(
        "--optimization_level",
        type=int,
        help="(Odd.Bot) Set optimization level of Hailo model compiler. Higher levels cause longer runtimes but less degradation."
    )
    optimization_base_parser.add_argument(
        "--compression_level",
        type=int,
        help="(Odd.Bot) Set compression level of Hailo model compiler. Higher levels cause faster inference but more degradation."
    )
    optimization_base_parser.add_argument(
        "--output_name",
        type=str,
        help="(Odd.Bot) Set output name of the model."
    )
    return optimization_base_parser


def make_hef_base():
    hef_base_parser = argparse.ArgumentParser(add_help=False)
    hef_base_parser.add_argument(
        "--hef", type=str, default=None, help="Use external HEF files", dest="hef_path"
    ).complete = HEF_COMPLETE
    return hef_base_parser


def make_profiling_base():
    profile_base_parser = argparse.ArgumentParser(add_help=False)
    return profile_base_parser


def make_evaluation_base():
    evaluation_base_parser = argparse.ArgumentParser(add_help=False)
    targets = TARGETS
    devices = ", ".join(DEVICE_NAMES)
    evaluation_base_parser.add_argument(
        "--target",
        type=str,
        choices=targets,
        metavar="",
        default="full_precision",
        help="Which target to run: full_precision (GPU) / emulator (GPU) / hardware (PCIe).\n"
        f"A specific device may be specified. Available devices: {devices}",
    )

    evaluation_base_parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size for INFERENCE (evaluation and pre-quant stats collection) only "
        "(feel free to increase to whatever your GPU can handle). "
        " the quant-aware optimizers s.a. QFT & IBC use the calibration batch size parameter from the ALLS",
    )

    evaluation_base_parser.add_argument(
        "--data-count",
        type=int,
        default=None,
        dest="eval_num_examples",
        help="Maximum number of images to use for evaluation",
    )

    evaluation_base_parser.add_argument(
        "--visualize",
        action="store_true",
        dest="visualize_results",
        help="Run visualization without evaluation. The default value is False",
    )
    evaluation_base_parser.add_argument(
        "--video-outpath",
        help="Make a video from the visualizations and save it to this path",
    ).complete = FILE_COMPLETE
    evaluation_base_parser.add_argument(
        "--data-path",
        type=Path,
        help="Path to external tfrecord for evaluation. In case you use --visualize \
            you can give a directory of images in jpg or png format",
    ).complete = TFRECORD_COMPLETE
    evaluation_base_parser.add_argument(
        "--ap-per-class",
        action="store_true",
        dest="show_results_per_class",
        help="Print AP results per class, relevant only for object detection and instance segmentation tasks",
    )

    evaluation_base_parser.add_argument(
        "--custom-infer-config",
        type=Path,
        dest="custom_infer_config",
        help="A file that indicates witch elements to set lossless or lossy",
    )
    evaluation_base_parser.set_defaults(
        print_num_examples=1e9,
        visualize_results=False,
        use_lite_inference=False,
        use_service=False,
    )
    return evaluation_base_parser


def make_oddbot_base():
    oddbot_base_parser = argparse.ArgumentParser(add_help=False)

    oddbot_base_parser.add_argument(
        "--config",
        type=str,
        help="(Odd.Bot) Path to .yaml file with settings to use for conversion/validation. Settings can be overridden by other arguments.",
    )
    oddbot_base_parser.add_argument(
        "--pt_filepath",
        type=str,
        help="(Odd.Bot) Path to PyTorch model."
    )
    oddbot_base_parser.add_argument(
        "--imgsize",
        type=int,
        nargs="+",
        action=OneResizeValueAction,
        help="(Odd.Bot) Set image size to given [h,w]",
    )
    oddbot_base_parser.add_argument(
        "--nms_scores_th",
        type=float,
        help="(Odd.Bot) Set output NMS confidence threshold to given value."
    )
    oddbot_base_parser.add_argument(
        "--nms_iou_th",
        type=float,
        help="(Odd.Bot) Set output NMS IOU threshold to given value."
    )
    oddbot_base_parser.add_argument(
        "--results_dir",
        default=None,
        type=lambda string: Path(string),
        help="(Odd.Bot) Results directory, where to save the output to."
    )

    return oddbot_base_parser


def make_validation_base():
    validation_base_parser = argparse.ArgumentParser(add_help=False)

    validation_base_parser.add_argument(
        "--har_filepath",
        type=str,
        help="(Odd.Bot) Path to Hailo archive (.har)."
    )

    validation_base_parser.add_argument(
        "--data_yaml",
        type=str,
        help="(Odd.Bot) Path to .yaml file describing data used for validation (ignoring training data)."
    )

    validation_base_parser.add_argument(
        "--ground_truth_src",
        choices=["pt", "sly"],
        help="(Odd.Bot) Whether to use the PyTorch predictions (pt) or Supervisely (sly) annotations as ground truth for comparing Hailo against."
    )

    validation_base_parser.add_argument(
        "--similarity_th",
        type=float,
        help="(Odd.Bot) Tolerated degradation w.r.t. the original PyTorch model."
    )

    validation_base_parser.add_argument(
        '--val_iou_th',
        type=float,
        help='(Odd.Bot) The IOU threshold used to determine whether a prediction at certain conversion stage is the same as the PyTorch one. Higher is stricter.'
    )

    validation_base_parser.add_argument(
        '--vis_error_th',
        type=float,
        help='(Odd.Bot) If specified, will save images/predictions with more than this number of mistakes.'
    )

    return validation_base_parser