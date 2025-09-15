#!/usr/bin/env python
import argparse
import importlib
from pathlib import Path
import yaml

# we try to minimize imports to make 'main.py --help' responsive. So we only import definitions.
import hailo_model_zoo.plugin
from hailo_model_zoo.base_parsers import (
    make_evaluation_base,
    make_hef_base,
    make_optimization_base,
    make_parsing_base,
    make_profiling_base,
    make_oddbot_base,
    make_validation_base,
    make_compile_and_validate_base,
)
from hailo_model_zoo.utils.cli_utils import HMZ_COMMANDS
from hailo_model_zoo.utils.plugin_utils import iter_namespace
from hailo_model_zoo.utils.version import get_version

import mlflow

discovered_plugins = {
    name: importlib.import_module(name) for finder, name, ispkg in iter_namespace(hailo_model_zoo.plugin)
}


def _create_args_parser():
    # --- create shared arguments parsers
    parsing_base_parser = make_parsing_base()
    optimization_base_parser = make_optimization_base()
    hef_base_parser = make_hef_base()
    profile_base_parser = make_profiling_base()
    evaluation_base_parser = make_evaluation_base()
    oddbot_base_parser = make_oddbot_base()
    validation_base_parser = make_validation_base()
    compile_and_validate_base = make_compile_and_validate_base()
    version = get_version("hailo_model_zoo")

    # --- create per action subparser
    parser = argparse.ArgumentParser(epilog="Example: hailomz parse resnet_v1_50")
    parser.add_argument("--version", action="version", version=f"Hailo Model Zoo v{version}")
    # can't set the entry point for each subparser as it forces us to add imports which slow down the startup time.
    # instead we'll check the 'command' argument after parsing
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser(
        "parse",
        parents=[parsing_base_parser],
        help="model translation of the input model into Hailo's internal representation.",
    )

    subparsers.add_parser(
        "optimize",
        parents=[parsing_base_parser, optimization_base_parser],
        help="run model optimization which includes numeric translation of \
                                the input model into a compressed integer representation.",
    )

    compile_help = (
        "run the Hailo compiler to generate the Hailo Executable Format file (HEF)"
        " which can be executed on the Hailo hardware."
    )
    subparsers.add_parser(
        "compile",
        parents=[
            oddbot_base_parser,
            parsing_base_parser, 
            optimization_base_parser,
        ],
        help=compile_help,
    )

    profile_help = (
        "generate profiler report of the model."
        " The report contains information about your model and expected performance on the Hailo hardware."
    )
    subparsers.add_parser(
        "profile",
        parents=[
            parsing_base_parser,
            optimization_base_parser,
            hef_base_parser,
            profile_base_parser,
        ],
        help=profile_help,
    )

    subparsers.add_parser(
        "eval",
        parents=[
            parsing_base_parser,
            optimization_base_parser,
            hef_base_parser,
            evaluation_base_parser,
        ],
        help="infer the model using the Hailo Emulator or the Hailo hardware and produce the model accuracy.",
    )

    subparsers.add_parser(
        "validate",
        parents=[
            oddbot_base_parser,
            validation_base_parser,
            parsing_base_parser,
        ],
        help="(Odd.Bot) Validate the performance of a compiled Hailo model against the original PyTorch model.",
    )

    subparsers.add_parser(
        "compile_and_validate",
        parents=[
            oddbot_base_parser,
            optimization_base_parser,
            validation_base_parser,
            parsing_base_parser,
            compile_and_validate_base,
        ],
        help="(Odd.Bot) Full model conversion pipeline, including optimization and performance validation.",
    )

    # add parsers for plugins
    for command in HMZ_COMMANDS:
        command_parser = command.parser_fn()
        subparsers.add_parser(command.name, parents=[command_parser], help=command_parser.description)
    return parser


def _get_mlflow_run_id(exp_name, run_name):
    mlflow.set_tracking_uri("http://localhost:5001")
    runs =  mlflow.search_runs(experiment_names=[exp_name],
                               filter_string=f"tags.mlflow.runName = '{run_name}'")
    n_hits = len(runs)
    if n_hits != 1:
        raise ValueError(f"Run name '{run_name}' matched {n_hits} runs in experiment '{exp_name}'. Please use --mlflow_exp_name and --mlflow_run_id to specify correct MLFlow run or use --no_mlflow.")
    else:
        run_id = runs.iloc[0]["run_id"]
        return run_id


def _process_config_file(args, command='compile_and_validate'):

    with open(args.config, 'r') as config_file:
        config = yaml.safe_load(config_file)

    if args.pt_filepath is None:
        args.pt_filepath = f'{config["input"]["folder_of_training_run"]}/weights/{config["base"]["model_filename"]}'
    if args.results_dir is None:
        args.results_dir = Path(args.pt_filepath.split("/weights")[0])
        args.results_dir = args.results_dir / "hailo"
        args.results_dir.mkdir(exist_ok=True)

    train_config_path = str(args.results_dir).replace("hailo", "args.yaml")
    with open(train_config_path, 'r') as train_config_file:
        train_config = yaml.safe_load(train_config_file)
    imgsize = (train_config["imgsz"], train_config["imgsz"])
    output_name = train_config["name"]

    if not args.no_mlflow:
        args.mlflow_exp_name = train_config['project'] if args.mlflow_exp_name is None else args.mlflow_exp_name
        args.mlflow_run_id = _get_mlflow_run_id(args.mlflow_exp_name, output_name) if args.mlflow_run_id is None else args.mlflow_run_id

    args.imgsize = imgsize if args.imgsize is None else args.imgsize
    args.classes = config["base"]["classes"] if args.classes is None else args.classes
    args.nms_scores_th = train_config["conf"] if args.nms_scores_th is None else args.nms_scores_th
    args.nms_iou_th = train_config["iou"] if args.nms_iou_th is None else args.nms_iou_th
    args.hw_arch = config["base"]["hw_arch"] if args.hw_arch is None else args.hw_arch

    if command.startswith('compile'):
        args.yaml_path = config["base"]["hailo_arch_yaml"] if args.yaml_path is None else args.yaml_path
        args.output_name = output_name if args.output_name is None else args.output_name
        args.calib_path = config["input"]["calib_path"] if args.calib_path is None else args.calib_path
        args.optimization_level = config["compile"]["optimization_level"] if args.optimization_level is None else args.optimization_level
        args.compression_level = config["compile"]["compression_level"] if args.compression_level is None else args.compression_level

    if command.endswith('validate'):
        args.har_filepath = f"{args.results_dir}/{output_name}.har" if args.har_filepath is None else args.har_filepath
        args.data_yaml = train_config["data"].replace("/home/thomas/Documents", "") if args.data_yaml is None else args.data_yaml
        args.ground_truth_src = config["validate"]["ground_truth_src"] if args.ground_truth_src is None else args.ground_truth_src
        args.similarity_th = config["validate"]["similarity_th"] if args.similarity_th is None else args.similarity_th
        args.val_iou_th = config["validate"]["val_iou_th"] if args.val_iou_th is None else args.val_iou_th
        args.vis_error_th = config["validate"]["vis_error_th"] if args.vis_error_th is None else args.vis_error_th

    args.folder_of_model_registry = config["base"]["folder_of_model_registry"]
    return args


def run(args):
    # search for commands from plugins
    command_to_handler = {command.name: command.fn for command in HMZ_COMMANDS}
    if args.command in command_to_handler:
        return command_to_handler[args.command](args)

    # we make sure to only import these now to keep loading & plugins fast
    from hailo_model_zoo.main_driver import compile, evaluate, optimize, parse, profile, validate, compile_and_validate

    handlers = {
        "parse": parse,
        "optimize": optimize,
        "compile": compile,
        "profile": profile,
        "eval": evaluate,
        "validate": validate, 
        "compile_and_validate": compile_and_validate
    }

    args = _process_config_file(args, command=args.command)
    return handlers[args.command](args)


def main():
    parser = _create_args_parser()
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    # from this point we can import heavy modules
    run(args)


if __name__ == "__main__":
    main()
