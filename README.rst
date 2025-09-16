Hailo Model Zoo
===============

Model Conversion for Odd.Bot
----------------------------
The Hailo Model Zoo was modified for: 

1. Converting Ultralytics' YOLO .pt files to Hailo .hef files
2. Validating the compiled model performance with the original
3. Uploading the compiled model + results to the model registry and MLFlow

Basic Usage
^^^^^^^^^^^
1. Navigate to the hailo folder

  .. code-block::

      cd ~/Documents/hailo

2. Update ``settings.yaml``

   - ``folder_of_training_run`` (always check!): Starting with /experiments (string)
   - ``calib_path`` (always check!): Path folder with images to be used for calibration (string)
   - ``optimization_level``: How hard to try to make the .hef model similar to the .pt one (\[0,4\], higher = less degration = longer compilation time)
   - ``compression_level``: How much to compress the model (\[0,5\], higher = more degradation but faster inference)
   - ``ground_truth_src``: What ground truth to compare the Hailo predictions against. Either PyTorch predictions (pt) or Supervisely annotation (sly)  
   - ``similarity_th``: What proportion of predictions must match the ground truth to pass validation (float=0.95)
   - ``val_iou_th``: The IoU threshold used for matching bounding boxes (float=0.65)
   - ``vis_error_th``: If specified, saves all images with more than this number of mismatches (int=5)
   - ``model_filename``: Which model file to convert (string='best.pt')
   - ``classes``: Number of detection classes (int=2)
   - ``hw_arch``: Hailo hardware type (string='hailo8')
   - ``hailo_arch_yaml``: Path to neural net configuration file for Hailo (string, depending on architecture)
   - ``folder_of_model_registry``: Model registry (string='/model_registry')

3. Start the model conversion

   .. code-block::
      
      docker compose run -d hailo_model_conversion


This might take a few hours, depending on the optimization level. There are two possible outcomes:

* Success: The model was successfully converted, obtaining a similar performance to that of the original model. The .hef file + model details are now in the model registry.
* Failure: The validation failed. To see the results, go to the ``folder_of_training_run`` as specified in ``settings.yaml`` and inspect ``degradation.png``. In case the problem occurs before validation, please find the docker container ID with ``docker ps -a`` and check the logs using ``docker logs <container_id>``.


Under the hood, the ``hailo_model_conversion`` Docker service just opens the Docker container and runs the ``hailomz compile_and_validate`` command as described under 'Advanced Usage'.
Consequently, one can add extra flags to the command in ``compose.yaml`` (e.g. specifying a specific MLFlow run ID using ``--mlflow_run_id 1234``)


Advanced usage
^^^^^^^^^^^^^^
Inside of the container, the conversion script can be used more flexibly. Here, one can choose between the following commands: 

* ``hailomz compile_and_validate``: The full conversion + validation pipeline.
* ``hailomz compile``: Only model conversion from YOLO .pt to .hef.
* ``hailomz validate``: Validating the performance of a .hef model against a .pt one.

The Docker service in 'Basic Usage' runs this command with the ``--config settings.yaml`` flag. This settings file is meant to make usage as easy as possible.
For more flexible usage options, one can add other flags (run ``hailomz <command> --help``) that provide more low-level settings. 
For example, with ``settings.yaml`` the IoU threshold is inferred, while you could also explicitly specify it with ``--nms_iou_th``.

.. |python| image:: https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue.svg
   :target: https://www.python.org/downloads/release/python-380/
   :alt: Python 3.8
   :width: 150
   :height: 20


.. |tensorflow| image:: https://img.shields.io/badge/Tensorflow-2.12.0-blue.svg
   :target: https://github.com/tensorflow/tensorflow/releases/tag/v2.12.0
   :alt: Tensorflow
   :width: 110
   :height: 20


.. |cuda| image:: https://img.shields.io/badge/CUDA-11.8-blue.svg
   :target: https://developer.nvidia.com/cuda-toolkit
   :alt: Cuda
   :width: 80
   :height: 20


.. |compiler| image:: https://img.shields.io/badge/Hailo%20Dataflow%20Compiler-3.31.0-brightgreen.svg
   :target: https://hailo.ai/company-overview/contact-us/
   :alt: Hailo Dataflow Compiler
   :width: 180
   :height: 20


.. |runtime| image:: https://img.shields.io/badge/HailoRT%20(optional)-4.21.0-brightgreen.svg
   :target: https://hailo.ai/company-overview/contact-us/
   :alt: HailoRT
   :width: 170
   :height: 20


.. |license| image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://github.com/hailo-ai/hailo_model_zoo/blob/master/LICENSE
   :alt: License: MIT
   :width: 80
   :height: 20


.. image:: docs/images/logo.png

|python| |tensorflow| |cuda| |compiler| |runtime| |license|


The Hailo Model Zoo provides pre-trained models for high-performance deep learning applications. Using the Hailo Model Zoo you can measure the full precision accuracy of each model, the quantized accuracy using the Hailo Emulator and measure the accuracy on the Hailo-8 device. Finally, you will be able to generate the Hailo Executable Format (HEF) binary file to speed-up development and generate high quality applications accelerated with Hailo-8. The Hailo Model Zoo also provides re-training instructions to train the models on custom datasets and models that were trained for specific use-cases on internal datasets.

Models
Hailo provides different pre-trained models in ONNX / TF formats and pre-compiled HEF (Hailo Executable Format) binary file to execute on the Hailo devices.

The models are divided to:

* Public models - which were trained on publicly available datasets.

    * For Hailo-8 - `Classification <docs/public_models/HAILO8/HAILO8_classification.rst>`_, `Object Detection <docs/public_models/HAILO8/HAILO8_object_detection.rst>`_, `Segmentation <docs/public_models/HAILO8/HAILO8_semantic_segmentation.rst>`_, `other tasks <docs/PUBLIC_MODELS.rst>`_

    * For Hailo-8L - `Classification <docs/public_models/HAILO8L/HAILO8L_classification.rst>`_, `Object Detection <docs/public_models/HAILO8L/HAILO8L_object_detection.rst>`_, `Segmentation <docs/public_models/HAILO8L/HAILO8L_semantic_segmentation.rst>`_, `other tasks <docs/PUBLIC_MODELS.rst>`_

    * For Hailo-15H - `Classification <docs/public_models/HAILO15H/HAILO15H_classification.rst>`_, `Object Detection <docs/public_models/HAILO15H/HAILO15H_object_detection.rst>`_, `Segmentation <docs/public_models/HAILO15H/HAILO15H_semantic_segmentation.rst>`_, `other tasks <docs/PUBLIC_MODELS.rst>`_

    * For Hailo-15M - `Classification <docs/public_models/HAILO15M/HAILO15M_classification.rst>`_, `Object Detection <docs/public_models/HAILO15M/HAILO15M_object_detection.rst>`_, `Segmentation <docs/public_models/HAILO15M/HAILO15M_semantic_segmentation.rst>`_, `other tasks <docs/PUBLIC_MODELS.rst>`_

    * For Hailo-10H - `Classification <docs/public_models/HAILO10H/HAILO10H_classification.rst>`_, `Object Detection <docs/public_models/HAILO10H/HAILO10H_object_detection.rst>`_, `Segmentation <docs/public_models/HAILO10H/HAILO10H_semantic_segmentation.rst>`_, `other tasks <docs/PUBLIC_MODELS.rst>`_



* | `HAILO MODELS <docs/HAILO_MODELS.rst>`_ which were trained in-house for specific use-cases on internal datasets.
  | Each Hailo Model is accompanied with retraining instructions.


Retraining
----------

Hailo also provides `RETRAINING INSTRUCTIONS <docs/RETRAIN_ON_CUSTOM_DATASET.rst>`_ to train a network from the Hailo Model Zoo with custom dataset.

Benchmarks
----------

| List of Hailo's benchmarks can be found in `hailo.ai <https://hailo.ai/developer-zone/benchmarks/>`_.
| In order to reproduce the measurements please refer to the following `page <docs/BENCHMARKS.rst>`_.


Quick Start Guide
------------------


* Install Hailo Dataflow Compiler and enter the virtualenv. In case you are not Hailo customer please contact `hailo.ai <https://hailo.ai/company-overview/contact-us/>`_
* Install HailoRT (optional). Required only if you want to run on Hailo-8. In case you are not Hailo customer please contact `hailo.ai <https://hailo.ai/company-overview/contact-us/>`_
* Clone the Hailo Model Zoo


  .. code-block::

      git clone https://github.com/hailo-ai/hailo_model_zoo.git

* Run the setup script


  .. code-block::

     cd hailo_model_zoo; pip install -e .

* Run the Hailo Model Zoo. For example, print the information of the MobileNet-v1 model:


  .. code-block::

     hailomz info mobilenet_v1

Getting Started
^^^^^^^^^^^^^^^

For full functionality please see the `INSTALLATION GUIDE <docs/GETTING_STARTED.rst>`_ page (full install instructions and usage examples). The Hailo Model Zoo is using the Hailo Dataflow Compiler for parsing, model optimization, emulation and compilation of the deep learning models. Full functionality includes:


* | Parse: model translation of the input model into Hailo's internal representation.
* | Profiler: generate profiler report of the model. The report contains information about your model and expected performance on the Hailo hardware.
* | Optimize: optimize the deep learning model for inference and generate a numeric translation of the input model into a compressed integer representation.
  | For further information please see our `OPTIMIZATION <docs/OPTIMIZATION.rst>`_ page.
* | Compile: run the Hailo compiler to generate the Hailo Executable Format file (HEF) which can be executed on the Hailo hardware.
* | Evaluate: infer the model using the Hailo Emulator or the Hailo hardware and produce the model accuracy.

For further information about the Hailo Dataflow Compiler please contact `hailo.ai <https://hailo.ai/company-overview/contact-us/>`_.


.. figure:: docs/images/usage_flow.svg


License
-------

The Hailo Model Zoo is released under the MIT license. Please see the `LICENSE <https://github.com/hailo-ai/hailo_model_zoo/blob/master/LICENSE>`_ file for more information.

Support & Issues
----------------

⚠️ **Issue reporting is disabled in this repository.**

For bug reports, feature requests, or discussions, please visit our `Hailo Community Forum <https://community.hailo.ai/>`_.

Changelog
---------

For further information please see our `CHANGELOG <docs/CHANGELOG.rst>`_ page.
