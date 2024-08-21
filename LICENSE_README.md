# PULP-Dronet v1
The project's structure is the following:

```
├── pulp-dronet-v1/
│   ├── bin/
│   │   ├── PULPDroNet_GAPuino
│   │   ├── PULPDroNet_PULPShield
│   ├── dataset/
│   │   ├── Himax_Dataset/
│   │   ├── Udacity_Dataset/
│   │   ├── Zurich_Bicycle_Dataset/
│   ├── imgs/
│   │   ├── PULP_dataset.png
│   │   ├── PULP_drone.png
│   │   ├── PULP_proto.png
│   │   ├── PULP_setup.png
│   ├── PULP-Shield/
│   │   ├── GAP8/
│   │   ├── jtag-convboard/
│   ├── src/
│   │   ├── autotiler/
│   │   ├── config.h
│   │   ├── config.ini
│   │   ├── PULPDronet.c
│   │   ├── PULPDronetGenerator.c
│   │   ├── PULPDronetKernels.c
│   │   ├── PULPDronetKernels.h
│   │   ├── PULPDronetKernelsInit.c
│   │   ├── PULPDronetKernelsInit.h
│   │   ├── Makefile
│   │   ├── run_dataset.sh
│   ├── weights/
│   │   ├── binary/
│   │   ├── WeightsPULPDroNet.raw
│   ├── LICENSE.apache.md
│   ├── LICENSE_README.md
│   ├── README.md
```

All the folders, sub-folders and files in this project except for the content of the following two folders:

* `pulp-dronet-v1/dataset/Udacity_Dataset`
* `pulp-dronet-v1/dataset/Zurich_Bicycle_Dataset`

are released open-source with the Apache 2.0 license.
A copy of the Apache 2.0 license is included in `pulp-dronet/LICENSE.apache.md` and in `pulp-dronet-v1/dataset/Himax_Dataset/LICENSE.apache.md`.

All the files in `pulp-dronet-v1/dataset/Udacity_Dataset` and  `pulp-dronet-v1/dataset/Zurich_Bicycle_Dataset` are released with the MIT License included in the same folder as `LICENSE.mit.md`.

# PULP-Dronet v2
The project's structure is the following:

The project's structure is the following:

```
.
└── pulp-dronet-v2/
    ├── dataset/
    │   ├── fine_tuning/
    │   ├── himax/
    │   ├── testing/
    │   ├── training/
    │   └── validation/
    ├── gapflow/
    ├── imgs/
    │   └── PULP_drone.png
    ├── model/
    │   ├── dronet_v2_gapflow.py
    │   ├── dronet_v2_nemo_dory.py
    │   ├── dronet_v2_gapflow_original.pth
    │   ├── dronet_v2_gapflow_original_himax.pth
    │   ├── dronet_v2_nemo_dory_original.pth
    │   └── dronet_v2_nemo_dory_original_himax.pth
    ├── nemo-dory/
    │   └── quantize.py
    ├── conda_deps.yml
    ├── config.py
    ├── evaluation.py
    ├── testing.py
    ├── training.py
    ├── utility.py
    ├── LICENSE
    └── README.md
```
All the folders, sub-folders and files in this project are released open-source with the Apache 2.0 license.

A copy of the Apache 2.0 license is included in `pulp-dronet/LICENSE`.



# Tiny-PULP-Dronet v2'3

The project's structure is the following:

```
.
└── tiny-pulp-dronet-v3/
    ├── dataset/ # directory where you must put the PULP-Dronet v3 dataset
    │   ├── training/ # put here the training dataset, dowloaded from Zenodo
    │   └── testing/  # put here the testing dataset, dowloaded from Zenodo
    ├── drone-applications/ # code running on the Bitcraze Crazyflie 2.1 nano-drone and the GAP8 SoC
    │   ├── crazyflie-dronet-app/ # Crazyflie STM32 code (flight controller)
    │   │   ├── inc/ ...
    │   │   ├── src/ ...
    │   │   └── Makefile
    |   ├── external/
    |   |    └── crazyflie-firmware # use tag 2022.01
    │   └── gap8-dronet-app/ # AI-Deck GAP8 code (CNN processing + camera). Use gap-sdk 3.9.1
    │       ├── pulp-dronet-v3/ # main running pulp-dronet v3 (19fps, 320kB)
    │       │   ├── DORY_network/
    │       │   │   ├── inc/ ...
    │       │   │   └── src/ ...
    │       │   └── Makefile
    │       └── tiny-pulp-dronet-v3/ # main running tiny-pulp-dronet v3 (139fps, 2.9kB)
    │           ├── DORY_network/
    │           │   ├── inc/ ...
    │           │   └── src/ ...
    │           └── Makefile
    ├── imgs/ # images for readme
    ├── model/
    │   ├── dronet_v3.py # pytorch definition of the PULP-Dronet v3 CNN.
    │   ├── pulp-dronet-v3-resblock-1.0.pth # pre-trained pytorch model of PULP-Dronet v3
    │   └── tiny-pulp-dronet-v3-dw-pw-0.125.pth # pre-trained pytorch model of Tiny-PULP-Dronet v3
    ├── nemo-dory/
    │   ├── nemo/ # external: NEMO tool for quantization of CNNs
    │   └── dory_dronet/  # external: DORY tool for deployment of CNNs on MCUs
    ├── training/ # directory where all training checkpoints, logs, and tensorboard files are saved.
    ├── classes.py # class definitions used in the project.
    ├── conda_deps.yml # conda environment file.
    ├── config.py # default args for .py files.
    ├── README.md
    ├── testing.py # Test a pre-trained model on the testing set in pytorch
    ├── training.py # training and validation of a PULP-Dronet v3 CNN.
    └── utility.py # utility for python scripts: dataset loader, loss functions, checkpoint loading
```

All the folders, sub-folders and files in this project are released open-source with the Apache 2.0 license.

A copy of the Apache 2.0 license is included in `pulp-dronet/LICENSE`.

External modules, which can be downloaded from external github reopositories, are released under different licenses:
* `pulp-dronet/tiny-pulp-dronet-v3/drone-applications/external/crazyflie-firmware` is released under GNU General Public License v3.0


# PULP-Dronet v3 Dataset

We release the dataset ([zenodo.org/records/13348430](https://zenodo.org/records/13348430)) as open source under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.


# Dataset collector framework

The project's structure is the following:

```
dataset_collection_framework
├── conda_deps.yml
├── crazyflie_firmware_patches/
│   ├── crazyflie-firmware-modified-dataset/
│   └── dataset_collection_app/
│       ├── current_platform.mk
│       ├── datasetcollector.c
│       ├── datasetcollector.h
│       └── Makefile
├── dataset_collector_python/
│   ├── control_config.json
│   ├── CrazyflieCameraStreamer.py
│   ├── CrazyflieCommunicator.py
│   ├── CrazyflieController.py
│   ├── datasetcollector_main.py
│   ├── DatasetLogger.py
│   ├── dataset_tools/
│   │   ├── count_images_dataset.sh
│   │   ├── dataset_partitioning.py
│   │   ├── dataset_post_processing.py
│   │   ├── image_viewer_cv.py
│   │   ├── image_viewer.py
│   │   ├── save_all_videos.sh
│   │   └── UI/
│   │       ├── create_ui_python_code.sh*
│   │       ├── imageViewer.ui
│   │       └── imageViewerUI.py
│   ├── default_config.json
│   ├── draw_binary_digits.py
│   ├── ds_logger.py
│   ├── logger_app.py
│   └── UI/
│       ├── controlConfigDialog.py
│       ├── controlConfigDialog.ui
│       ├── coordinate_system.jpg
│       ├── create_ui_python_code.sh*
│       ├── datasetcollector.ui
│       ├── datasetcollector_ui.py
│       ├── logConfigDialog.py
│       └── logConfigDialog.ui
├── external/
├── GAP8_streamer/
│   ├── inc/
│   │   └── frame_streamer.h
│   └── wifi_jpeg_streamer/
│       ├── config.ini
│       ├── Makefile
│       └── wifi_frame_streamer.c
├── gap_sdk_setup.md
├── images/
└── README.md
```


All the folders, sub-folders and files in this project are released open-source with the Apache 2.0 license.

A copy of the Apache 2.0 license is included in `pulp-dronet/LICENSE`.

External modules, which can be downloaded from external github reopositories, are released under different licenses:
* `/pulp-dronet/dataset_collection_framework/crazyflie_firmware_patches/crazyflie-firmware-modified-dataset` is released under GNU General Public License v3.0
* `/pulp-dronet/dataset_collection_framework/external/crazyflie-firmware` is released under GNU General Public License v3.0.

# Dataset visualizer

The project's structure is the following:

```
dataset_visualizer
├── dataset_visualizer/
│   ├── acquisition_visualizer.py
│   ├── classes.py
│   ├── dataset_partitioning.py
│   ├── dataset_post_processing.py
│   ├── image_visualizer.py
│   ├── main.py
│   ├── metadata_visualizer.py
│   ├── plot_statistics.py
│   ├── stats_config.json
│   └── video.py
├── imgs/
│   ├── dataset_visualizer_grid_view.png
│   └── dataset_visualizer.png
├── README.md
└── setup.py
```

All the folders, sub-folders and files in this project are released open-source with the Apache 2.0 license.
A copy of the Apache 2.0 license is included in `pulp-dronet/LICENSE`.
