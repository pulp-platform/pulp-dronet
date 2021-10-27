# PULP-Dronet V2

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
    ├── LICENSE.apache.md
    └── README.md
```

# Requirements 

### Python dependences

Install all the python dependences (for CNN training and testing):

```
conda env create -f conda_deps.yml
```

### Install NEMO (quantization tool)

NEMO can be installed with `pip install pytorch-nemo` and it is already included in the python dependences `conda_deps.yml`. Therefore, no further actions required.

### Clone DORY (deployment tool)

[DORY](https://github.com/pulp-platform/dory) is the automatic tool for deploying DNNs on low-cost MCUs with typically less than 1MB of on-chip SRAM memory.

```
git clone https://github.com/pulp-platform/dory
cd dory/
git clone https://github.com/pulp-platform/dory_examples.git
git clone https://github.com/pulp-platform/pulp-nn.git```
```

_Tested on this commit: a98227263db3f9fd4ba7eca85d4210acee3a4af3_

### Install the GAP sdk

_Full GWT guide can be found at: https://greenwaves-technologies.com/setting-up-sdk/_

We provide some basic instructions at the end of the README: ([Go to GAP sdk setup](#gap-sdk))

### Download and prepare the dataset

The PULP-DroNet dataset is composed by the following three sub-sets:

* [Himax Dataset](https://github.com/pulp-platform/Himax_Dataset) (only testing set);
* [Udacity Dataset](https://github.com/udacity/self-driving-car/tree/master/datasets/CH2);
* [Zurich Bicycle Dataset](http://rpg.ifi.uzh.ch/dronet.html).

The original Udacity and Zurich Bicycle datasets were previously released in their respective open-source projects. 

The Udacity dataset and the Zurich Bicycle dataset must be pre-processed to match out HIMAX configuration, and the hierarchy of the files must be re-arranged. 


**Udacity dataset: download and reorganize**:

_Please follow the step-by-step guide of the original [DroNet](https://github.com/uzh-rpg/rpg_public_dronet) repository for downloading and extracting the Udacity and Zurich bicycle datasets._

Our summary:

- Download the three torrent files from [here](https://github.com/udacity/self-driving-car/tree/master/datasets/CH2): Ch2_001.tar.gz.torrent, Ch2_002.tar.gz.torrent, HMB_3.bag.tar.gz.torrent
- Once downloaded, you get:

```
Ch2_001.tar.gz   (4.4 GB)-> includes the training set HMB_1 HMB_2 HMB_4 HMB_5 HMB_6
Ch2_002.tar.gz   (456 MB)-> includes the testing set HMB_3 (no labels!) -> discard
HMB_3.bag.tar.gz (896 MB)-> testing set HMB_3 (with labels!)
```

- untar all files: `tar -xvf file_name.tar.gz`

```
Ch2_001 and HMB_3.bag.tar.gz, once extracted:
HMB.txt	  	  -> info on dataset, discard
HMB_1.bag 	  -> bag file to be extracted with udacity-driving-reader
HMB_2.bag 	  -> bag file to be extracted with udacity-driving-reader
HMB_3.bag 	  -> bag file to be extracted with udacity-driving-reader
HMB_4.bag 	  -> bag file to be extracted with udacity-driving-reader
HMB_5.bag 	  -> bag file to be extracted with udacity-driving-reader
HMB_6.bag 	  -> bag file to be extracted with udacity-driving-reader
```

- Create a directory for each file: `mkdir -p ./extract/HMB_1 ./extract/HMB_2 ./extract/HMB_3 ./extract/HMB_4 ./extract/HMB_5 ./extract/HMB_6`
- Extract the .bag files with [udacity-driving-reader](https://github.com/rwightman/udacity-driving-reader) in these separate folders (then we will merge them). for example:

```
mv Ch2_002/HMB_1.bag  extract/HMB_1/
cd udacity-driving-reader/
./run-bagdump.sh -i /<absolute_path_to>/extract/HMB_1/ -o /<absolute_path_to>/extract/HMB_1/ -- -f png
```
_How to use run-bagdump.sh:_  `./run-bagdump.sh -i [absolute dir with folders containing bag files] -o [absolute output dir] -- [args to pass to python script]`


- Clean unecessary files

```
cd extract/HMB_1/
rm -r brake.csv camera.csv gear.csv gps.csv HMB_2.bag HMB_2.yaml imu.csv steering.csv throttle.csv left/ right/
```

- We are left wit only `center/` folder and `interpolated.csv`
- Rename the `center/` folder to `images/
- As explained in [DroNet](https://github.com/uzh-rpg/rpg_public_dronet), process the `interpolated.csv` labels file with this [time_stamp_matching.py](https://github.com/uzh-rpg/rpg_public_dronet/blob/master/data_preprocessing/time_stamp_matching.py) script.

- Process the images to convert them to the HIMAX format:cropping (center bottom) to 200x200 pixels format, conversion to grapyscale colormap, conversion to jpeg format

**Zurich Bicycles dataset: download and reorganize**

- download from [here](http://rpg.ifi.uzh.ch/dronet.html).
- Process the images to convert them to the HIMAX format:cropping (center bottom) to 200x200 pixels format, conversion to grapyscale colormap, conversion to jpeg format


**The final hierarchy of the dataset files:**

```
.
├── Readme.md
|
├── fine_tuning/
│   ├── DSCN2561/ # Zurich_Bicycle_Dataset
│   ├── ...
│   ├── DSCN2697/
│   ├── GOPR0201/
│   ├── ...
│   ├── GOPR0387/
|
│   ├── HMB_1_3900/ # Udacity_Dataset
│   ├── ...
│   ├── HMB_6/
|
│   ├── test_01/ # Himax_Dataset
│   ├── ...
│   └── test_24/
├── himax/
│   └── jpg/
│       └── testing/
│           ├── test_02/ # Himax_Dataset
│           ├── ...
│           └── test_23/
├── testing/
│   ├── DSCN2571/ # Zurich_Bicycle_Dataset
│   ├── GOPR0200/
│   ├── ...
│   ├── GOPR0386/
|   |
│   └── HMB_3/ # Udacity_Dataset
├── training/
│   ├── DSCN2561/ # Zurich_Bicycle_Dataset
│   ├── ...
│   ├── DSCN2697/
│   ├── GOPR0201/
│   ├── ...
│   ├── GOPR0387/
|
│   ├── HMB_1_3900/ # Udacity_Dataset
│   ├── ...
│   ├── HMB_6/
|
│   ├── test_01/ # Himax_Dataset
│   ├── ...
│   └── test_24/
|
└── validation/
    ├── DSCN2682/ # Zurich_Bicycle_Dataset
    ├── GOPR0227/
    |
    └── HMB_1_501# Zurich_Bicycle_Dataset
```

#### Important note

The himax dataset folders (`test_*`) must be named with 2 figures, for example: test_01 and **not** test_1.

# How to use

All the python scripts (training.py, testing.py, evaluation.py, quantize.py) take default values of variables from the config.py file. Each argument added by command line will override default values.

## Training

```
python training.py --data_path=/path/to/pulp_dronet_dataset --dataset=original_and_himax --flow=nemo_dory --gpu=0 --batch_size=32
```

## Evaluating all the weights of a training session

When the "--early_stopping" is disabled, the training script will save the weights of the network after each epoch (by default in the "checkpoints/pulp_dronet_v2" folder)
evaluation.py script provides a way to test (default:validation dataset) all these weights saved for each training session. After this, you can manually select the best performing set of weights.

```
python evaluation.py --data_path=/path/to/pulp_dronet_dataset --dataset=validation --flow=nemo_dory --gpu=0 --batch_size=32 --cherry_picking_path=checkpoints/pulp_dronet_v2/
```

## Testing

Testing on the original dataset only (Udacity and Zurich bicycle datasets) will provide performances of both Accuracy (collision) and RMSE (steering angle)

```
python testing.py --data_path=/path/to/pulp_dronet_dataset --dataset=original --flow=nemo_dory --gpu=0 --batch_size=32
```

Testing on the HIMAX dataset only will provide performances of Accuracy only (kust "collision" labels, no "steering angle" labels)

```
python testing.py --data_path=/path/to/pulp_dronet_dataset --dataset=himax --flow=nemo_dory --gpu=0 --batch_size=32
```

## Deployment flow: NEMO/DORY and GAP8 (GVSoC) flow: quickstart  

How to run PULP-DroNet on GAP8 or GVSoC in three steps, starting from a pretrained model. 

**NEMO (quantization):**
- **Input**: model definition (pytorch format, can be found in "models/dronet_v2_nemo_dory.py") + pre-trained weights (".pth file", can be found in "models/dronet_v2_dory.pth" ) 
- **Output**: ONNX graph model (including weights) + golden activations (".txt" files, used by DORY for checksums)

**DORY (generation of optimized C code):**
- **Input**: ONNX graph model + golden activations (".txt" files)
- **Output**: optimized C code for deployment on GAP8, generated in the "dory_examples/application/" folder

**GAP8 (run on platform):**
- **Input**: optimized C code generated by DORY (dory_examples/application/" folder)

### Detailed steps:

**1. Generate the onnx model with nemo script**
```
conda activate your_env
python quantize.py --data_path=/path/to/your/pulpdronet/dataset  --export_path=./nemo_output/
```

**2. Use DORY to generate the C code** 

DORY generates the deployment C code  under the "dory_examples/application/" folder:

```
cd /dory/dory_example/
conda activate your_env
python network_generate.py --network_dir ../nemo_output/ --verbose_level Check_all+Perf_final --Bn_Relu_Bits 64 --l2_buffer_size 420000 --sdk=gap_sdk
```

**3. Build and run on GAP8 (or GVSoC)**

_remember: open a new terminal, source your sdk and export the cfg for your debugger_

_remember: your gap sdk (or pulp sdk) must be correctly installed before you try to run on GAP8_ ([Go to GAP sdk setup](#gap-sdk))

```
source gap_sdk/configs/ai_deck.sh
export GAPY_OPENOCD_CABLE=$HOME/work/gap_sdk/tools/gap8-openocd/tcl/interface/ftdi/olimex-arm-usb-ocd-h.cfg 
```

then run PULP-DroNet on GAP8 **: )**

```
cd dory/dory_example/application
make clean all run CORE=8 platform=gvsoc  (GVSoC)
make clean all run CORE=8 platform=board  (GAP8)
```


# Bonus: Install GAP SDK 

_Tested versions of the sdk are: 3.8.1_

You must install the GAP sdk (or the [PULP sdk](https://github.com/pulp-platform/pulp-sdk)) to use GVSoC or to run code on GAP8 SoC. Here you can find the basic steps to install the GAP sdk.

_Full GWT guide can be found at: https://greenwaves-technologies.com/setting-up-sdk/_

#### Prerequisites
```
sudo apt-get install -y build-essential git libftdi-dev libftdi1 doxygen python3-pip libsdl2-dev curl cmake libusb-1.0-0-dev scons gtkwave libsndfile1-dev rsync autoconf automake texinfo libtool pkg-config libsdl2-ttf-dev
```

Tip: create also a conda environment (and call it gap_sdk) to install all the packets needed by the sdk

```
conda create --name gap_sdk python numpy cython
conda activate gap_sdk
```

#### Toolchain

```
git clone https://github.com/GreenWaves-Technologies/gap_riscv_toolchain_ubuntu_18.git
cd ~/gap_riscv_toolchain_ubuntu_18
./install.sh
```

**TIP**: if you chose a custom path for install, add this to your .bashrc file:

> export GAP_RISCV_GCC_TOOLCHAIN="custom/path/that/you/chose"

#### Build GAP SDK 

```
git clone https://github.com/GreenWaves-Technologies/gap_sdk.git
cd gap_sdk
git submodule update --init --recursive
source configs/ai-ai_deck.sh 
```

```
pip install -r requirements.txt
pip install -r tools/nntool/requirements.txt
```

```
make sdk
```

**IMPORTANT:** always run the sourceme.sh in a fresh new terminal. Never source this file two times in the same terminal (might have issues)


#### TEST: To check everything works

**Test on GVSoC**
```
cd examples/pmsis/helloworld
make clean all run platform=gvsoc
```

**Test on the board**

```
export GAPY_OPENOCD_CABLE=interface/ftdi/olimex-arm-usb-ocd-h.cfg
cd examples/pmsis/helloworld
make clean all run platform=board
```

There are different cables setup by default for each board ([here the list of defices supported](https://github.com/GreenWaves-Technologies/gap_sdk/tree/master/tools/gap8-openocd/tcl/interface/ftdi)). In case you want to use a different cable, you can define this environment variable:

> GAPY_OPENOCD_CABLE=$HOME/gap_sdk/tools/gap8-openocd/tcl/interface/ftdi/olimex-arm-usb-ocd-h.cfg




#### How can I use the GAP sdk a second time?:

after installing, the only commands you need to run on a fresh new terminal in order to use the gap sdk are:

```
cd gap_sdk
source configs/ai_deck.sh 
GAPY_OPENOCD_CABLE=$HOME/gap_sdk/tools/gap8-openocd/tcl/interface/ftdi/olimex-arm-usb-ocd-h.cfg

```




