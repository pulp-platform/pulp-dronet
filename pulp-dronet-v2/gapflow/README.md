# Requirements

## Install GAP SDK (at least v3.9)

_Tested versions of the sdk are: 3.9

You must install the GAP sdk to use the GVSoC or to run the code on GAP8 SoC. Here you can find the basic steps to install the GAP sdk.

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


#### How can I use the GAP sdk a second time?:

after installing, the only commands you need to run on a fresh new terminal in order to use the gap sdk are:

```
cd gap_sdk
source configs/ai_deck.sh 
```

## Running the code
As a prerequisitve for both A. and B., it is important to firstly generate the onnx models out of the pyTorch .pth files. To do so, go in the root folder and use the command:
```
python3 onnx_converter.py
```
This fetches the .pth model files from the "trained_models" folder to generate the onnx files required by the GapFlow.
The onnx models are exported to "nntool_input/models_onnx".

**A. Evaluating the accuracy of the model**

1. In file "nntool_model_eval.py" adjust line 38 to point to the path where you stored the dataset (i.e., the folder that contains the "testing" and "himax" folders).
2. To check accuracy of the quantized model, run the following commands:
```
make clean
make nntool_model_evaluation
```
By default, the evaluation uses the model trained on the Original+Himax dataset. In case the evaluation of the model trained only on the Original dataset is desired, run:
```
make clean
make nntool_model_evaluation TRAINING_DATASET=ORIGINAL
```
The scores will appear on the screen after the inference in completed. This can take about 1h for the whole testing dataset.

**B. Deploying the running the model**

To generate, compile and run the one-frame inference, run the following:
```
make clean
make all platform=gvsoc
make run platform=gvsoc
```
The default the framework uses the model trained on the Original+Himax dataset. In case the model trained only on the Original dataset is desired, run:
```
make clean
make all platform=gvsoc TRAINING_DATASET=ORIGINAL
make run platform=gvsoc
```
Note: Steps A. and B. are independent. Therefore, executing B does not require to firstly execute A.
