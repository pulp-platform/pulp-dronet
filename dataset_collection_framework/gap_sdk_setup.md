# Dataset building framework for PULP-based nano-drones

Before you can start using the software you should setup the environment with all the packages needed. The guide in the next section shows how to setup a virtual machine with the needed packages and software. The development environment was build upon the following technology stack:

Software:
- On Windows 10:
    - [Zadig (for installation of JTAG driver)](https://zadig.akeo.ie/)
    - [VirtualBox](https://www.virtualbox.org/wiki/Downloads) or [VMware Workstation Player 16](https://my.vmware.com/de/web/vmware/downloads/info/slug/desktop_end_user_computing/vmware_workstation_player/16_0#product_downloads)
- Crazyflie Software:
    - [Guest OS Xubuntu 18.04 including crazyflie software (Bionic Beaver, the BitcrazeVM_2018-12provided by bitcraze was used)](https://github.com/bitcraze/bitcraze-vm/releases/tag/2018.12)
    - [crazyflie-firmware (tag 2020.09)](https://github.com/bitcraze/crazyflie-firmware/tree/2020.09)
    - [crazyflie-clients-python (tag 2020.09.01)](https://github.com/bitcraze/crazyflie-clients-python/tree/2020.09.1)
- GreenWaves Technologies:
    - [gap-sdk realease v3.8.1](https://github.com/GreenWaves-Technologies/gap_sdk/tree/release-v3.8.1)
    - [gap8_opencd (commit b84d97ec4d2e601e704b54351e954b1c58d41683)](https://github.com/GreenWaves-Technologies/gap8_openocd/commit/b84d97ec4d2e601e704b54351e954b1c58d41683)
    - [gap_riscv_toolchain_ubuntu_18 (tag 1.4)](https://github.com/GreenWaves-Technologies/gap_riscv_toolchain_ubuntu_18/tree/1.4)
- [Anaconda](https://docs.anaconda.com/anaconda/install/linux/)

Hardware:
- Crazyflie
    - [Crazyflie 2.1](https://www.bitcraze.io/products/crazyflie-2-1/)
    - [Crazyradio-PA](https://www.bitcraze.io/products/crazyradio-pa/)
    - [Flow deck V2](https://www.bitcraze.io/products/flow-deck-v2/)
    - [Multi-ranger deck](https://www.bitcraze.io/products/multi-ranger-deck/)
    - [AI-Deck](https://store.bitcraze.io/products/ai-deck?variant=32072312750167)
- Olimex (JTAG adapter to program gap8 chip)
    - [ARM-USB-OCD-H](https://www.olimex.com/Products/ARM/JTAG/ARM-USB-OCD-H/)
    - [ARM JTAG 20 to 10 pin adapter](https://www.mouser.ch/ProductDetail/Olimex-Ltd/ARM-JTAG-20-10/?qs=DUTFWDROaMbVQp3WoAdijQ%3D%3D)
    - [USB A-B cable](https://www.brack.ch/delock-usb-2-0-kabel-a-b-easy-usb-2-m-262703?utm_source=google&utm_medium=cpc&utm_campaign=%21cc-pssh%21l-d%21e-g%21t-pla%21k1-it%21z-it_multimedia_channable&utm_term=&adgroup_id=95297775786&ad_type=pla&prod_id=262703&campaign_id=9422718872&gclid=Cj0KCQiA7NKBBhDBARIsAHbXCB4eDwFG6XQCziNHhPQsPRGSSuCkGP0RUmqMDGo8wSCZC4ZMAWzRsQcaAiQ5EALw_wcB&hc_fcv=YDUVHAKYAtRgXKIc~M2IJ69g30CpJU-6azzzzzzzz~LphVNdI3Akg7PU0szzzzzzzz)

# How to set up development environment
First the crazyflie environment is set up and afterwards the gap-sdk will be installed into the VM.
## Crazyflie SDK
Prerequisites:
- A VM Player is installed (e.g. [VMware Workstation Player 16](https://my.vmware.com/de/web/vmware/downloads/info/slug/desktop_end_user_computing/vmware_workstation_player/16_0#product_downloads) or [VirtualBox](https://www.virtualbox.org/))

1. Download the [bitcraze VM (2018.12)](https://github.com/bitcraze/bitcraze-vm/releases/tag/2018.12) ([downloadlink](https://files.bitcraze.se/dl/BitcrazeVM_2018.12.ova))
2. [Import the VM](https://docs.oracle.com/cd/E26217_01/E26796/html/qs-import-vm.html)
3. Before you can launch the VM you need to install the extension pack for USB2.0/3.0 support. The extension pack can downloaded [here](https://www.virtualbox.org/wiki/Downloads). You can install the package under: Tools > Preferences > Extensions.
4. Now launch the VM and open a terminal to checkout this repo with the following command.  
~~~~~shell
git clone https://iis-git.ee.ethz.ch/Drone-dev/dataset-building-framework-for-pulp-based-nano-drones.git --recursive
~~~~~

Try hello.c (follow this [tutorial](https://www.bitcraze.io/documentation/repository/crazyflie-firmware/master/development/howto/))  
**Bugs when trying hello.c:**  
./vendor/FreeRTOS/include/task.h line 1600, declaration of: void vApplicationStackOverflowHook( TaskHandle_t xTask,char * pcTaskName ); (not used in task.c) does not concide with definition in ./src/hal/src/freeRTOSdebig.c line 52: void vApplicationStackOverflowHook(xTaskHandle *pxTask, signed portCHAR *pcTaskName){...}  
Workaround: just comment out the declaration in task.h

Launch python client:  
~~~~~shell
python3 /home/bitcraze/bitcrazeSoftware/crazyflie-clients-python/bin/cfclient
~~~~~

## GAP8 SDK
Prerequisites:
- Running VM with Xubuntu 18.04 (Bionic Beaver)

The following steps are needed to install the gap-sdk:
1. Download and install [Anaconda](https://docs.anaconda.com/anaconda/install/linux/). Close and reopen the console after installation.
2. Create a new virtual environment and activate it.  
~~~~~shell
conda create -n gap-sdk python=3.8 numpy cython  
conda activate gap-sdk
~~~~~
3. Also install the config parser since it will be used later to buidling the gap-sdk. 
~~~~~shell 
sudo apt-get update -y  
sudo apt-get install -y python-configparser  
~~~~~
4. Furthermore later on the python package Pillow will used which need additional packages. 
~~~~~shell 
sudo apt install libjpeg8-dev zlib1g-dev  
~~~~~
Next the Olimex drivers will be installed for the communication through JTAG with the gap8 chip. First the drivers need to be installed on the Host OS as well as on the VM. For Windows OS you can use Zadig to install the driver.  

5. Download [Zadig](https://zadig.akeo.ie/), launch it and plug in the Olimex JTAG adapter. In the Application two interfaces (0 and 1) should show up. For both install the driver. The official description can be found [here](https://www.olimex.com/Products/ARM/JTAG/_resources/ARM-USB-OCD_and_OCD_H_manual.pdf#page=17).  
6. In the VM the following command install the dependencies needed.  
~~~~~shell
sudo apt-get install libftdi-dev libftdi1
~~~~~
7. Since root privileges are required to use the JTAG communication it is necessary to define a rule. You can use the following commands to do so.  
~~~~~shell
sudo touch /etc/udev/rules.d/olimex-arm-usb-ocd-h.rules  
sudo echo 'SUBSYSTEM=="usb", ACTION=="add", ATTRS{idProduct}=="002b", ATTRS{idVendor}=="15ba", MODE="664", GROUP="plugdev"' | sudo tee -a /etc/udev/rules.d/olimex-arm-usb-ocd-h.rules 
~~~~~ 
Please note that if you use another JTAG adapter than ARM-USB-OCD-H you need to adapt the Product and Vendor ID. Please refer to the official description [here](https://www.olimex.com/Products/ARM/JTAG/_resources/ARM-USB-OCD_and_OCD_H_manual.pdf#page=19) in case of any issues.  

8. Next the gap-sdk will be installed. To do so the following packages must be installed. ([Official description](https://github.com/GreenWaves-Technologies/gap_sdk))
~~~~~shell
sudo apt-get install -y build-essential git libftdi-dev libftdi1 doxygen python3-pip libsdl2-dev curl cmake libusb-1.0-0-dev scons gtkwave libsndfile1-dev rsync autoconf automake texinfo libtool pkg-config libsdl2-ttf-dev
~~~~~
9. Install the gap8 openOCD for On-Chip debugging.
~~~~~shell
git clone https://github.com/GreenWaves-Technologies/gap8_openocd.git
cd gap8_openocd
./bootstrap
./configure --program-prefix=gap8- --prefix=/usr --datarootdir=/usr/share/gap8-openocd
make -j
sudo make -j install

#Finally, copy openocd udev rules and reload udev rules
sudo cp /usr/share/gap8-openocd/openocd/contrib/60-openocd.rules /etc/udev/rules.d
sudo udevadm control --reload-rules && sudo udevadm trigger
~~~~~
10. Now, add your user to dialout group.
~~~~~shell
sudo usermod -a -G dialout <username>
# This will require a logout / login to take effect
~~~~~
Finally, logout of your session and log back in. Make sure that the USB device is attached to the VM. You can attach to the VM by going to Devices > USB and enable the Olimex adapter.

11. After that the toolchain can be installed ([Official guide](https://github.com/GreenWaves-Technologies/gap_riscv_toolchain_ubuntu_18)). Clone the GAP8 SDK and the GAP8/RISC-V toolchain.
~~~~~shell
cd ~
sudo apt-get install git-lfs
git lfs install
git lfs clone https://github.com/GreenWaves-Technologies/gap_riscv_toolchain_ubuntu_18.git
cd ~/gap_riscv_toolchain_ubuntu_18
sudo ./install.sh
~~~~~
 
12. Next the gap_sdk release-v3.8.1 can be cloned as follows.  
~~~~~shell
cd ~
git clone https://github.com/GreenWaves-Technologies/gap_sdk.git
cd ~/gap_sdk
git checkout tags/release-v3.8.1 -b gap_sdk-v3.8.1
~~~~~
13. For the installation of the gap_sdk you need to install the python packages into the virtual env.  
~~~~~shell
pip install -r tools/nntool/requirements.txt
pip install -r requirements.txt
~~~~~
14. Finally, we install the full tool suite of the sdk (including nntool and autotiler).  
~~~~~shell
git submodule update --init --recursive
source configs/ai_deck.sh
make sdk
make openocd
make gap_tools
~~~~~
To program the gap8 chip the correspondig config file of the ARM-USB-OCD-H must be loaded.
~~~~~shell
export GAPY_OPENOCD_CABLE=interface/ftdi/olimex-arm-usb-ocd-h.cfg
~~~~~
Now you can try the helloworld from the gap-sdk (make sure the JTAG adapter is attached to the VM).
~~~~~shell
cd examples/pmsis/helloworld
make clean all run platform=gvsoc   \\ use virtual platform
make clean all run platform=board	\\ program chip
~~~~~
For easier loading of the environment you can use the following alias and add it to your ~/.bashrc file.
~~~~~shell
alias GAP_SDK='mycwd=$(pwd) && cd ~/gap_sdk && conda activate gap-sdk && source configs/ai_deck.sh && export GAPY_OPENOCD_CABLE=interface/ftdi/olimex-arm-usb-ocd-h.cfg && cd $mycwd'
~~~~~






