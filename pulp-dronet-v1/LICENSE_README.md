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
A copy of the Apache 2.0 license is included in `pulp-dronet\LICENSE.apache.md` and in `pulp-dronet-v1/dataset/Himax_Dataset/LICENSE.apache.md`.

All the files in `pulp-dronet-v1/dataset/Udacity_Dataset` and  `pulp-dronet-v1/dataset/Zurich_Bicycle_Dataset` are released with the MIT License included in the same folder as `LICENSE.mit.md`.
