# Video Streamer
The video streamer pipeline is designed to mimic real-time video analytics. Real-time data is provided to an inference endpoint that executes single-shot object detection. The metadata created during inference is then uploaded to a database for curation.

## Installation

The software stack works on Python3.

For installing required software, please perform the following commands:

Go to the directory where you cloned the repo:
```
conda create -n vdms-test python=3.8
conda activate vdms-test
./install.sh
```

By default, this will install intel-tensorflow-avx512.  If it is necessary to run the workflow using a specific tensorflow then update this line in `requirements.txt`.


## Preparation

Before you run the workload, you need to prepare data and models.

Go to the directory where you cloned the repo:
```
mkdir dataset
cd dataset
#download a sample video, or copy your own video to dataset folder 
wget https://github.com/intel-iot-devkit/sample-videos/raw/master/classroom.mp4
cd ..
mkdir models
cd models
#download pretrained ssd-resnet34 fp32 and int8 models
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/ssd_resnet34_fp32_1200x1200_pretrained_model.pb
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/ssd_resnet34_int8_1200x1200_pretrained_model.pb
```
The metadata created during inference will be uploaded to a database.The easies way to setup a VDMS instance for database testing is using docker with the following command:
```
docker run --net=host -d vuiseng9/intellabs-vdms:demo-191220
```

By defualt, VDMS will attempt to use all threads available which can impact performance.  For testing here it seems that four cores is more than enough to handle data ingestion, so it is recommended that the database is pinned to the last 4 cores of the second socket on a multi-socket system. I.e. for a dual socket Xeon 8280:
```
numactl --physcpubind=52-55 --membind=1 docker run --net=host -d vuiseng9/intellabs-vdms:demo-191220
```

## Configuration

Modify the parameter `gst_plugin_dir` and `video_path` in `config/pipeline-settings` to fit your Gstreamer plugin directory and input video path.

Customize `config/settings.yaml` to choose FP32, AMPBF16 or INT8 for inference.

CPU Optimization settings are found in two files:

`config/settings.yaml`
1. Tensorflow thread settings

`config/pipeline-settings`
1. cores_per_pipeline

## Run the workload

`run.sh` is configured to accept a single input parameter which defines how many separate instances of the gstreamer ipelines to run. The instances are pinned to $OMP_NUM_THREADS real processors and pinned to local memory.  I.e. when running four pipelines with OMP_NUM_THREADS=4
|*Pipeline*|*Cores*|*Memory*|
| ---- | ---- | ---- |
|1| 0-3| Local |
|2| 4-7| Local |
|3| 8-11| Local |
|4|12-15| Local |

It is very important that the pipelines don't overlap numa domains or any other hardware non-uniformity.  These values must be updated for each core architecture to get optimum performance.

For launching the workload using a single instance, use the following command:
`./run.sh 1`


For launching 14 instances with 4 cores per instance on a dual socket Xeon 8280, just run `./run.sh 14`

`benchmark.sh` can be used to easily collect performance characterization data on a target hardware platform.

This script runs the `run.sh` script and obeys the hardware config defined in the configuration. 
