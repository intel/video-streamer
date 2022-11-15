#!/bin/bash
# Copyright (C) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.
#
cat ./logo

source config/pipeline-settings

export GST_PLUGIN_PATH=$gst_plugin_dir:$PWD/gst-plugin
echo $GST_PLUGIN_PATH

export GST_DEBUG="GST_TRACER:7"
export GST_TRACERS="proctime"
export GST_DEBUG_FILE=/tmp/gst-detect
if [ ${gst_graph} == "1" ]; then
  export GST_DEBUG_DUMP_DOT_DIR=runtime/pipeline/
fi
rm -rf runtime/pipeline/*

python3 gst-plugin/python/gst-detection-${framework}.py

sink='fakesink'
if [ ${show_fps} == "1" ]; then
   #display the current and average framerate every 1 second
   sink='fpsdisplaysink text-overlay=false silent=false video-sink=fakesink sync=false fps-update-interval=1000 --verbose'
fi
if [ ${demo_mode} == "1" ]; then
   #sink=" videoconvert ! x264enc ! videoscale ! video/x-raw,format=RGB,width=${demo_mode_display_width},height=${demo_mode_dispaly_height} ! autovideosink sync=false" 
   sink=" autovideoconvert ! videoscale ! video/x-raw,width=${demo_mode_display_width},height=${demo_mode_dispaly_height} ! autovideosink sync=false" 
fi

export OMP_NUM_THREADS=${cores_per_pipeline}

j=$1
for ((i=0;i<$j;++i))
do
  socket=$((i%2))
  start=$((i*$OMP_NUM_THREADS))
  tend=$((i+1))
  tend=$((tend*$OMP_NUM_THREADS-1))
  set -x
  numactl --physcpubind=$start-$tend --localalloc gst-launch-1.0 filesrc location=${video_path} ! decodebin ! videoconvert ! video/x-raw,format=RGB ! videoconvert ! queue ! gst_detection_${framework} conf=${dl_config} ! $sink &
  set +x
done

wait
echo "Finished all pipelines"
