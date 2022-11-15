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
export VIDEO_FILE=${VIDEO_FILE-dataset/classroom.mp4}
sed -i "s%video_path=.*%video_path=${VIDEO_FILE}%" config/pipeline-settings
core_count=`lscpu -p |  awk -F, '{print $2}' | tail -1`
pipes=$((($core_count + 1)/4))
#run mulit-instance by using all cores, modify below PIPES value to run the number of instances you want
export PIPES=$pipes

#set pipeline to FP32
export DTYPE="FP32"
sed -i "s/\"INT8\"/\"$DTYPE\"/" config/settings.yaml
sed -i "s/\"BF16\"/\"$DTYPE\"/" config/settings.yaml
sed -i "s/\"FP32\"/\"$DTYPE\"/" config/settings.yaml
sed -i "s/\"AMPBF16\"/\"$DTYPE\"/" config/settings.yaml

#preheat the pipeline
for i in {1..1}; do ./run.sh $PIPES; done

#get the dnnl verbose log
#DNNL_VERBOSE=1 ./run.sh $PIPES > ../resnet_${PIPES}_${DTYPE}_dnnl.csv

#get the timings
./run.sh $PIPES 2>&1 | tee ../resnet_${PIPES}_${DTYPE}_timings.txt

#enable BF16
export DTYPE='AMPBF16'
sed -i "s/\"INT8\"/\"$DTYPE\"/" config/settings.yaml
sed -i "s/\"BF16\"/\"$DTYPE\"/" config/settings.yaml
sed -i "s/\"FP32\"/\"$DTYPE\"/" config/settings.yaml
sed -i "s/\"AMPBF16\"/\"$DTYPE\"/" config/settings.yaml

#preheat the pipeline
for i in {1..1}; do ./run.sh $PIPES; done

#get the dnnl verbose log
#DNNL_VERBOSE=1 ./run.sh $PIPES > ../resnet_${PIPES}_${DTYPE}_dnnl_amx.csv

#get the timings
./run.sh $PIPES 2>&1 | tee ../resnet_${PIPES}_${DTYPE}_timings_amx.txt


#set pipeline to INT8
export DTYPE='INT8'
sed -i "s/\"INT8\"/\"$DTYPE\"/" config/settings.yaml
sed -i "s/\"BF16\"/\"$DTYPE\"/" config/settings.yaml
sed -i "s/\"FP32\"/\"$DTYPE\"/" config/settings.yaml
sed -i "s/\"AMPBF16\"/\"$DTYPE\"/" config/settings.yaml

#preheat the pipeline
for i in {1..1}; do ./run.sh $PIPES; done

#get the dnnl verbose log
#DNNL_VERBOSE=1 ./run.sh $PIPES > ../resnet_${PIPES}_${DTYPE}_dnnl_amx.csv

#get the timings
./run.sh $PIPES 2>&1 | tee ../resnet_${PIPES}_${DTYPE}_timings_amx.txt

