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
pip install --upgrade wheel pip setuptools
pip install --upgrade --requirement requirements.txt
conda install -c conda-forge pygobject=3.40.1
conda install -c conda-forge gst-python=1.18.4
conda install -c conda-forge gst-plugins-good=1.18.4
conda install -c conda-forge gst-plugins-bad=1.18.4
conda install -c conda-forge gst-plugins-ugly=1.18.4
conda install -c conda-forge gst-libav=1.18.4
sudo yum install -y mesa-libGL
# For Ubuntu, sudo apt install libgl1-mesa-glx
mkdir -p runtime/pipeline
