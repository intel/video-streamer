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
import os
import cv2
import logging
import yaml
import numpy as np
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstBase', '1.0')
from gi.repository import Gst, GObject, GstBase
import time as timer
import vdms
# for https://github.com/tensorflow/tensorflow/issues/45994 ####
import sys
if not hasattr(sys, "argv"):
   setattr(sys, "argv", [])
if not sys.argv:
   sys.argv.append("(C++)")
import tensorflow as tf
from tensorflow.core.protobuf import rewriter_config_pb2
################################################################

Gst.init(None)

################################################################
# Gstreamer element for object detection
################################################################
class GstDetectionTF(GstBase.BaseTransform):

    # Metadata of the Gsteamer element
    __gstmetadata__ = ("gst_detection_tf",
                       "Transform",
                       "gst_detection_tf",
                       "Intel")    
   
    # Define the input of the element: video, and frames with RGB format
    _srctemplate = Gst.PadTemplate.new('src', Gst.PadDirection.SRC,
                                       Gst.PadPresence.ALWAYS,
                                       Gst.Caps.from_string("video/x-raw,format=RGB"))
    
    # Define the output of the element
    _sinktemplate = Gst.PadTemplate.new('sink', Gst.PadDirection.SINK,
                                        Gst.PadPresence.ALWAYS,
                                        Gst.Caps.from_string("video/x-raw,format=RGB"))
    
    __gsttemplates__ = (_srctemplate, _sinktemplate)
    
    # Define the args of the element
    __gproperties__ = {
        "conf": (
            str,
            "Path to configuration file",
            "The configuration file should be a YAML file",
            "config/settings.yaml", # default
            GObject.ParamFlags.READWRITE
        ),
    }

    '''
    Get execution time
    ''' 
    def exec(self, t_accumulator, func, *args):
        t_start = timer.time()
        rs = func(*args)
        t_exec = timer.time() - t_start

        if t_accumulator in self.metrics:
            self.metrics[t_accumulator] += t_exec
        else:
            self.metrics[t_accumulator] = t_exec
        
        func_name = f'{t_accumulator}/{func.__name__}'
        if func_name in self.metrics:
            self.metrics[func_name] += t_exec
        else:
            self.metrics[func_name] = t_exec

        return rs
        
    '''
    Initializing the element
    ''' 
    def __init__(self):
        self.starttime = timer.time()
        self.metrics = {
            'total' : 0.0,
            'tf'    : 0.0,
            'cv'    : 0.0,
            'np'    : 0.0,
            'py'    : 0.0,
            'vdms'  : 0.0
        }
        super().__init__()
        # Logger
        logging.basicConfig(level=logging.INFO)
        self.log = logging.getLogger('gst_detection_tf')

    '''
    Load coco label dictionary
    '''  
    def load_labels(self, filename) -> dict:
        if not os.path.isfile(filename):
            raise ValueError(f"Invalid filename {filename}")
        self.labl_dict = {}
        with open(filename, 'r') as f:
            for line in f:
                label_id, label_name = line.split(":")[:2]
                self.labl_dict[int(label_id)] = label_name.strip()

    '''
    Parse and set args
    '''      
    def do_set_property(self, prop, value):
        self.num_frames = 0
        self.queries = []
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 1

        # Set parameters
        if prop.name == 'conf':

            # Read configuration file
            config_file = value
            if not os.path.isfile(config_file):
                raise ValueError(f"Invalid configuration file: {config_file}")

            with open(config_file) as f:
                try: 
                    self.config = yaml.safe_load(f)
                except yaml.YAMLError as exc:
                    raise OSError(f'Parsing error: {config_file}')
        
            self.device = self.config['device']
            self.dtype = self.config['data_type']
            self.threshold = self.config['face_threshold']
            try:
                self.total_frames = self.config['total_frames']
            except KeyError:
                self.total_frames = -1

            # Intel CPU/GPU parameters
            if (self.device == 'CPU') or (self.device == 'ARCGPU'):
                os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
                os.environ['inter_op_parallelism'] = self.config['inter_op_parallelism']
                os.environ['intra_op_parallelism'] = self.config['intra_op_parallelism']
                os.environ['KMP_BLOCKTIME'] = '0'
                os.environ['TF_ENABLE_MKL_NATIVE_FORMAT'] = '1'
                os.environ['OMP_NUM_THREADS'] = self.config['intra_op_parallelism']
                os.environ['KMP_AFFINITY'] = 'granularity=fine,compact'
                os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1' if self.config['onednn'] else '0'
                if self.dtype == "FP32":
                    self.model_path = self.config['ssd_resnet34_fp32_1200x1200_model']
                elif self.dtype == "AMPBF16":
                    self.model_path = self.config['ssd_resnet34_fp32_1200x1200_model']
                    if self.config['amx']:
                        os.environ['DNNL_MAX_CPU_ISA'] = 'AVX512_CORE_AMX'
                elif self.dtype == "INT8":
                    self.model_path = self.config['ssd_resnet34_int8_1200x1200_model']
                    if self.config['amx']:
                        os.environ['DNNL_MAX_CPU_ISA'] = 'AVX512_CORE_AMX'
                else:
                    raise ValueError(f"Unsupported data type:{self.dtype}")

            # NV GPU parameters
            elif self.device == "GPU":
                os.environ['CUDA_VISIBLE_DEVICES'] = '0'
                if self.dtype == "FP32":
                    self.model_path = self.config['ssd_resnet34_fp32_gpu_model']
                elif self.dtype == "FP16":
                    self.model_path = self.config['ssd_resnet34_fp16_gpu_model']
                else:
                    raise ValueError(f"Unsupported data type:{self.dtype}")
        else:
            raise ValueError(f"Unsupported device: {self.device}")

        self.load_labels(self.config['label_file'])

        # Load detection model
        self.exec('tf', self.load_model)

        self.log.info(f'Parameters: {self.config}')
        self.log.debug(f'OS Env: {os.environ}')

    '''
    Load pretrained, state-of-the-art models
    '''
    def load_model(self):
        self.log.info(f'Loading model: {self.model_path}')

        with tf.io.gfile.GFile(self.model_path, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())

        # Import graph_def
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name='')
            self.detection_graph = graph

        # Prepare TF config
        if self.device == 'CPU' or self.device == 'ARCGPU':
            config = tf.compat.v1.ConfigProto(
                intra_op_parallelism_threads = int(self.config['intra_op_parallelism']),
                inter_op_parallelism_threads = int(self.config['inter_op_parallelism']), 
                allow_soft_placement = True)
            if self.dtype == "AMPBF16":
                config.graph_options.rewrite_options.auto_mixed_precision_mkl = rewriter_config_pb2.RewriterConfig.ON
            elif self.dtype == "FP32":
                config.graph_options.rewrite_options.remapping = rewriter_config_pb2.RewriterConfig.AGGRESSIVE
            elif self.dtype == "FP16":
                config.graph_options.rewrite_options.auto_mixed_precision_mkl = rewriter_config_pb2.RewriterConfig.ON
        elif self.device == "GPU":
            config = tf.compat.v1.ConfigProto()
            config.allow_soft_placement = True
            config.gpu_options.allow_growth = True
            if self.dtype == "FP16":
                config.graph_options.rewrite_options.auto_mixed_precision = rewriter_config_pb2.RewriterConfig.ON

        # Create TF session
        self.sess = tf.compat.v1.Session(graph=self.detection_graph, config=config)

        # Define graph input/output   
        self.input_tensor = self.detection_graph.get_tensor_by_name("image:0")
        output_layers = ['detection_bboxes', 'detection_scores', 'detection_classes']
        self.output_tensors = [self.detection_graph.get_tensor_by_name(x + ":0") for x in output_layers]
        
    
    '''
    Get video height and width from incaps
    '''
    def do_set_caps(self, incaps, outcaps):
        struct = incaps.get_structure(0)
        self.width = struct.get_int("width").value
        self.height = struct.get_int("height").value
        return True

    '''
    Convert Gst.Buffer to np.ndarray
    '''
    def gst_buf_to_ndarray(self, buf):
        with buf.map(Gst.MapFlags.READ | Gst.MapFlags.WRITE) as info:
            return np.ndarray(shape = (self.height, self.width, 3), dtype = np.uint8, buffer = info.data)

    '''
    Normalize image pixels (numpy)
    '''
    def normalize_image_np(self, image):
        image = image.astype('float32') / 255.
        std = np.array([0.229, 0.224, 0.225], dtype = np.float32)
        mean = np.array([0.485, 0.456, 0.406], dtype = np.float32)
        image = (image - mean) / std
        return image
 
    '''
    Format image ndarray to fit inference model 
    '''
    def format_image(self, image):
        if self.device == "GPU":
            if self.config['preproc_fw'] == 'tf':
                # gpu models wants channels first
                #image = np.moveaxis(image, -1, 0)
                image = tf.transpose(image, [2, 0, 1])
                image = tf.expand_dims(image, 0)
            else:
                image = np.moveaxis(image, -1, 0)
                image = np.expand_dims(image, axis=0)
        elif self.config['preproc_fw'] == 'tf': 
            image = tf.expand_dims(image, 0).numpy()
        else:
            image = np.expand_dims(image, axis=0)
        return image
    
    '''
    Inference
    '''
    def inference(self, image):
        return self.sess.run(self.output_tensors, {self.input_tensor: image})
        
    '''
    Inference result post preprocessing
    '''   
    def process_inference_result(self, width, height, confidences, boxes, labels, prob_threshold):
        # NMS done in HR model
        mask = confidences > prob_threshold
        pbox = boxes[mask]
        pbox[:, 0] *= height
        pbox[:, 1] *= width
        pbox[:, 2] *= height
        pbox[:, 3] *= width

        if np.count_nonzero(mask) == 0:
            return np.array([]), np.array([]), np.array([])
        
        return pbox.astype(np.int32), labels[mask], confidences[mask]

    def build_db_data(self, box, label, frame_id):
        if self.config['database'] != 'VDMS':
            return
        data = {}
        data['AddBoundingBox'] = {}
        data['AddBoundingBox']['rectangle'] = {}
        data['AddBoundingBox']["_ref"] = frame_id
        data['AddBoundingBox']["rectangle"]['x'] = int(box[0])
        data['AddBoundingBox']["rectangle"]['y'] = int(box[1])
        data['AddBoundingBox']["rectangle"]['w'] = int(box[2])
        data['AddBoundingBox']["rectangle"]['h'] = int(box[3])
        data['properties'] = {
            'label': int(label)
        }
        self.queries.append(data)

    def int_to_rgb(self, num):
        num = int(num / 80 * 255)
        r = num & 255
        g = (num >> 8) & 255
        b = (num >> 16) & 255
        return (r, g, b)

    def bound_box(self, image, box, label, label_id):
        if self.config['bounding_box']:
            color = self.int_to_rgb(label_id)
            image = cv2.rectangle(image, (box[1], box[0]), 
                    (box[3], box[2]), color, 4)
            image = cv2.putText(image, label, (box[1], box[0]), self.font, 
                    self.font_scale, color, 2, cv2.LINE_AA)

    def save2db(self):
        if self.config['database'] != "VDMS":
            return
        self.vdms_db = vdms.vdms()
        self.vdms_db.connect()
        self.vdms_db.query(self.queries)

    def preprocess(self, buffer):
        # data conversion: gst buffer -> numpy ndarray
        raw_img = self.exec('np', self.gst_buf_to_ndarray, buffer)
        image = raw_img.copy()

        if self.config['preproc_fw'] == 'cv2':
            # normalization
            image = self.exec('cv', cv2.normalize, image, None, 0, 
                    1, cv2.NORM_MINMAX, cv2.CV_32F)
            # image resizing
            image = self.exec('cv', cv2.resize, image, (1200, 1200))
            # data conversion: expand dimenstions
            image = self.exec('np', self.format_image, image)

        elif self.config['preproc_fw'] == 'tf':
            # normalization
            image = self.exec('tf', tf.image.per_image_standardization, image) 
            # image resizing
            image = self.exec('tf', tf.image.resize, image, [1200, 1200])
            # data conversion: expand dimenstions
            image = self.exec('tf', self.format_image, image)
            # if self.device == 'GPU':
            #     image = image.numpy()

        elif self.config['preproc_fw'] == 'np':  
            # normalization
            image = self.exec('np', self.normalize_image_np, image) 
            # image resizing
            image = self.exec('cv', cv2.resize, image, (1200, 1200))
            # data conversion: expand dimenstions
            image = self.exec('np', self.format_image, image)

        else:
            raise ValueError(f'Unsupported preprcessing framework/lib: {self.config["preproc_fw"]}')

        if self.device == "GPU" and self.config['preproc_fw'] == 'tf':
            return tf.Variable(image), raw_img
        else:
            return image, raw_img

    def postprocess(self, image, rs, raw_img):
        # low-confidence boxes filtering
        boxes, labels, probs = self.exec('np', self.process_inference_result, 
            self.width, self.height, rs[1][0], rs[0][0], rs[2][0], self.threshold)

        # process single box
        for i in range(boxes.shape[0]):
            box = boxes[i]
            # get text of an integer coco label
            label_id = int(labels[i])
            label = self.labl_dict[label_id]
            # build vdms data structure
            self.exec('py', self.build_db_data, boxes[i], labels[i], (i + 1))
            # bound box TODO: tf.image.draw_bounding_boxes
            self.exec('cv', self.bound_box, raw_img, box, label, label_id)

    '''
    Process single frame
    '''
    def processSingleFrame(self, buffer):     
        # quit pipeline if you don't want to process the complete video
        if self.total_frames != -1 and self.num_frames >= self.total_frames:
            self.quit_gracefully()

        self.num_frames += 1

        # preprocessing (data conversion, normalization, image resizing)
        image, raw_img = self.exec('e2e', self.preprocess, buffer)

        # DL (object detection)
        rs = self.exec('tf', self.inference, image)
        
        # post-processing (low-confidence boxes filtering, draw bounding-boxes, vdms writes)
        self.exec('e2e', self.postprocess, image, rs, raw_img)

    def do_last_frame(self):
        # vdms writes
        self.exec('vdms', self.save2db)

        # metrics calculation
        self.metrics['total'] = timer.time() - self.starttime
        self.metrics['frames'] = self.num_frames
        if 'e2e/postprocess' in self.metrics:
            self.metrics['e2e/postprocess'] += self.metrics['vdms/save2db']
        else:
            self.metrics['e2e/postprocess'] = self.metrics['vdms/save2db']
        del self.metrics['e2e']
        self.log.info(f"Pipeline completes: {self.metrics}")

    def quit_gracefully(self):
        self.do_last_frame()
        exit()

    '''
    Main
    '''
    def do_transform_ip(self, buffer: Gst.Buffer) -> Gst.FlowReturn:
        try:                             
            self.processSingleFrame(buffer)
        except Exception as err:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            line = exc_tb.tb_lineno
            self.log.error(f"<{fname}><{line}>: {err}")
            import traceback
            traceback.print_exc()
            return Gst.FlowReturn.ERROR
        return Gst.FlowReturn.OK
    
    def do_stop(self):
        self.quit_gracefully()

# Required for registering plugin dynamically
GObject.type_register(GstDetectionTF)
__gstelementfactory__ = ("gst_detection_tf", Gst.Rank.NONE, GstDetectionTF)
