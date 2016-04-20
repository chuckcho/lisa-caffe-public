#!/usr/bin/env python

'''
classify_video.py will classify a video using LRCN RGB model, and save results
as if "${APIServer}/utils/run.py" would have processed
'''

import numpy as np
import glob
caffe_root = '../../'
import sys
sys.path.insert(0,caffe_root + 'python')
import caffe
caffe.set_mode_gpu()
caffe.set_device(0)
import pickle
import os
import time
import json
from subprocess import Popen, PIPE

# global var
verbose = False
num_categories = 1170
detection_threshold = 0.02

#Initialize transformers
def initialize_transformer(image_mean, is_flow):
  shape = (10*16, 3, 227, 227)
  transformer = caffe.io.Transformer({'data': shape})
  channel_mean = np.zeros((3,227,227))
  for channel_index, mean_val in enumerate(image_mean):
    channel_mean[channel_index, ...] = mean_val
  transformer.set_mean('data', channel_mean)
  transformer.set_raw_scale('data', 255)
  transformer.set_channel_swap('data', (2, 1, 0))
  transformer.set_transpose('data', (2, 0, 1))
  transformer.set_is_flow('data', is_flow)
  return transformer

#classify video with LRCN model
def LRCN_classify_video(frames, net, transformer, is_flow):
  if verbose:
    print "[Info] len(frames)={}".format(len(frames))
  clip_length = 16
  offset = 8
  input_images = []
  for im in frames:
    input_im = caffe.io.load_image(im)
    if verbose:
      print "[Info] input_im.shape={}".format(input_im.shape)
    if (input_im.shape[0] < 240 or input_im.shape[1] < 320):
      if verbose:
        print "[Info] enlarge image to fit (240,320)"
      input_im = caffe.io.resize_image(input_im, (240,320))
    input_images.append(input_im)
  vid_length = len(input_images)
  input_data = []
  for i in range(0,vid_length,offset):
    if (i + clip_length) < vid_length:
      if verbose:
        print "[Info] input_data += input_images[{}:{}]".format(i,i+clip_length)
      input_data.extend(input_images[i:i+clip_length])
    else:  #video may not be divisible by clip_length
      input_data.extend(input_images[-clip_length:])
  output_predictions = np.zeros((len(input_data),num_categories))
  if verbose:
    print "[Info] output_predictions.shape={}".format(output_predictions.shape)
    print "[Info] run forward with range(0,len(input_data)={}, clip_length={})".format(
            len(input_data), clip_length)
  for i in range(0,len(input_data),clip_length):
    clip_input = input_data[i:i+clip_length]
    clip_input = caffe.io.oversample(clip_input,[227,227])
    clip_clip_markers = np.ones((clip_input.shape[0],1,1,1))
    clip_clip_markers[0:10,:,:,:] = 0
#    if is_flow:  #need to negate the values when mirroring
#      clip_input[5:,:,:,0] = 1 - clip_input[5:,:,:,0]
    caffe_in = np.zeros(np.array(clip_input.shape)[[0,3,1,2]], dtype=np.float32)
    for ix, inputs in enumerate(clip_input):
      caffe_in[ix] = transformer.preprocess('data',inputs)
      if verbose:
        print "[Info] i={}, ix={}, caffe_in[ix].shape={}".format(i, ix, caffe_in[ix].shape)
    if verbose:
      print "[Info] caffe_in.shape={}".format(caffe_in.shape)
    out = net.forward_all(data=caffe_in, clip_markers=np.array(clip_clip_markers))
    output_predictions[i:i+clip_length] = np.mean(out['probs'],1)
  return np.mean(output_predictions,0).argmax(), output_predictions

def main():

  if len(sys.argv) < 3:
    print "[Fatal] usage: {} <video file/dir> <result output file>".format(
            sys.argv[0]
            )
    sys.exit(-1)
  video = sys.argv[1]
  result_file = sys.argv[2]

  if os.path.isfile(result_file) and os.path.getsize(result_file):
    print "[Info] Already processed. Skipping..."
    sys.exit(0)

  # default video length: may be overriden by the actual number if a video file
  # is available
  video_length = 0.0

  # check given "video" is a directory that contains extracted frames, or
  # a video file that hasn't been extracted yet
  if os.path.isdir(video):
    if not glob.glob('{}/*.jpg'.format(video)):
      print("[Fatal] video={} is a directory but does not have jpg files "
            "within. Exitting...".format(video))
      sys.exit(-2)
  elif os.path.isfile(video):
    print "[Info] video file={} is given. Extracting them...".format(video)
    file_no_ext, ext = os.path.splitext(video)
    # already a directory exists?
    if os.path.isdir(file_no_ext) and glob.glob('{}/*.jpg'.format(file_no_ext)):
      print "[Warning] video dir={} already contains jpg files.".format(
              file_no_ext)
      video = file_no_ext
    else:
      # extract video into frames
      video_id = os.path.basename(file_no_ext)
      if not os.path.isdir(file_no_ext):
        os.makedirs(file_no_ext)
      cmd = [ 'ffmpeg',
              '-i',
              video,
              #'-r',
              #fps,
              '{}/{}.%6d.jpg'.format(file_no_ext, video_id)
            ]
      process = Popen(cmd, stdout=PIPE)
      process.communicate()
      exit_code = process.wait()
      if 0 == exit_code:
        print "[Info] frame extraction has been successful!"
      else:
        print "[Warning] frame extraction has been **unsuccessful**!"
        sys.exit(-3)

      # get video length (extra info)
      cmd = [ 'ffprobe',
              '-v',
              'error',
              '-select_streams',
              'v:0',
              '-show_entries',
              'stream=duration',
              '-of',
              'default=noprint_wrappers=1:nokey=1',
              video
              ]
      process = Popen(cmd, stdout=PIPE)
      out, _ = process.communicate()
      exit_code = process.wait()
      video_length = float(out)

      # set video as directory
      video = file_no_ext
  else:
    print "[Fatal] video={} can not be found. Exitting...".format(video)
    sys.exit(-3)

  # mean pixel
  mean_RGB = np.zeros((3,1,1))
  mean_RGB[0,:,:] = 103.939
  mean_RGB[1,:,:] = 116.779
  mean_RGB[2,:,:] = 128.68

  # initialize net / model & weight
  lstm_model = 'deploy_lstm.prototxt'
  RGB_lstm = 'snapshots_dextro_benchmark_lstm_iter_30000.caffemodel'
  transformer_RGB = initialize_transformer(mean_RGB, False)
  RGB_lstm_net =  caffe.Net(lstm_model, RGB_lstm, caffe.TEST)

  # Load label mapping
  category_map_file = 'content_type_mapping.json'
  if os.path.isfile(category_map_file):
    #if pickle
    #category_hash = pickle.load(open(category_map_file,'rb'))
    category_hash = json.load(open(category_map_file))
  else:
    print "[Fatal] category mapping file does not exist."

  # Extract list of frames in video
  RGB_frames = sorted(glob.glob('{}/*.jpg'.format(video)))
  video_id = os.path.basename(video)

  if verbose:
    print "[Debug] RGB_frames={}".format(RGB_frames)

  if not RGB_frames:
    print "[Fatal] no RGB images found"
    sys.exit(-1)

  print "-" * 19
  print "[Info] Processing video={}".format(video_id)
  start_time = time.time()
  class_RGB_LRCN, predictions_RGB_LRCN = \
           LRCN_classify_video(
                   RGB_frames,
                   RGB_lstm_net,
                   transformer_RGB,
                   is_flow=False)
  del RGB_lstm_net
  RGB_lstm_processing_time = (time.time() - start_time)
  avg_pred = np.mean(predictions_RGB_LRCN,0).flatten()
  detected_categories = set(np.where(avg_pred >= detection_threshold)[0])

  outf = open(result_file, 'w')
  # output looks something like: Received: { json_payload }
  # where json_payload:
  '''
{
    "detections": [
        {
            "id": 10656,
            "instance_occurrences": [
                [
                    0.0,
                    190.8
                ]
            ],
            "name": "projector_screen",
            "salience": 0.9958246346555324
        },
        {
            "id": 10658,
            "instance_occurrences": [
                [
                    0.0,
                    190.8
                ]
            ],
            "name": "computer_game",
            "salience": 0.9958246346555324
        },
        {
            "id": 10698,
            "instance_occurrences": [
                [
                    0.0,
                    190.8
                ]
            ],
            "name": "interview",
            "salience": 0.9958246346555324
        },
        {
            "id": 10703,
            "instance_occurrences": [
                [
                    0.0,
                    190.8
                ]
            ],
            "name": "press_conference",
            "salience": 0.9958246346555324
        },
        {
            "id": 11039,
            "instance_occurrences": [
                [
                    0.0,
                    190.8
                ]
            ],
            "name": "cartoon",
            "salience": 0.9958246346555324
        }
    ],
    "length": 191.6,
    "original_url": "/media/6TB/Videos/dextro-benchmark-2016-03-30/1743.mp4",
    "request_id": "cmdline_a9ceccd3-8b61-44eb-b9dc-64bcbd1141e0"
}
'''

  detections = []
  for det in detected_categories:
    this_det = dict()
    this_det["id"] = 10000  # random number
    this_det["instance_occurrences"] = [[0.0, video_length]]
    this_det["name"] = category_hash[det]
    this_det["salience"] = 0.99  # random number
    detections.append(this_det)

  payload = dict()
  payload["detections"] = detections
  payload["length"] = video_length
  payload["original_url"] = "http://dextro.co/duh"
  payload["request_id"] = "hello_world"

  outf.write('Received: {}'.format(json.dumps(payload)))
  outf.close()

if __name__ == '__main__':
    main()
