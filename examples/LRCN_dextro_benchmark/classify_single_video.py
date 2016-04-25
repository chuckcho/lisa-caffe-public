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
from xml.dom import minidom

# global var
verbose = True
# benchmark-v2
#num_categories = 1170
# benchmark-2016-03-30
num_categories = 1473
#detection_threshold = 0.02
detection_threshold = 0.092

def revmapper(ref_dict, search_value):
  ''' For a dict, find a key given a value. Beware: if there are multiple keys
  associated with the value, the first key will be returned. '''
  for k,v in ref_dict.iteritems():
    if v == search_value:
      return k
  return None

def merge_timelines(detections):
  ''' input: detections (list of lists)
      output: merged detections (list of dicts) '''

  merged_detections = []

  detected_ct_names = set()
  for det_list in detections:
    for det in det_list:
      detected_ct_names.add(det["name"])

  print "[Info] detected_ct_names={}".format(detected_ct_names)

  for ct_name in detected_ct_names:
    this_ct_name_not_merged_yet = True
    for det_list in detections:
      for det in det_list:
        print "[Debug] ct_name={}, det[\"name\"]={}".format(
                ct_name,
                det["name"]
                )
        if ct_name == det["name"]:
          if this_ct_name_not_merged_yet:
            merged_detection_per_ct = {
                                  "id": det["id"],
                                  "instance_occurrences": det["instance_occurrences"],
                                  "name": ct_name,
                                  "salience": det["salience"]
                                }
            print "[Debug] first time, merged_detection_per_ct={}".format(
                    merged_detection_per_ct
                    )
            this_ct_name_not_merged_yet = False
          else:
            merged_detection_per_ct["instance_occurrences"].append(
                    det["instance_occurrences"])
            merged_detection_per_ct["salience"] += det["salience"]
            print "[Debug] merged_detection_per_ct={}".format(
                    merged_detection_per_ct
                    )
    merged_detections.append(merged_detection_per_ct)

  return merged_detections

def get_video_length(video_file):
  ''' get video length using ffmpeg '''
  cmd = [ 'ffprobe',
          '-v',
          'error',
          '-select_streams',
          'v:0',
          '-show_entries',
          'stream=duration',
          '-of',
          'default=noprint_wrappers=1:nokey=1',
          video_file
          ]
  process = Popen(cmd, stdout=PIPE)
  out, _ = process.communicate()
  exit_code = process.wait()
  if exit_code == 0:
    video_length = float(out)
  else:
    video_length = 0.0

  return video_length

def initialize_transformer():
  ''' Initialize transformers '''

  # mean pixel
  mean_RGB = np.zeros((3,1,1))
  mean_RGB[0,:,:] = 103.939
  mean_RGB[1,:,:] = 116.779
  mean_RGB[2,:,:] = 128.68

  shape = (10*16, 3, 227, 227)
  transformer = caffe.io.Transformer({'data': shape})
  channel_mean = np.zeros((3,227,227))
  for channel_index, mean_val in enumerate(mean_RGB):
    channel_mean[channel_index, ...] = mean_val
  transformer.set_mean('data', channel_mean)
  transformer.set_raw_scale('data', 255)
  transformer.set_channel_swap('data', (2, 1, 0))
  transformer.set_transpose('data', (2, 0, 1))
  transformer.set_is_flow('data', False)
  return transformer

def LRCN_classify_video(
        frames,
        net,
        transformer
        ):
  ''' Classify video with LRCN model '''
  if verbose:
    print "[Info] len(frames)={}".format(len(frames))
  clip_length = 16
  offset = 8
  input_images = []

  for im in frames:
    input_im = caffe.io.load_image(im)
    #if verbose:
    #  print "[Info] input_im.shape={}".format(input_im.shape)
    if (input_im.shape[0] < 240 or input_im.shape[1] < 320):
      #if verbose:
      #  print "[Info] enlarge image to fit (240,320)"
      input_im = caffe.io.resize_image(input_im, (240,320))
    input_images.append(input_im)
  vid_length = len(input_images)
  input_data = []

  for i in range(0,vid_length,offset):
    if (i + clip_length) < vid_length:
      #if verbose:
      #  print "[Info] input_data += input_images[{}:{}]".format(i,i+clip_length)
      input_data.extend(input_images[i:i+clip_length])
    else:  #video may not be divisible by clip_length
      input_data.extend(input_images[-clip_length:])
  output_predictions = np.zeros((len(input_data),num_categories))

  if verbose:
    pass
    #print "[Info] output_predictions.shape={}".format(output_predictions.shape)
    #print "[Info] run forward with range(0,len(input_data)={}, clip_length={})".format(
    #        len(input_data), clip_length)
  for i in range(0,len(input_data),clip_length):
    clip_input = input_data[i:i+clip_length]
    clip_input = caffe.io.oversample(clip_input,[227,227])
    clip_clip_markers = np.ones((clip_input.shape[0],1,1,1))
    clip_clip_markers[0:10,:,:,:] = 0
    caffe_in = np.zeros(np.array(clip_input.shape)[[0,3,1,2]], dtype=np.float32)
    for ix, inputs in enumerate(clip_input):
      caffe_in[ix] = transformer.preprocess('data',inputs)
    #  if verbose:
    #    print "[Info] i={}, ix={}, caffe_in[ix].shape={}".format(i, ix, caffe_in[ix].shape)
    #if verbose:
    #  print "[Info] caffe_in.shape={}".format(caffe_in.shape)
    out = net.forward_all(data=caffe_in, clip_markers=np.array(clip_clip_markers))
    output_predictions[i:i+clip_length] = np.mean(out['probs'],1)

  return np.mean(output_predictions,0).argmax(), output_predictions

def main():

  # first, find shot boundaries and feed each shot into LSTM
  shot_based = True
  #shot_based = False
  min_shot_length = 1.0  # minimum shot length in sec

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

  # already, extracted frames from a video
  if os.path.isdir(video):
    if not glob.glob('{}/*.jpg'.format(video)):
      print("[Fatal] video={} is a directory but does not have jpg files "
            "within. Exitting...".format(video))
      sys.exit(-2)
    potential_video_files = sorted(
            glob.glob(video.rstrip('/') + '.mp4'),
            key=os.path.getsize
            )
    if len(potential_video_files):
        video_file = potential_video_files[-1]
        video_length = get_video_length(video_file)
    else:
        video_length = 0.0

  # a video file is provided. check if frames were already extracted.
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
      video_length = get_video_length(video)

      # set video as directory
      video = file_no_ext

  # not a directory, nor a video file
  else:
    print "[Fatal] video={} can not be found. Exitting...".format(video)
    sys.exit(-3)

  # video length detected?
  if video_length and verbose:
    print "[Info] video_length={}".format(video_length)

  # initialize net / model & weight
  lstm_model = 'deploy_lstm.prototxt'
  RGB_lstm = 'snapshots_dextro_benchmark_2016_03_30_lstm_iter_30000.caffemodel'
  transformer_RGB = initialize_transformer()
  RGB_lstm_net =  caffe.Net(lstm_model, RGB_lstm, caffe.TEST)

  # Load label mapping
  category_map_file = 'content_type_mapping_2016_03_30.json'
  if os.path.isfile(category_map_file):
    #if pickle
    #category_hash = pickle.load(open(category_map_file,'rb'))
    category_hash = json.load(open(category_map_file))
  else:
    print "[Fatal] category mapping file does not exist."

  # Load content_type ID to name mapping
  id_to_name_map_file = '/srv/models/networks/id_name_map.json'
  with open(id_to_name_map_file, 'r') as fid:
    id_to_name_map = json.load(fid)

  # Extract list of frames in video
  shots = []
  video_id = os.path.basename(video)
  if shot_based:
    # read shot boundary information
    shot_xml = os.path.join(
            video + '_shot',  video_id, 'result.xml'
            )
    if not os.path.isfile(shot_xml):
      print("[Fatal] Can not read shot XML file={}. Skipping this video..."
            "".format(shot_xml))
      sys.exit(-11)
    xmldoc = minidom.parse(shot_xml)
    video_length = float(
            xmldoc.getElementsByTagName('duration')[0].childNodes[0].data
            ) / 1000
    print "[Info] video_length={} overridden by parsing shot_detection output".format(video_length)
    shotlist = xmldoc.getElementsByTagName('shot')

    # overwrite video length (msec to sec)
    for shot_count, shotitem in enumerate(shotlist):
      begin_sec = float(shotitem.attributes['msbegin'].value) / 1000
      duration_sec = float(shotitem.attributes['msduration'].value) / 1000

      if duration_sec < min_shot_length:
        if verbose:
          print "[Info] duration={} was too short. Skipping...".format(
                  duration_sec
                  )
        continue

      start_frame = int(shotitem.attributes['fbegin'].value)
      num_frames = int(shotitem.attributes['fduration'].value)
      end_frame = start_frame + num_frames - 1
      print "[Info] shot#{} (start_frame,end_frame)=({},{})".format(
              shot_count,
              start_frame,
              end_frame
              )
      all_RGB_frames = sorted(glob.glob('{}/*.jpg'.format(video)))
      RGB_frames = []
      for this_frame in all_RGB_frames:
        frame_num = int(os.path.basename(this_frame).split('.')[-2])
        if frame_num >= start_frame and frame_num <= end_frame:
          RGB_frames.append(this_frame)
      shot = dict()
      shot['start_time'] = begin_sec
      shot['stop_time'] = begin_sec + duration_sec
      shot['RGB_frames'] = RGB_frames
      shots.append(shot)

      if verbose:
        print "[Info] len(RGB_frames)={}".format(len(RGB_frames))

  else:
    RGB_frames = sorted(glob.glob('{}/*.jpg'.format(video)))
    if not RGB_frames:
      print "[Fatal] no RGB images found"
      sys.exit(-1)
    shot = dict()
    shot['start_time'] = 0.0
    shot['stop_time'] = video_length
    shot['RGB_frames'] = RGB_frames
    shots.append(shot)

  payload = dict()
  payload["detections"] = []
  print "[Info] Processing video={}".format(video_id)
  for shot_count, shot in enumerate(shots):
    print "-" * 19
    print "[Info] Processing shot={}/{}".format(shot_count+1, len(shots))
    RGB_frames = shot['RGB_frames']
    start_time = time.time()
    class_RGB_LRCN, predictions_RGB_LRCN = LRCN_classify_video(
                                                               RGB_frames,
                                                               RGB_lstm_net,
                                                               transformer_RGB
                                                               )
    RGB_lstm_processing_time = (time.time() - start_time)
    avg_pred = np.mean(predictions_RGB_LRCN,0).flatten()
    detected_categories = set(np.where(avg_pred >= detection_threshold)[0])

    outf = open(result_file, 'w')
    detections = []
    for det in detected_categories:
      this_det = dict()
      # find start and stop time
      this_det["instance_occurrences"] = [
              [shot['start_time'], shot['stop_time']]
              ]
      this_det["name"] = category_hash[det]
      ct_id = revmapper(id_to_name_map, category_hash[det])
      if verbose:
        print "[Info] category_hash['{}']={}".format(
                det,
                category_hash[det]
                )
      if ct_id:
        this_det["id"] = ct_id
      else:
        this_det["id"] = 0
      if verbose:
        print "[Info] ct_id={}".format(ct_id)
      this_det["salience"] = (shot['stop_time']-shot['start_time']) / video_length
      detections.append(this_det)

    payload["detections"].append(detections)

  del RGB_lstm_net

  # clean up detections
  payload["detections"] = merge_timelines(payload["detections"])
  payload["length"] = video_length
  payload["original_url"] = "http://dextro.co/duh"
  payload["request_id"] = "hello_world"
  payload["lstm_processing_time"] = RGB_lstm_processing_time

  outf.write('Received: {}'.format(json.dumps(payload)))
  outf.close()

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

if __name__ == '__main__':
    main()
