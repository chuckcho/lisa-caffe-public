#!/usr/bin/env python

'''
classify_video.py will classify a video using LRCN RGB model
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

# global var
verbose = False
num_categories = 1170
detection_threshold = 0.02
test_video_list_file = 'dextro_benchmark_2016_02_03_test.txt'
train_video_list_file = 'dextro_benchmark_2016_02_03_train.txt'

def get_gt(video_path):
  gt = set()
  with open(test_video_list_file) as f:
    for line in f:
      if video_path in line:
        gt.add(int(line.split(' ')[1].rstrip()))

  with open(train_video_list_file) as f:
    for line in f:
      if video_path in line:
        gt.add(int(line.split(' ')[1].rstrip()))

  return gt

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
    print "[info] len(frames)={}".format(len(frames))
  clip_length = 16
  offset = 8
  input_images = []
  for im in frames:
    input_im = caffe.io.load_image(im)
    if verbose:
      print "[info] input_im.shape={}".format(input_im.shape)
    if (input_im.shape[0] < 240 or input_im.shape[1] < 320):
      if verbose:
        print "[info] enlarge image to fit (240,320)"
      input_im = caffe.io.resize_image(input_im, (240,320))
    input_images.append(input_im)
  vid_length = len(input_images)
  input_data = []
  for i in range(0,vid_length,offset):
    if (i + clip_length) < vid_length:
      if verbose:
        print "[info] input_data += input_images[{}:{}]".format(i,i+clip_length)
      input_data.extend(input_images[i:i+clip_length])
    else:  #video may not be divisible by clip_length
      input_data.extend(input_images[-clip_length:])
  output_predictions = np.zeros((len(input_data),num_categories))
  if verbose:
    print "[info] output_predictions.shape={}".format(output_predictions.shape)
    print "[info] run forward with range(0,len(input_data)={}, clip_length={})".format(
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
        print "[info] i={}, ix={}, caffe_in[ix].shape={}".format(i, ix, caffe_in[ix].shape)
    if verbose:
      print "[info] caffe_in.shape={}".format(caffe_in.shape)
    out = net.forward_all(data=caffe_in, clip_markers=np.array(clip_clip_markers))
    output_predictions[i:i+clip_length] = np.mean(out['probs'],1)
  return np.mean(output_predictions,0).argmax(), output_predictions

def main():

  simulate_only = True
  #perf_csv = 'performance.csv'
  perf_csv = 'performance_simulate.csv'

  mean_RGB = np.zeros((3,1,1))
  mean_RGB[0,:,:] = 103.939
  mean_RGB[1,:,:] = 116.779
  mean_RGB[2,:,:] = 128.68

  transformer_RGB = initialize_transformer(mean_RGB, False)

  #Models and weights
  lstm_model = 'deploy_lstm.prototxt'
  RGB_lstm = 'snapshots_dextro_benchmark_lstm_iter_30000.caffemodel'

  RGB_lstm_net =  caffe.Net(lstm_model, RGB_lstm, caffe.TEST)

  # Load label
  category_map_file = 'content_type_mapping.json'
  if os.path.isfile(category_map_file):
    #if pickle
    #category_hash = pickle.load(open(category_map_file,'rb'))
    category_hash = json.load(open(category_map_file))
  else:
    print "[fatal] category mapping file does not exist."

  # loop over all test videos
  #test_videos = set()
  #with open(test_video_list_file) as f:
  #  for line in f:
  #    test_videos.add(line.split(' ')[0])

  # process short video clips first
  test_videos = sorted(glob.glob('/media/6TB/Videos/dextro-benchmark2/*.mp4'), key=os.path.getsize)
  test_videos = [video[:-4] for video in test_videos]

  bufsize = 0
  outf = open(perf_csv, 'w', bufsize)

  for video in test_videos:
    # Get GT class(es)
    groundtruth = get_gt(video)
    # there can be videos that doesn't have groundtruth
    if not groundtruth:
        continue

    # Extract list of frames in video
    #print "[debug] video={}".format(video)
    RGB_frames = sorted(glob.glob('{}/*.jpg'.format(video)))
    video_id = os.path.basename(video)

    if verbose:
      print "[debug] RGB_frames={}".format(RGB_frames)

    if not RGB_frames:
      print "[fatal] no RGB images found"
      sys.exit(-1)

    print "-" * 19
    print "[info] Processing video={}".format(video_id)
    start_time = time.time()
    if simulate_only:
        RGB_lstm_processing_time = 0.0
        class_RGB_LRCN = -1
        avg_pred = set()
        detected_categories = set()
    else:
        class_RGB_LRCN, predictions_RGB_LRCN = \
                 LRCN_classify_video(
                         RGB_frames,
                         RGB_lstm_net,
                         transformer_RGB,
                         is_flow=False)
        RGB_lstm_processing_time = (time.time() - start_time)
        avg_pred = np.mean(predictions_RGB_LRCN,0).flatten()
        detected_categories = set(np.where(avg_pred >= detection_threshold)[0])

    # show groundtruth
    print "[info] Groundtruth(s): {} ({})".format(
            groundtruth,
            [str(unicode(category_hash[class_ID])) for class_ID in groundtruth]
            )

    # top 5 or class prob >= threshold -- whichever is smaller set
    if not simulate_only:
        top5_categories = set(np.argsort(avg_pred)[-5:][::-1])
        above_threshold_categories = set(np.where(avg_pred >= detection_threshold)[0])
        detected_categories = top5_categories & above_threshold_categories
        print "[info] LRCN top-1 class={} ({}), detected_categories={} ({}), # of detected & gt={}, time={}".format(
                class_RGB_LRCN,
                category_hash[class_RGB_LRCN],
                detected_categories,
                [str(unicode(category_hash[class_ID])) for class_ID in detected_categories],
                len(groundtruth & detected_categories),
                RGB_lstm_processing_time)

    # write # of gt labels, # of detected labels, # of gt&detected, bool of top1 being gt, processing time
    top1_hit = class_RGB_LRCN in groundtruth
    outf.write('{}, {}, {}, {}, {}, {}\n'.format(
            video_id,
            len(groundtruth),
            len(detected_categories),
            len(groundtruth & detected_categories),
            top1_hit,
            RGB_lstm_processing_time
            ))

  outf.close()
  del RGB_lstm_net

if __name__ == '__main__':
    main()
