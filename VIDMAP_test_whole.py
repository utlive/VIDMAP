import tensorflow as tf
import skvideo.io
import skvideo.utils
import numpy as np
import sys

try:
    import Queue
except:
    import queue as Queue

import threading
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

sentinel = object()

def md5_for_file(fname, block_size=512):
  f = open(fname, 'rb')
  md5 = hashlib.md5()
  data = f.read(block_size)
  md5.update(data)
  return md5.hexdigest()

def preprocess(data, atype):
  if atype == "droppedFrames":
    processed_data = np.zeros((data.shape[0], data.shape[1]-10, data.shape[2]-10, 3))
    for i in range(data.shape[0]):
      for j in range(3):
        a, b, c = skvideo.utils.compute_image_mscn_transform(data[i, :, :, j], avg_window=win, C=1.0)
        a = a[5:-5, 5:-5]
        processed_data[i, :, :, j] = a 
  else:
    processed_data = np.zeros((data.shape[0], data.shape[1]-10, data.shape[2]-10, 2))
    for i in range(data.shape[0]):
          a, b, c = skvideo.utils.compute_image_mscn_transform(data[i, :, :], avg_window=win, C=1.0)
          a = a[5:-5, 5:-5]
          b = b[5:-5, 5:-5]
          # find max point in b
          processed_data[i, :, :, 0] = a #/255.0
          processed_data[i, :, :, 1] = b/255.0
  return processed_data

if __name__ == "__main__":

  if len(sys.argv) == 1:
    print("Usage:")
    print("  python VIDMAP_test_whole.py [artifact] [single/framediff/2layer] [input_video_file]")
    exit(0)

  atype = sys.argv[1]

  VIDMAPtype = sys.argv[2]

  videoInputFile = sys.argv[3]

  outputdir = VIDMAPtype

  vidData = skvideo.io.vread(videoInputFile, as_grey=True)[:, :, :, 0].astype(np.float32)

  win = np.array(skvideo.utils.gen_gauss_window(2, 7.0/6.0))

  # remove part of the border due to pre-processing
  patch_height = vidData.shape[1] - 10
  patch_width = vidData.shape[2] - 10

  sess = tf.InteractiveSession()

  n_C = 2
  if atype == "droppedFrames":
    n_C = 3

  batch_size = 1

  xinput = tf.placeholder(tf.float32, shape=[batch_size, patch_height, patch_width, n_C])
  yinput = tf.placeholder(tf.float32, shape=[None, 2])

  def gen_network(x, y, n_filters=10, atype="", vtype=""):
    n_C = 2
    if atype == "droppedFrames":
      n_C = 3

    W1_class0 = tf.Variable(tf.random_normal([11, 11, n_C, n_filters], stddev=0.0001))
    W1_class1 = tf.Variable(tf.random_normal([11, 11, n_C, n_filters], stddev=0.0001))
    b1_class0 = tf.Variable(tf.random_normal([n_filters], stddev=0.001))
    b1_class1 = tf.Variable(tf.random_normal([n_filters], stddev=0.001))

    if vtype != "2layer":
      W2_class0 = tf.Variable(tf.random_normal([11, 11, n_filters, n_filters], stddev=0.0001))
      W2_class1 = tf.Variable(tf.random_normal([11, 11, n_filters, n_filters], stddev=0.0001))
      b2_class0 = tf.Variable(tf.random_normal([n_filters], stddev=0.001))
      b2_class1 = tf.Variable(tf.random_normal([n_filters], stddev=0.001))

    W3_class0 = tf.Variable(tf.random_normal([11, 11, n_filters, 1], stddev=0.0001))
    W3_class1 = tf.Variable(tf.random_normal([11, 11, n_filters, 1], stddev=0.0001))
    b3_class0 = tf.Variable(tf.random_normal([1], stddev=0.001))
    b3_class1 = tf.Variable(tf.random_normal([1], stddev=0.001))

    if vtype == "2layer":
      var = [W1_class0, W1_class1, b1_class0, b1_class1, W3_class0, W3_class1, b3_class0, b3_class1]
    else:
      var = [W1_class0, W1_class1, b1_class0, b1_class1, W2_class0, W2_class1, b2_class0, b2_class1, W3_class0, W3_class1, b3_class0, b3_class1]


    net_code_class0 = tf.nn.conv2d(x, W1_class0, strides=[1, 1, 1, 1], padding='VALID') + b1_class0
    net_code_class0 = tf.contrib.layers.group_norm(net_code_class0, groups=n_filters)
    net_code_class0 = tf.nn.relu(net_code_class0)

    if vtype != "2layer":
      net_code_class0 = tf.nn.conv2d(net_code_class0, W2_class0, strides=[1, 1, 1, 1], padding='VALID') + b2_class0
      net_code_class0 = tf.contrib.layers.group_norm(net_code_class0, groups=n_filters)
      net_code_class0 = tf.nn.relu(net_code_class0)

    net_code_class0 = tf.nn.conv2d(net_code_class0, W3_class0, strides=[1, 1, 1, 1], padding='VALID') + b3_class0

    net_code_class1 = tf.nn.conv2d(x, W1_class1, strides=[1, 1, 1, 1], padding='VALID') + b1_class1
    net_code_class1 = tf.contrib.layers.group_norm(net_code_class1, groups=n_filters)
    net_code_class1 = tf.nn.relu(net_code_class1)

    if vtype != "2layer":
      net_code_class1 = tf.nn.conv2d(net_code_class1, W2_class1, strides=[1, 1, 1, 1], padding='VALID') + b2_class1
      net_code_class1 = tf.contrib.layers.group_norm(net_code_class1, groups=n_filters)
      net_code_class1 = tf.nn.relu(net_code_class1)

    net_code_class1 = tf.nn.conv2d(net_code_class1, W3_class1, strides=[1, 1, 1, 1], padding='VALID') + b3_class1

    net_flat_class0 = tf.reshape(net_code_class0, [batch_size, -1])
    net_flat_class1 = tf.reshape(net_code_class1, [batch_size, -1, 1])

    net_flat_mask = tf.zeros_like(net_flat_class0)

    total_logits = tf.zeros((batch_size, 2), dtype=tf.float32)
    loss = 0
    idx = []
    numScales = 1
    d0 = net_code_class0
    d1 = net_code_class1
    tmp = net_code_class1 - net_code_class0

    tmp = tf.reshape(tmp, [batch_size, -1])
    d0 = tf.reshape(d0, [batch_size, -1])
    d1 = tf.reshape(d1, [batch_size, -1])

    maxidx1 = tf.argmax(tmp, axis=1)
    maxidx1 = tf.cast(maxidx1, tf.int32)

    logits1 = []
    for j in range(batch_size):
      logits1.append([d0[j, maxidx1[j]], d1[j, maxidx1[j]]])
    logits1 = tf.stack(logits1)

    #loss += tf.losses.softmax_cross_entropy(y, logits1)
    loss += tf.losses.hinge_loss(y, logits1)

    total_logits += logits1

    probimg = tf.nn.softmax(tf.stack([net_code_class0[:, :, :, 0], net_code_class1[:, :, :, 0]], axis=3))
    probimg = probimg[:, :, :, 1]

    probabilities = tf.nn.softmax(total_logits)

    return probimg, probabilities

  prob1, pred = gen_network(xinput, yinput, n_filters=50, atype=atype, vtype=VIDMAPtype)

  sess.run(tf.global_variables_initializer())
  saver = tf.train.Saver(max_to_keep=None)

  saver.restore(sess, outputdir + "/%s/trained.ckpt" % (atype,))

  writer = skvideo.io.FFmpegWriter("%s_prediction_%s.avi" % (atype,VIDMAPtype), outputdict={'-vcodec':'rawvideo', '-pix_fmt':'yuv420p'}) 

  if atype == "droppedFrames":
      for fidx in range(vidData.shape[0]-3):
        frame1 = vidData[[fidx]]
        frame2 = vidData[[fidx+1]]
        frame3 = vidData[[fidx+2]]
        frame4 = vidData[[fidx+3]]

        buff = np.zeros((1, frame1.shape[1], frame1.shape[2], 3), dtype=np.float32)
        buff[0, :, :, 0] = frame2 - frame1
        buff[0, :, :, 1] = frame3 - frame2
        buff[0, :, :, 2] = frame4 - frame3

        preproc = preprocess(buff, atype)

        pimg, p1 = sess.run([prob1, pred], feed_dict={xinput: preproc})

        writer.writeFrame(pimg*255)
        print("Frame #%d, probability of %s: %0.3f" % (fidx, atype, p1[0, 1]))

      writer.close()
  else:
    if (VIDMAPtype == "single") or (VIDMAPtype == "2layer"):
      for fidx, frame in enumerate(vidData):
        frame = frame[np.newaxis]
        preproc = preprocess(frame, atype)
        pimg, p1 = sess.run([prob1, pred], feed_dict={xinput: preproc})

        writer.writeFrame(pimg*255)
        print("Frame #%d, probability of %s: %0.3f" % (fidx, atype, p1[0, 1]))

      writer.close()
    elif (VIDMAPtype == "framediff"):
      for fidx in range(vidData.shape[0]-1):
        frame1 = vidData[[fidx]]
        frame2 = vidData[[fidx+1]]
        preproc = preprocess(frame2 - frame1, atype)

        pimg, p1 = sess.run([prob1, pred], feed_dict={xinput: preproc})

        print("Frame #%d, probability of %s: %0.3f" % (fidx, atype, p1[0, 1]))

        writer.writeFrame(pimg*255)

      writer.close()
