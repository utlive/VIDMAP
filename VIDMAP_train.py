import tensorflow as tf
import h5py
import skvideo.utils
import numpy as np
import sklearn.metrics
import sys
import hashlib

try:
    import Queue
except:
    import queue as Queue

import threading
import os
from sklearn.externals import joblib
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
    processed_data = np.zeros((data.shape[0], data.shape[1]-10, data.shape[1]-10, 2))
    for i in range(data.shape[0]):
          a, b, c = skvideo.utils.compute_image_mscn_transform(data[i, :, :], avg_window=win, C=1.0)
          a = a[5:-5, 5:-5]
          b = b[5:-5, 5:-5]
          # find max point in b
          processed_data[i, :, :, 0] = a #/255.0
          processed_data[i, :, :, 1] = b/255.0
  return processed_data

def load_task(hdf5_patches, hdf5_labels, train_idx1, batch_size, atype, vtype):
    half_batch = batch_size//2
    n_iterations = np.min((
      len(train_idx1)/(half_batch), 
      len(train_idx2)/(half_batch), 
    ))
    n_iterations = np.int(n_iterations)

    for M in range(0, n_iterations):
      lst1 = np.sort(train_idx1[M*(half_batch):(M+1)*(half_batch)]).tolist()
      lst2 = np.sort(train_idx2[M*(half_batch):(M+1)*(half_batch)]).tolist()
      X_batch1 = hdf5_patches[lst1]
      X_batch2 = hdf5_patches[lst2]
      X_batch = np.vstack((X_batch1, X_batch2))
      
      labelr = np.hstack((hdf5_labels[lst1].astype(np.uint8), hdf5_labels[lst2].astype(np.uint8)))
      labelr[labelr>0] = 1

      y_labels = np.eye(2)[labelr]

      if atype in ["hitsh264", "hitsmpeg2", "combing"]:
        if vtype == "framediff":
          X_batch = X_batch[:, 1, :, :] - X_batch[:, 0, :, :]
        else:
          X_batch = X_batch[:, 1, :, :]
        X_batchp = preprocess(X_batch, atype)
      elif atype == "droppedFrames":
        X_batch1 = X_batch[:, 1, :, :] - X_batch[:, 0, :, :]
        X_batch2 = X_batch[:, 2, :, :] - X_batch[:, 1, :, :]
        X_batch3 = X_batch[:, 3, :, :] - X_batch[:, 2, :, :]

        X_batchn = np.zeros((X_batch1.shape[0],X_batch1.shape[1], X_batch1.shape[2], 3)) 
        X_batchn[:, :, :, 0] = X_batch1
        X_batchn[:, :, :, 1] = X_batch2
        X_batchn[:, :, :, 2] = X_batch3

        X_batchp = preprocess(X_batchn, atype)

      else:
        if vtype == "framediff":
          X_batch = X_batch[:, 1, :, :] - X_batch[:, 0, :, :]
        else:
          X_batch = X_batch[:, 0, :, :]
        X_batchp = preprocess(X_batch, atype)

      # randomly flip horizontally/vertically
      # to reduce overfitting to training set 
      # if your artifact depends on left/right or
      # up/down orientation, you MUST disable this
      ridx = np.random.randint(0, 4)
      if ridx == 0:
        X_batchp = X_batchp[:, ::-1, :]
      if ridx == 1:
        X_batchp = X_batchp[:, :, ::-1]
      if ridx == 2:
        X_batchp = X_batchp[:, ::-1, ::-1]
        
      data_queue.put([X_batchp, y_labels], True)

    # tell the other side we're "done"
    data_queue.put(sentinel, True)

md5lut = {}
md5lut['aliasing'] = "d3ba75e2bd0ffeba7ab485acc442609c"
md5lut['compression'] = "77eafeb723665de10c344b3c8a531f02"
md5lut['combing'] = "aceb56fd2bfa147325b0788583698d82"
md5lut['contour'] = "186e1f9aba010a34a9cc4ee929a08c2b"
md5lut['upscaling'] = "7d279eb019c685452d95c5357adaabaf"
md5lut['aspectratio'] = "0f19de01c42cd64fee819726cd5648c5"
md5lut['banding'] = "3bc771e239ffd95b9e03384c0ba696a9"
md5lut['droppedFrames'] = "4ce75b3e252ccbe40d2a2a6bf53b2833"
md5lut['hitsh264'] = "3727cbcec85a1ac11bc8cf650b8f20af"
md5lut['hitsmpeg2'] = "c47d84cc4ada2f11e003fbfafea3e643"

lutdata = {}
lutdata['aliasing'] = ["/home/todd/aliasingdataset_sf4.hdf5",249288]
lutdata['compression'] = ["/home/todd/compressiondataset.hdf5",63012]
lutdata['combing'] = ["/home/todd/combingdataset_sf.hdf5",61653]
lutdata['contour'] = ["/home/todd/contourdataset_sf.hdf5",730600]
lutdata['upscaling'] = ["/home/todd/upscalingdataset_sf.hdf5",129428]
lutdata['aspectratio'] = ["/home/todd/aspectratiodataset_sf.hdf5",34360]
lutdata['banding'] = ["/home/todd/bandingdataset_sf.hdf5", 64925]
lutdata['droppedFrames'] = ["/mnt/droppedFramesdataset_sf.hdf5", 51562]
lutdata['hitsh264'] = ["/home/todd/hitsdataset_sf_h264.hdf5", 62417]
lutdata['hitsmpeg2'] = ["/home/todd/hitsdataset_sf_mpeg2.hdf5", 59941]

if __name__ == "__main__":

  if len(sys.argv) == 1:
    print("Usage:")
    print("  python VIDMAP_train.py [artifact] [single/framediff/2layer]")
    exit(0)

  atype = sys.argv[1]

  VIDMAPtype = sys.argv[2]

  outputdir = VIDMAPtype

  win = np.array(skvideo.utils.gen_gauss_window(2, 7.0/6.0))

  h5py_file = lutdata[atype][0]
  print("dataset hash: ", md5_for_file(lutdata[atype][0], block_size=128*10000))
  assert (md5_for_file(lutdata[atype][0], block_size=128*10000) == md5lut[atype]) 
  f = h5py.File(h5py_file, mode='r')
  os.system("mkdir -p " + outputdir + "/" + atype + "/")

  f = h5py.File(h5py_file, mode='r')
  image_patches = f['image_patches']
  print(f.keys())
  if "label" in f:
    glabels = f['label']
  else:
    glabels = f['labels']
  sets = f['set']

  n_totalpatches, n_channels, patch_height, patch_width = image_patches.shape
  batch_size = 10

  # remove part of the border due to pre-processing
  patch_height -= 10
  patch_width -= 10

  n_totalpatches = lutdata[atype][1]

  train_indices1 = np.arange(n_totalpatches)[(sets[:n_totalpatches] == 0) & (glabels[:n_totalpatches] == 0)]
  train_indices2 = np.arange(n_totalpatches)[(sets[:n_totalpatches] == 0) & (glabels[:n_totalpatches] == 1)]

  halfbatch = batch_size/2
  n_iterations = np.min((
    len(train_indices1)/(halfbatch), 
    len(train_indices2)/(halfbatch), 
  ))
  n_iterations = np.int(n_iterations)

  sess = tf.InteractiveSession()

  n_C = 2
  if atype == "droppedFrames":
    n_C = 3

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

    return loss, probabilities, var

  loss1, prob1, var = gen_network(xinput, yinput, n_filters=50, atype=atype, vtype=VIDMAPtype)

  train_step1 = tf.train.AdamOptimizer(1e-4).minimize(loss1)
  sess.run(tf.global_variables_initializer())
  saver = tf.train.Saver(max_to_keep=None)

  for EPOCH in range(0, 1000):
    if EPOCH > 1:
      saver.save(sess, outputdir + "/%s/trained.ckpt" % (atype,))

    np.random.shuffle(train_indices1)
    np.random.shuffle(train_indices2)

    data_queue = Queue.Queue(2)
    p = threading.Thread(target=load_task, args=(image_patches, glabels, train_indices1, train_indices2, batch_size, atype, VIDMAPtype))
    p.daemon = True
    p.start()

    M = 0
    losses = []
    evall = []
    while True:
        if ((M % 100) == 0):
           print(M, n_iterations)

        batch = data_queue.get(True)
        data_queue.task_done()

        if batch is sentinel:
          break # we're done now!

        _, l1, p1 = sess.run([train_step1, loss1, prob1], feed_dict={xinput: batch[0], yinput: batch[1]})
        losses.append([l1])

        stack = np.hstack((
          np.argmax(p1, axis=1).reshape(-1, 1), 
          np.argmax(batch[1], axis=1).reshape(-1, 1)
        ))

        if evall == []:
          evall = stack
        else:
          evall = np.vstack((evall, stack))

        M += 1

    losses = np.array(losses)
    print("losses: ", np.mean(losses, axis=0))

    evall = np.array(evall)
    f1_p1 = sklearn.metrics.f1_score(evall[:,1], evall[:, 0]) 
    acc_p1 = np.mean(evall[:,1] == evall[:, 0])

    print("n=50 (%0.4f, %0.4f)" % (f1_p1, acc_p1))
