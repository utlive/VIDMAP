import skimage.io
import skvideo.io
import os
import h5py
from sklearn.externals import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import f1_score
import scipy.misc
import scipy.signal
import numpy as np
from sporco import util
import matplotlib.pyplot as plt
import pylab as py
import glob
from PIL import Image 
import cv2
import sys

def get_negtrainpatches(hdf5_im, hdf5_trainset, hdf5_label, idx=0, traintest=0):
  return genericnegpatcher(hdf5_im, hdf5_trainset, hdf5_label, idx=idx, traintest=traintest)

def get_postrainpatches(hdf5_im, hdf5_trainset, hdf5_label, idx=0, traintest=0):
  return genericpospatcher(hdf5_im, hdf5_trainset, hdf5_label, idx=idx, traintest=traintest)

#def get_postestpatches(hdf5_im, hdf5_lab, trainset, idx=0):
#  patch_list = make_dict("report_train.txt")
#  return genericpospatcher(hdf5_im, hdf5_lab, trainset, idx=idx, patch_list=patch_list, traintest=1)
def genericnegpatcher(hdf5_im, hdf5_trainset, hdf5_label, idx=0, traintest=0):
  width = 256
  height = 256

  lst480p = np.array(glob.glob("/mnt/hd3/scenes/480p/*avi"))
  lst1080p = np.array(glob.glob("/mnt/hd3/scenes/1080p/*avi"))
  lst480p = np.sort(lst480p)
  lst1080p = np.sort(lst1080p)

  if traintest == 0:
    lst = np.hstack((lst1080p[:215], lst480p[:575]))
    reallst = np.hstack((glob.glob("train/*/*.avi"), glob.glob("train/*/*.m2t"), glob.glob("train/*/*.mpg")))
    #lst = lst1080p[:215]
  else:
    lst = np.hstack((lst480p[575:], lst1080p[215:]))
    reallst = np.hstack((glob.glob("test/*/*.avi"), glob.glob("test/*/*.m2t"), glob.glob("test/*/*.mpg")))

  n_samples = len(lst)

  for fname in lst:
    print fname, len(lst)
    vid = skvideo.io.vread(fname)
    # bitrate = 100 - 2500 for 1080p
    base_name = os.path.basename(fname)

    T, H, W, C = vid.shape

    adj_h = H - height
    adj_w = W - width
    iv, jv = np.meshgrid(np.arange(adj_h), np.arange(adj_w), sparse=False, indexing='ij')
    iv = iv.reshape(-1)
    jv = jv.reshape(-1)

    jdx = np.random.permutation(adj_h*adj_w)

    iv = iv[jdx]
    jv = jv[jdx]
    tv = np.arange(0, T-1)

    limit = 0
    for (y, x) in zip(iv, jv):
      np.random.shuffle(tv)
      t = tv[0]
      patch = vid[t:t+2, y:y+height, x:x+width, 0]

      # only care for information in the postivie case

      hdf5_im[idx] = patch
      hdf5_trainset[idx] = traintest
      hdf5_label[idx] = 0

      limit += 1
      idx += 1

      if limit >= 20:
        break

  return idx

def genericpospatcher(hdf5_im, hdf5_trainset, hdf5_label, idx=0, traintest=0):
  width = 256
  height = 256

  lst480p = np.array(glob.glob("/mnt/hd3/scenes/480p/*avi"))
  lst1080p = np.array(glob.glob("/mnt/hd3/scenes/1080p/*avi"))
  lst480p = np.sort(lst480p)
  lst1080p = np.sort(lst1080p)

  if traintest == 0:
    lst = np.hstack((lst1080p[:215], lst480p[:575]))
    reallst = np.hstack((glob.glob("train/*/*.avi"), glob.glob("train/*/*.m2t"), glob.glob("train/*/*.mpg")))
    #lst = lst1080p[:215]
  else:
    lst = np.hstack((lst480p[575:], lst1080p[215:]))
    reallst = np.hstack((glob.glob("test/*/*.avi"), glob.glob("test/*/*.m2t"), glob.glob("test/*/*.mpg")))

  n_samples = len(lst)

  for fname in lst:
    print fname, len(lst)
    vid = skvideo.io.vread(fname)
    # bitrate = 100 - 2500 for 1080p
    # bitrate = 100 - r12500 for 1080p
    
    crf = np.random.randint(24, 38)
    vcprof = ['baseline', 'main', 'high']
    select = np.random.randint(0, 3)
    skvideo.io.vwrite("/tmp/tmp.avi", vid, outputdict={
      '-c:v': 'libx264', 
      '-pix_fmt': 'yuv420p', 
      '-profile:v': vcprof[select], 
      '-crf': str(crf),
    })
    vid = skvideo.io.vread("/tmp/tmp.avi")
    base_name = os.path.basename(fname)

    T, H, W, C = vid.shape

    adj_h = H - height
    adj_w = W - width
    iv, jv = np.meshgrid(np.arange(adj_h), np.arange(adj_w), sparse=False, indexing='ij')
    iv = iv.reshape(-1)
    jv = jv.reshape(-1)

    jdx = np.random.permutation(adj_h*adj_w)

    iv = iv[jdx]
    jv = jv[jdx]
    tv = np.arange(0, T-1)

    limit = 0
    for (y, x) in zip(iv, jv):
      np.random.shuffle(tv)
      t = tv[0]
      patch = vid[t:t+2, y:y+height, x:x+width, 0]

      # only care for information in the postivie case
      if np.std(patch[0]) < 20:
        continue
      if np.std(patch[1]) < 20:
        continue
        #skimage.io.imsave("extract/%d.png" % (idx,), patch[0].astype(np.uint8))

      hdf5_im[idx] = patch
      hdf5_trainset[idx] = traintest
      hdf5_label[idx] = 1

      limit += 1
      idx += 1

      if limit >= 20:
        break

  return idx

# get the number of patches
np.random.seed(12345)

n_total_images = 63012#43000*2 #12000
patch_height = 256
patch_width = 256
n_channels = 2

# sf = single frame
# fd = frame diff
f = h5py.File('/mnt/hd2/compressiondataset.hdf5', mode='w')

image_patches = f.create_dataset('image_patches', (n_total_images, n_channels, patch_height, patch_width), dtype='float')
image_patches.dims[0].label = 'batch'
image_patches.dims[1].label = 'channel'
image_patches.dims[2].label = 'height'
image_patches.dims[3].label = 'width'

trainset = f.create_dataset('set', (n_total_images,), dtype='uint8')
contentIdx = f.create_dataset('contentNumber', (n_total_images,), dtype='uint8')
label = f.create_dataset('label', (n_total_images,), dtype='uint8')

n_idx = 0
n_idx = get_postrainpatches(image_patches, trainset, label, n_idx, traintest=0)
n_idx = get_negtrainpatches(image_patches, trainset, label, n_idx, traintest=0)
n_idx = get_postrainpatches(image_patches, trainset, label, n_idx, traintest=1)
n_idx = get_negtrainpatches(image_patches, trainset, label, n_idx, traintest=1)
print n_idx, n_total_images

f.flush()
f.close()
