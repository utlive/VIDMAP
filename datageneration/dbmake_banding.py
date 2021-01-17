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

def get_postrainpatches(hdf5_im, hdf5_lab, hdf5_trainset, idx=0, traintest=0):
  return genericpospatcher(hdf5_im, hdf5_lab, hdf5_trainset, idx=idx, traintest=traintest)

def genericpospatcher(hdf5_im, hdf5_lab, hdf5_trainset, idx=0, traintest=0):
  width = 256+40
  height = 256+40

  lst480p = np.array(glob.glob("/mnt/hd3/scenes/480p/*avi"))
  lst1080p = np.array(glob.glob("/mnt/hd3/scenes/1080p/*avi"))
  lst480p = np.sort(lst480p)
  lst1080p = np.sort(lst1080p)

  if traintest == 0:
    lst = np.hstack((lst480p[:575], lst1080p[:215]))
  else:
    lst = np.hstack((lst480p[575:], lst1080p[215:]))

  n_samples = len(lst)
  for jjj, fname in enumerate(lst):
    print jjj, n_samples
    vid_pris = skvideo.io.vread(fname, as_grey=True).astype(np.float32)
    T, H, W, C = vid_pris.shape

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
      goodpatch = vid_pris[t:t+2, y:y+height, x:x+width, 0]
      badpatch = goodpatch.copy()

      # random amount of change
      amt = np.random.randint(3, 6)
      badpatch /= 2**amt
      badpatch = np.floor(badpatch)
      badpatch *= 2**amt

      diff = np.mean((badpatch[0] - goodpatch[0])**2)
      if diff < 1.5:
        continue
      if np.std(badpatch[0]) < 20:
        continue
      if np.std(badpatch[1]) < 20:
        continue
      #print diff
      #skimage.io.imsave("extract/test_%d.png" % (idx,), np.hstack((goodpatch[0].astype(np.uint8), badpatch[0].astype(np.uint8))))

      #preprocess = preprocess_image(badpatch)

      #preprocess = preprocess[:, 5:-5, 5:-5]
      badpatch = badpatch[:, 20:-20, 20:-20]

      hdf5_im[idx] = badpatch
      hdf5_lab[idx] = 1
      hdf5_trainset[idx] = traintest

      #skimage.io.imsave("extract/%d.png" % (idx,), patch)
      limit += 1
      idx += 1
      if limit >= 25:
        break

  return idx

def get_negtrainpatches(image_patches, labels, trainset, idx=0, traintest=0):
  return genericnegpatcher(image_patches, labels, trainset, idx=idx, traintest=traintest)

def genericnegpatcher(hdf5_im, hdf5_lab, hdf5_trainset, idx=0, traintest=0):
  width = 256+40
  height = 256+40

  lst480p = np.array(glob.glob("/mnt/hd3/scenes/480p/*avi"))
  lst1080p = np.array(glob.glob("/mnt/hd3/scenes/1080p/*avi"))
  lst480p = np.sort(lst480p)
  lst1080p = np.sort(lst1080p)

  if traintest == 0:
    lst = np.hstack((lst480p[:575], lst1080p[:215]))
  else:
    lst = np.hstack((lst480p[575:], lst1080p[215:]))

  n_samples = len(lst)
  for jjj, fname in enumerate(lst):
    print jjj, n_samples
    vid_pris = skvideo.io.vread(fname, as_grey=True).astype(np.float32)
    T, H, W, C = vid_pris.shape

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
      goodpatch = vid_pris[t:t+2, y:y+height, x:x+width, 0]

      #preprocess = preprocess[:, 5:-5, 5:-5]
      goodpatch = goodpatch[:, 20:-20, 20:-20]
      #if np.std(goodpatch[0]) < 20:
      #  continue
      #if np.std(goodpatch[1]) < 20:
      #  continue

      hdf5_im[idx] = goodpatch
      hdf5_lab[idx] = 0
      hdf5_trainset[idx] = traintest

      #skimage.io.imsave("extract/%d.png" % (idx,), patch)
      limit += 1
      idx += 1
      if limit >= 25:
        break

  return idx

# get the number of patches
np.random.seed(12345)

n_total_images = 64925#12000
patch_height = 256
patch_width = 256
n_channels = 2

# sf = single frame
# fd = frame diff
f = h5py.File('bandingdataset_sf.hdf5', mode='w')

image_patches = f.create_dataset('image_patches', (n_total_images, n_channels, patch_height, patch_width), dtype='float')
image_patches.dims[0].label = 'batch'
image_patches.dims[1].label = 'channel'
image_patches.dims[2].label = 'height'
image_patches.dims[3].label = 'width'

labels = f.create_dataset('labels', (n_total_images,), dtype='uint8')
trainset = f.create_dataset('set', (n_total_images,), dtype='uint8')

n_idx = 0
n_idx = get_postrainpatches(image_patches, labels, trainset, n_idx, 0)
n_idx = get_negtrainpatches(image_patches, labels, trainset, n_idx, 0)
n_idx = get_postrainpatches(image_patches, labels, trainset, n_idx, 1)
n_idx = get_negtrainpatches(image_patches, labels, trainset, n_idx, 1)

print n_idx, n_total_images

f.flush()
f.close()
