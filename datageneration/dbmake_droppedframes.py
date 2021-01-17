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

# normalizations
def gauss_window(lw, sigma):
    sd = float(sigma)
    lw = int(lw)
    weights = [0.0] * (2 * lw + 1)
    weights[lw] = 1.0
    sum = 1.0
    sd *= sd
    for ii in range(1, lw + 1):
        tmp = np.exp(-0.5 * float(ii * ii) / sd)
        weights[lw + ii] = tmp
        weights[lw - ii] = tmp
        sum += 2.0 * tmp
    for ii in range(2 * lw + 1):
        weights[ii] /= sum
    return weights
avg_window = gauss_window(3, 1.0)
var_window = gauss_window(3, 1.0)

def hp_image(image):
    extend_mode = 'reflect'
    image = np.array(image).astype(np.float32)
    w, h = image.shape
    mu_image = np.zeros((w, h))
    scipy.ndimage.correlate1d(image, avg_window, 0, mu_image, mode=extend_mode) 
    scipy.ndimage.correlate1d(mu_image, avg_window, 1, mu_image, mode=extend_mode) 
    return image - mu_image, mu_image

def var_image(hpimg):
    extend_mode = 'reflect'
    w, h = hpimg.shape
    varimg = np.zeros((w, h))
    scipy.ndimage.correlate1d(hpimg**2, var_window, 0, varimg, mode=extend_mode)
    scipy.ndimage.correlate1d(varimg, var_window, 1, varimg, mode=extend_mode)
    return varimg

def preprocess_image(rgbimg):
    win = np.array(skvideo.utils.gen_gauss_window(13, 7.0/6.0))
    buff = np.zeros_like(rgbimg)
    for i, img in enumerate(rgbimg):
      a, b, c = skvideo.utils.compute_image_mscn_transform(img, avg_window=win, C = 1.0)
      buff[i] = img - c # highpass processing
    buff = buff[:, 20:-20, 20:-20]
    return buff

def load_coords(f):
  fi = open(f)
  coords = []
  for item in fi:
    parts = item.split(',')
    coords.append([np.int(parts[1]), np.int(parts[0])])
  fi.close()
  return np.array(coords)

def make_dict(place):
  dat = {}
  fi = open(place)
  for line in fi:
    parts = line.split(',')
    fpath = parts[0]
    y = np.int(parts[2])
    x = np.int(parts[1])
    if fpath not in dat:
      dat[fpath] = []
    dat[fpath].append([y, x])
  return dat

def get_postrainpatches(hdf5_im, hdf5_lab, hdf5_trainset, idx=0, traintest=0):
  return genericpospatcher(hdf5_im, hdf5_lab, hdf5_trainset, idx=idx, traintest=traintest)

#def get_postestpatches(hdf5_im, hdf5_lab, trainset, idx=0):
#  patch_list = make_dict("report_train.txt")
#  return genericpospatcher(hdf5_im, hdf5_lab, trainset, idx=idx, patch_list=patch_list, traintest=1)

def genericpospatcher(hdf5_im, hdf5_lab, hdf5_trainset, idx=0, traintest=0):
  width = 64
  height = 64

  lst480p = np.array(glob.glob("/mnt/hd3/scenes/480p/*avi"))
  lst1080p = np.array(glob.glob("/mnt/hd3/scenes/1080p/*avi"))
  lst480p = np.sort(lst480p)
  lst1080p = np.sort(lst1080p)

  if traintest == 0:
    lst = np.hstack((lst480p[:437], lst1080p[:215]))
    totalcontents = np.hstack((np.arange(437), np.arange(1000, 1215)))
  else:
    lst = np.hstack((lst480p[437:], lst1080p[215:]))
    totalcontents = np.hstack((np.arange(437,437 + len(lst480p[437:])), np.arange(1215, 1215 + len(lst1080p[215:]))))

  n_samples = len(lst)
  for fcount, fname in enumerate(lst):
    print fcount, n_samples, idx, fname
    vid = skvideo.io.vread(fname, as_grey=True).astype(np.float32)
    # get frame rate, only use frame rates of a certain type
    # get 1 second of frames before and after

    T, H, W, C = vid.shape

    if T < 30:
      continue

    adj_h = H - height
    adj_w = W - width
    iv, jv = np.meshgrid(np.arange(adj_h), np.arange(adj_w), sparse=False, indexing='ij')
    iv = iv.reshape(-1)
    jv = jv.reshape(-1)

    jdx = np.random.permutation(adj_h*adj_w)

    iv = iv[jdx]
    jv = jv[jdx]

    # drop of 3 frames (hardest perhaps)
    for NMidx, NM in enumerate([3, 6, 9]):
      tv = np.arange(10, T-NM-10)
      limit = 0
      searchiterates = 0
      for (y, x) in zip(iv, jv):
        searchiterates += 1
        if searchiterates > 1000:
          break
        np.random.shuffle(tv)
        t = tv[0]
        vidslicestart= vid[t-10:t, y:y+height, x:x+width, 0].astype(np.float32)
        vidsliceend = vid[t+NM:t+NM+10, y:y+height, x:x+width, 0].astype(np.float32)

        vidslice = np.zeros((20, height, width), dtype=np.float32)
        vidslice[0:10] = vidslicestart
        vidslice[10:] = vidsliceend
        # drop the frames
        # make sure the difference between frames is large enough
        # check for no variance img
        if np.std(vidslice[8]) < 10:
          continue
        if np.std(vidslice[9]) < 10:
          continue
        if np.std(vidslice[10]) < 10:
          continue
        if np.std(vidslice[11]) < 10:
          continue

        # check the TI difference signal
        TI = np.std(vidslice[9] - vidslice[10])
        if TI < 5: # incredibly low motion difference
          continue
        TI = np.std(vidslice[8] - vidslice[9])
        if TI < 5: # incredibly low motion difference
          continue
        TI = np.std(vidslice[10] - vidslice[11])
        if TI < 5: # incredibly low motion difference
          continue


        #preprocess = preprocess[:, 5:-5, 5:-5]
        hdf5_im[idx] = vidslice
        hdf5_lab[idx] = NMidx+1
        hdf5_trainset[idx] = traintest

        #skimage.io.imsave("extract/%d.png" % (idx,), patch)
        limit += 1
        idx += 1
        if limit >= 20:
          break

  return idx

def get_negtrainpatches(image_patches, labels, hdf5_trainset, idx=0, traintest=0):
  return genericnegpatcher(image_patches, labels, hdf5_trainset, idx=idx, traintest=traintest)

def genericnegpatcher(hdf5_im, hdf5_lab, hdf5_trainset, idx=0, traintest=0):
  width = 64
  height = 64

  lst480p = np.array(glob.glob("/mnt/hd3/scenes/480p/*avi"))
  lst1080p = np.array(glob.glob("/mnt/hd3/scenes/1080p/*avi"))
  lst480p = np.sort(lst480p)
  lst1080p = np.sort(lst1080p)

  if traintest == 0:
    lst = np.hstack((lst480p[:437], lst1080p[:215]))
    totalcontents = np.hstack((np.arange(437), np.arange(1000, 1215)))
  else:
    lst = np.hstack((lst480p[437:], lst1080p[215:]))
    totalcontents = np.hstack((np.arange(437,437 + len(lst480p[437:])), np.arange(1215, 1215 + len(lst1080p[215:]))))

  n_samples = len(lst)
  for fcount, fname in enumerate(lst):
    print fcount, n_samples, idx, fname
    vid = skvideo.io.vread(fname, as_grey=True).astype(np.float32)
    # get frame rate, only use frame rates of a certain type
    # get 1 second of frames before and after

    T, H, W, C = vid.shape

    if T < 30:
      continue

    adj_h = H - height
    adj_w = W - width
    iv, jv = np.meshgrid(np.arange(adj_h), np.arange(adj_w), sparse=False, indexing='ij')
    iv = iv.reshape(-1)
    jv = jv.reshape(-1)

    jdx = np.random.permutation(adj_h*adj_w)

    iv = iv[jdx]
    jv = jv[jdx]

    # drop of 3 frames (hardest perhaps)
    tv = np.arange(10, T-10)
    limit = 0
    searchiterates = 0
    for (y, x) in zip(iv, jv):
      searchiterates += 1
      if searchiterates > 1000:
        break
      np.random.shuffle(tv)
      t = tv[0]
      vidslicestart= vid[t-10:t+10, y:y+height, x:x+width, 0].astype(np.float32)

      # drop the frames
      # make sure the difference between frames is large enough
      hdf5_im[idx] = vidslicestart
      hdf5_lab[idx] = 0
      hdf5_trainset[idx] = traintest

      #skimage.io.imsave("extract/%d.png" % (idx,), patch)
      limit += 1
      idx += 1
      if limit >= 20:
        break

  return idx

# get the number of patches
np.random.seed(12345)

n_total_images = 85206
patch_height = 256
patch_width = 256
n_channels = 20
n_frames = 100

# sf = single frame
# fd = frame diff
f = h5py.File('/mnt/hd2/droppedFramesdataset_cubes.hdf5', mode='w')

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
n_idx = get_negtrainpatches(image_patches, labels, trainset, n_idx, 1)
n_idx = get_postrainpatches(image_patches, labels, trainset, n_idx, 1)
print n_idx, n_total_images

f.flush()
f.close()
