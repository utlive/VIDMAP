import skimage.io
import skimage.transform
import skvideo.io
import skvideo.measure

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

def get_postrainpatches(hdf5_im, hdf5_lab, hdf5_traintest, aliasingamt, contentidx, idx=0, traintest=0):
  return genericpospatcher(hdf5_im, hdf5_lab, hdf5_traintest, aliasingamt, contentidx, idx=idx, traintest=traintest)

def genericpospatcher(hdf5_im, hdf5_lab, hdf5_traintest, aliasingamt, contentidx, idx=0, traintest=0):
  width = 100
  height = 100

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
  for jjj, item in enumerate(lst):
    print item, idx, jjj, len(lst)
    vid = skvideo.io.vread(item, as_grey=True)
    T, H, W, C = vid.shape

    tv = np.arange(vid.shape[0]-1)
    tmp112 = 0
    limit = 0
    while True:
      tmp112 += 1
      if tmp112 > 1000:
        break
      np.random.shuffle(tv)
      slc = vid[tv[0]:tv[0]+2, :, :, 0]
      # downscale up to 4x
      factor = np.random.random()*2.0+2.0
      newHeight = np.int32(vid.shape[1] * (1.0/factor))
      newWidth = np.int32(vid.shape[2] * (1.0/factor))

      #goodimg1 = skimage.transform.resize(slc[0], (newHeight, newWidth), anti_aliasing=True)
      #goodimg2 = skimage.transform.resize(slc[1], (newHeight, newWidth), anti_aliasing=True)
      badimg1 = cv2.resize(slc[0], (newWidth, newHeight), interpolation=cv2.INTER_NEAREST).astype(np.float32)
      badimg2 = cv2.resize(slc[1], (newWidth, newHeight), interpolation=cv2.INTER_NEAREST).astype(np.float32)

      goodimg1 = cv2.resize(slc[0], (newWidth, newHeight), interpolation=cv2.INTER_LANCZOS4).astype(np.float32)
      goodimg2 = cv2.resize(slc[1], (newWidth, newHeight), interpolation=cv2.INTER_LANCZOS4).astype(np.float32)

      hh, ww = badimg1.shape

      adj_h = hh - height
      adj_w = ww - width
      iv, jv = np.meshgrid(np.arange(adj_h), np.arange(adj_w), sparse=False, indexing='ij')
      iv = iv.reshape(-1)
      jv = jv.reshape(-1)

      jdx = np.random.permutation(adj_h*adj_w)

      iv = iv[jdx]
      jv = jv[jdx]
      iv = iv[0]
      jv = jv[0]

      badimg1 = badimg1[iv:iv+height, jv:jv+width]
      goodimg1 = goodimg1[iv:iv+height, jv:jv+width]
      badimg2 = badimg2[iv:iv+height, jv:jv+width]
      goodimg2 = goodimg2[iv:iv+height, jv:jv+width]

      if skvideo.measure.ssim(badimg1, goodimg1) > 0.95: 
        continue
      if np.std(badimg1)<20:
        continue
      # look at the 10% worst SSIMs
      # compute local variances
      win = np.array(skvideo.utils.gen_gauss_window(13, 7.0/6.0))
      _, sig1, _ = skvideo.utils.compute_image_mscn_transform(badimg1[15:-15, 15:-15], avg_window=win, C = 1.0)
      _, sig2, _ = skvideo.utils.compute_image_mscn_transform(goodimg1[15:-15, 15:-15], avg_window=win, C = 1.0)
      sig1 = sig1**2
      sig2 = sig2**2
      energy = sig1 - sig2

      energy[energy<50] = 0
      energy[energy>=50] = 1
      energy = energy.astype(np.uint8)
      #skimage.io.imsave("examples/%d.png" % (idx,), energy*255)
      #exit(0)
      # use contour length
      #ret,thresh = cv.threshold(img,127,255,0)
      im2,contours,hierarchy = cv2.findContours(energy, 1, 2)
      minLength = 40
      for item in contours:
        if len(item) > minLength:
          minLength = 0
          break

      if minLength>0:
        continue

      #print diffavg
      badimg1[badimg1<0] = 0
      badimg1[badimg1>255] = 255
      badimg2[badimg2<0] = 0
      badimg2[badimg2>255] = 255
      #badimg1 = badimg1.astype(np.uint8)
      #skimage.io.imsave("examples/%d.png" % (idx,), badimg1)
      hdf5_im[idx, 0, :, :] = badimg1
      hdf5_im[idx, 1, :, :] = badimg2
      hdf5_lab[idx] = 1
      aliasingamt[idx] = factor
      contentidx[idx] = totalcontents[jjj]
      hdf5_traintest[idx] = traintest
      idx += 1
      limit += 1
      if limit >= 100:
        break

  return idx

def get_negtrainpatches(image_patches, labels, hdf5_traintest, aliasingamt, contentidx, idx=0, traintest=0):
  return genericnegpatcher(image_patches, labels, hdf5_traintest, aliasingamt, contentidx, idx=idx, traintest=traintest)

def genericnegpatcher(hdf5_im, hdf5_lab, hdf5_traintest, aliasingamt, contentidx, idx=0, traintest=0):
  width = 100
  height = 100

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
  for jjj, item in enumerate(lst):
    print item, jjj, len(lst)
    vid = skvideo.io.vread(item, as_grey=True)
    T, H, W, C = vid.shape

    tv = np.arange(vid.shape[0]-1)
    tmp112 = 0
    limit = 0
    while True:
      tmp112 += 1
      if tmp112 > 1000:
        break
      np.random.shuffle(tv)
      slc = vid[tv[0]:tv[0]+2, :, :, 0].astype(np.float32)

      # downscale up to 4x
      factor = np.random.random()*3.0+1.0
      goodimg1 = slc[0].astype(np.float32)
      goodimg2 = slc[1].astype(np.float32)

      hh, ww = goodimg1.shape

      adj_h = hh - height
      adj_w = ww - width
      iv, jv = np.meshgrid(np.arange(adj_h), np.arange(adj_w), sparse=False, indexing='ij')
      iv = iv.reshape(-1)
      jv = jv.reshape(-1)

      jdx = np.random.permutation(adj_h*adj_w)

      iv = iv[jdx]
      jv = jv[jdx]
      iv = iv[0]
      jv = jv[0]

      goodimg1 = goodimg1[iv:iv+height, jv:jv+width]
      goodimg2 = goodimg2[iv:iv+height, jv:jv+width]

      goodimg1[goodimg1<0] = 0
      goodimg1[goodimg1>255] = 255
      goodimg2[goodimg2<0] = 0
      goodimg2[goodimg2>255] = 255

      hdf5_im[idx, 0, :, :] = goodimg1
      hdf5_im[idx, 1, :, :] = goodimg2
      hdf5_lab[idx] = 0
      aliasingamt[idx] = factor
      contentidx[idx] = totalcontents[jjj]
      hdf5_traintest[idx] = traintest
      idx += 1
      limit += 1
      if limit >= 100:
        break

  return idx

# get the number of patches
np.random.seed(12345)

# sf2
n_total_images = 249288
patch_height = 100
patch_width = 100
n_channels = 2

# sf = single frame
# fd = frame diff
f = h5py.File('aliasingdataset_sf4.hdf5', mode='w')

image_patches = f.create_dataset('image_patches', (n_total_images, n_channels, patch_height, patch_width), dtype='float')
image_patches.dims[0].label = 'batch'
image_patches.dims[1].label = 'channel'
image_patches.dims[2].label = 'height'
image_patches.dims[3].label = 'width'

labels = f.create_dataset('labels', (n_total_images,), dtype='uint8')
trainset = f.create_dataset('set', (n_total_images,), dtype='uint8')
aliasingamt = f.create_dataset('ratio', (n_total_images,), dtype='float32')
contentidx = f.create_dataset('contentidx', (n_total_images,), dtype='float32')

n_idx = 0
n_idx = get_postrainpatches(image_patches, labels, trainset, aliasingamt, contentidx, n_idx, traintest=0)
n_idx = get_negtrainpatches(image_patches, labels, trainset, aliasingamt, contentidx, n_idx, traintest=0)
n_idx = get_negtrainpatches(image_patches, labels, trainset, aliasingamt, contentidx, n_idx, traintest=1)
n_idx = get_postrainpatches(image_patches, labels, trainset, aliasingamt, contentidx, n_idx, traintest=1)
print n_idx, n_total_images

f.flush()
f.close()
