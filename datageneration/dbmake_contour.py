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
avg_window = gauss_window(100, 30.0)

def hp_image(image, window_arr):
    extend_mode = 'reflect'
    image = np.array(image).astype(np.float32)
    w, h = image.shape
    mu_image = np.zeros((w, h))
    scipy.ndimage.correlate1d(image, window_arr, 0, mu_image, mode=extend_mode) 
    scipy.ndimage.correlate1d(mu_image, window_arr, 1, mu_image, mode=extend_mode) 
    return image - mu_image, mu_image

def get_postrainpatches(hdf5_im, hdf5_lab, hdf5_trainset, idx=0, traintest=0):
  return genericpospatcher(hdf5_im, hdf5_lab, hdf5_trainset, idx=idx, traintest=traintest)

def genericpospatcher(hdf5_im, hdf5_lab, hdf5_trainset, idx=0, traintest=0):
  width = 100+40
  height = 100+40

  #lst480p = np.array(glob.glob("/mnt/hd3/scenes/480p/*avi"))
  lst = np.array(glob.glob("images/*"))
  #lst480p = np.sort(lst480p)

  n_samples = 1000
  for jjj in range(n_samples):
    print jjj, n_samples
    #vid_pris = skvideo.io.vread(fname, as_grey=True).astype(np.float32)
    vid_pris = np.random.random(size=(1, 500, 500, 1))*255
    mi = np.min(vid_pris)
    ma = np.max(vid_pris)

    randomwidth=np.random.random()*20 + 20
    avg_window = gauss_window(np.int32(np.round(randomwidth)*3), randomwidth)
    _, blur = hp_image(vid_pris[0, :, :, 0], avg_window)
    blur -= np.min(blur)
    blur /= np.max(blur)
    blur *= (ma - mi)
    blur += mi


    # film grain noise
    blur = blur.astype(np.uint8)

    vid_pris[0, :, :, 0] = blur

    T, H, W, C = vid_pris.shape


    adj_h = H - height
    adj_w = W - width
    iv, jv = np.meshgrid(np.arange(adj_h), np.arange(adj_w), sparse=False, indexing='ij')
    iv = iv.reshape(-1)
    jv = jv.reshape(-1)

    jdx = np.random.permutation(adj_h*adj_w)

    iv = iv[jdx]
    jv = jv[jdx]

    rpy = 0
    limit = 0
    for (y, x) in zip(iv, jv):
      rpy += 1
      t = 0
      goodpatch = vid_pris[0, y:y+height, x:x+width, 0]
      badpatch = goodpatch.copy()
      badpatch = badpatch.astype(np.float32)
      badpatch2 = badpatch.copy()
      A = np.random.normal(size=badpatch.shape)
      B = np.random.random(1)*10
      if B>5:
        B -= 5
      else:
        B *= 0
      badpatch += A*B#np.random.normal(size=vid_pris.shape) 
      badpatch[badpatch<0] = 0
      badpatch[badpatch>255] = 255

      # random amount of change
      amt = np.random.randint(3, 6)
      badpatch /= 2**amt
      badpatch = np.floor(badpatch)
      badpatch *= 2**amt

      badpatch2 /= 2**amt
      badpatch2 = np.floor(badpatch2)
      badpatch2 *= 2**amt

      diff = np.mean((badpatch - goodpatch)**2)
      if diff < 1.5:
        continue
      #print diff
      #skimage.io.imsave("extract/test_%d.png" % (idx,), badpatch.astype(np.uint8))
      #exit(0)

      # make sure there is some non-zero variance in the center of the patch
      if(np.std(badpatch2[55:-55, 55:-55])<1e-9):
        print "bad patch"
        continue

      #preprocess = preprocess[:, 5:-5, 5:-5]
      badpatch = badpatch[20:-20, 20:-20]

      hdf5_im[idx] = badpatch
      hdf5_lab[idx] = 1
      hdf5_trainset[idx] = traintest

      #skimage.io.imsave("extract/%d.png" % (idx,), patch)
      limit += 1
      idx += 1
      if limit >= 200:
        break

  return idx

def get_negtrainpatches(image_patches, labels, trainset, idx=0, traintest=0):
  return genericnegpatcher(image_patches, labels, trainset, idx=idx, traintest=traintest)

def genericnegpatcher(hdf5_im, hdf5_lab, hdf5_trainset, idx=0, traintest=0):
  width = 100+40
  height = 100+40

  lst = np.array(glob.glob("images/*"))
  #lst480p = np.sort(lst480p)
  tlst = []
  for i in xrange(1000/10):
    tlst = np.hstack((tlst, lst[:10]))
  lst = tlst

  lst480p = np.array(glob.glob("/mnt/hd3/scenes/480p/*avi"))
  lst1080p = np.array(glob.glob("/mnt/hd3/scenes/1080p/*avi"))
  lst480p = np.sort(lst480p)
  lst1080p = np.sort(lst1080p)

  if traintest == 0:
    lst = np.hstack((lst, lst480p[:575], lst1080p[:215]))
  else:
    lst = np.hstack((lst, lst480p[575:], lst1080p[215:]))
    #lst = np.hstack((lst, lst480p[57:114], lst1080p[21:42]))

  n_samples = len(lst)
  for jjj, fname in enumerate(lst):
    print jjj, n_samples
    if "images" in fname:
      print "gen"
      vid_pris = np.random.random(size=(1, 500, 500, 1))*255
      mi = np.min(vid_pris)
      ma = np.max(vid_pris)

      randomwidth=np.random.random()*20 + 10
      avg_window = gauss_window(np.int32(np.round(randomwidth)*3), randomwidth)
      _, blur = hp_image(vid_pris[0, :, :, 0], avg_window)
      blur -= np.min(blur)
      blur /= np.max(blur)
      blur *= (ma - mi)
      blur += mi

      blur = blur.astype(np.uint8)

      vid_pris[0, :, :, 0] = blur
      vid_pris = vid_pris.astype(np.float32)
    else:
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
    limit = 0
    rby = 0
    tv = np.arange(T)
    for (y, x) in zip(iv, jv):
      np.random.shuffle(tv)
      t = tv[0]
      goodpatch = vid_pris[0, y:y+height, x:x+width, 0].astype(np.float32)

      #preprocess = preprocess[:, 5:-5, 5:-5]
      goodpatch = goodpatch[20:-20, 20:-20]

      if "images" in fname:
        A = np.random.normal(size=goodpatch.shape)
        B = np.random.random(1)*10
        if B>5:
          B -= 5
        else:
          B *= 0
        goodpatch += A*B#np.random.normal(size=vid_pris.shape) 
        goodpatch[goodpatch<0] = 0
        goodpatch[goodpatch>255] = 255

      hdf5_im[idx] = goodpatch
      hdf5_lab[idx] = 0
      hdf5_trainset[idx] = traintest

      #skimage.io.imsave("extract/%d.png" % (idx,), patch)
      limit += 1
      idx += 1
      if limit >= 100:
        break

  return idx

# get the number of patches
np.random.seed(12345)

n_total_images = 730600 #12000
patch_height = 100
patch_width = 100
n_channels = 1

# sf = single frame
# fd = frame diff
f = h5py.File('contourdataset_sf.hdf5', mode='w')

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
