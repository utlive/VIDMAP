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

def get_postrainpatches(hdf5_im, hdf5_lab, hdf5_trainset, offsets, idx=0, traintest=0):
  return genericpospatcher(hdf5_im, hdf5_lab, hdf5_trainset, offsets, idx=idx, traintest=traintest)

def genericpospatcher(hdf5_im, hdf5_lab, hdf5_trainset, offsets, idx=0, traintest=0):
  width = 256
  height = 256

  lst480p = np.array(glob.glob("/mnt/hd3/scenes/480p/*avi"))
  lst1080p = np.array(glob.glob("/mnt/hd3/scenes/1080p/*avi"))
  lst480p = np.sort(lst480p)
  lst1080p = np.sort(lst1080p)

  if traintest == 0:
    lst = np.hstack((lst480p[:575], lst1080p[:215]))
  else:
    lst = np.hstack((lst480p[575:], lst1080p[215:]))

  # go through list twice
  for repeates  in [1, 2]:
    n_samples = len(lst)
    for fname in lst:
      print fname
      runOnce = True
      while runOnce:
        #vid = skvideo.io.vread(fname, as_grey=True).astype(np.float32)
        try:
          cmd = "ffmpeg -y -i %s -codec:v mpeg2video -g 50 -mpv_flags +strict_gop -bsf noise=100000 -b:v 40000k /tmp/test_distorted.mp4" % (fname,)
          os.system(cmd)

          cmd = "ffmpeg -y -ec 0 -i /tmp/test_distorted.mp4 -vcodec rawvideo -pix_fmt yuv420p /tmp/test_distorted.avi"
          os.system(cmd)

          cmd = "ffmpeg -y -i %s -codec:v mpeg2video -b:v 40000k /tmp/test_pristine.mp4" % (fname,)
          os.system(cmd)

          cmd = "ffmpeg -y -ec 0 -i /tmp/test_pristine.mp4 -vcodec rawvideo -pix_fmt yuv420p /tmp/test_pristine.avi"
          os.system(cmd)
          runOnce = False
        except:
          pass

      vid_dis = skvideo.io.vread("/tmp/test_distorted.avi", as_grey=True).astype(np.float32)
      vid_pris = skvideo.io.vread("/tmp/test_pristine.avi", as_grey=True).astype(np.float32)

      os.remove("/tmp/test_distorted.mp4")
      os.remove("/tmp/test_pristine.mp4")
      os.remove("/tmp/test_distorted.avi")
      os.remove("/tmp/test_pristine.avi")

      T, H, W, C = vid_dis.shape

      adj_h = H - height
      adj_w = W - width
      iv, jv = np.meshgrid(np.arange(adj_h), np.arange(adj_w), sparse=False, indexing='ij')
      iv = iv.reshape(-1)
      jv = jv.reshape(-1)

      jdx = np.random.permutation(adj_h*adj_w)

      iv = iv[jdx]
      jv = jv[jdx]
      tv = np.arange(1, T-1)

      limit = 0
      for (y, x) in zip(iv, jv):
        np.random.shuffle(tv)
        t = tv[0]
        goodpatch = vid_pris[t-1:t+2, y:y+height, x:x+width, 0]
        badpatch = vid_dis[t-1:t+2, y:y+height, x:x+width, 0]

        # difference the magntudes, so we don't worry about phase shifts
        if badpatch.shape[0] == goodpatch.shape[0]:
          diff = np.mean(np.abs(badpatch[1, 30:-30, 30:-30] - goodpatch[1, 30:-30, 30:-30])**2)
          if diff < 50:
            continue
          error1 = np.sum((goodpatch[0] - badpatch[0])**2)
          error2 = np.sum((goodpatch[2] - badpatch[2])**2)
        else:
          error1 = np.sum((goodpatch[0] - badpatch[0])**2)
          error2 = np.sum((goodpatch[2] - badpatch[2])**2)
          continue

        # check for no variance img
        if np.std(badpatch[0, 60:-60, 60:-60]) < 10:
          continue
        if np.std(badpatch[1, 60:-60, 60:-60]) < 10:
          continue
        if np.std(badpatch[2, 60:-60, 60:-60]) < 10:
          continue

        #preprocess = preprocess[:, 5:-5, 5:-5]
        #preprocess = preprocess[20:-20, 20:-20]

        hdf5_im[idx] = badpatch
        hdf5_lab[idx] = 1
        hdf5_trainset[idx] = traintest
        offsets[idx] = [y, x]

        #skimage.io.imsave("extract/%d.png" % (idx,), patch)
        limit += 1
        idx += 1
        if limit >= 10:
          break

  return idx

def get_negtrainpatches(image_patches, labels, hdf5_traintest, offsets, idx=0, traintest=0):
  return genericnegpatcher(image_patches, labels, hdf5_traintest, offsets, idx=idx, traintest=traintest)

def genericnegpatcher(hdf5_im, hdf5_lab, hdf5_traintest, offsets, idx=0, traintest=0):

  width = 256
  height = 256

  lst480p = np.array(glob.glob("/mnt/hd3/scenes/480p/*avi"))
  lst1080p = np.array(glob.glob("/mnt/hd3/scenes/1080p/*avi"))
  lst480p = np.sort(lst480p)
  lst1080p = np.sort(lst1080p)

  if traintest == 0:
    lst = np.hstack((lst480p[:575], lst1080p[:215]))
  else:
    lst = np.hstack((lst480p[575:], lst1080p[215:]))

  n_samples = len(lst)
  for fname in lst:
    #vid = skvideo.io.vread(fname, as_grey=True).astype(np.float32)
    cmd = "ffmpeg -y -i %s -codec:v mpeg2video -b:v 40000k /tmp/test_pristine.mp4" % (fname,)
    os.system(cmd)

    cmd = "ffmpeg -y -ec 0 -i /tmp/test_pristine.mp4 -vcodec rawvideo -pix_fmt yuv420p /tmp/test_pristine.avi"
    os.system(cmd)
    vid_pris = skvideo.io.vread("/tmp/test_pristine.avi", as_grey=True).astype(np.float32)
    os.remove("/tmp/test_pristine.mp4")
    os.remove("/tmp/test_pristine.avi")

    T, H, W, C = vid_pris.shape

    adj_h = H - height
    adj_w = W - width
    iv, jv = np.meshgrid(np.arange(adj_h), np.arange(adj_w), sparse=False, indexing='ij')
    iv = iv.reshape(-1)
    jv = jv.reshape(-1)

    jdx = np.random.permutation(adj_h*adj_w)

    iv = iv[jdx]
    jv = jv[jdx]
    tv = np.arange(1, T-1)

    limit = 0
    for (y, x) in zip(iv, jv):
      np.random.shuffle(tv)
      t = tv[0]
      goodpatch = vid_pris[t-1:t+2, y:y+height, x:x+width, 0]

      #print diff
      #skimage.io.imsave("/tmp/test_%d.png" % (limit,), np.hstack((goodpatch.astype(np.uint8), badpatch.astype(np.uint8))))

      #preprocess = preprocess[:, 5:-5, 5:-5]
      #preprocess = preprocess[20:-20, 20:-20]

      hdf5_im[idx] = goodpatch
      hdf5_lab[idx] = 0
      hdf5_traintest[idx] = traintest
      offsets[idx] = [y, x]

      #skimage.io.imsave("extract/%d.png" % (idx,), patch)
      limit += 1
      idx += 1
      if limit >= 20:
        break

  return idx

# get the number of patches
np.random.seed(12345)


n_total_images = 59941#43000*2 #12000
patch_height = 256
patch_width = 256
n_channels = 3

# sf = single frame
# fd = frame diff
f = h5py.File('/mnt/hd2/hitsdataset_sf_mpeg2.hdf5', mode='w')

image_patches = f.create_dataset('image_patches', (n_total_images, n_channels, patch_height, patch_width), dtype='float')
image_patches.dims[0].label = 'batch'
image_patches.dims[1].label = 'channel'
image_patches.dims[2].label = 'height'
image_patches.dims[3].label = 'width'

labels = f.create_dataset('labels', (n_total_images,), dtype='uint8')
trainset = f.create_dataset('set', (n_total_images,), dtype='uint8')
offsets = f.create_dataset('offsets', (n_total_images, 2), dtype='int32')

n_idx = 0
n_idx = get_postrainpatches(image_patches, labels, trainset, offsets, n_idx, traintest=0)
n_idx = get_negtrainpatches(image_patches, labels, trainset, offsets, n_idx, traintest=0)
n_idx = get_postrainpatches(image_patches, labels, trainset, offsets, n_idx, traintest=1)
n_idx = get_negtrainpatches(image_patches, labels, trainset, offsets, n_idx, traintest=1)
print n_idx, n_total_images

f.flush()
f.close()
