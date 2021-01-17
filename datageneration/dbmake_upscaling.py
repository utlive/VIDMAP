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
    jdx = jdx[:1000]

    iv = iv[jdx]
    jv = jv[jdx]
    tv = np.arange(0, T-1)

    limit1 = 0
    limit2 = 0
    for (y, x) in zip(iv, jv):
      np.random.shuffle(tv)
      t = tv[0]
      utype = np.array([cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4])
      np.random.shuffle(utype)

      mmethod = np.array([0, 1])
      np.random.shuffle(mmethod)

      fctr = np.random.random()*4.75 + 1.25

      if mmethod[0] == 0:
        if limit1 >= 25:
          continue
        slc = vid_pris[t:t+2, :, :, 0].astype(np.float32)
        # take the whole frame, shrink using lanczos4, expand it using an upscaling type
        print (np.int32(slc[0].shape[1]/fctr), np.int32(slc[0].shape[0]/fctr))
        tmp1 = cv2.resize(slc[0], (np.int32(slc[0].shape[1]/fctr), np.int32(slc[0].shape[0]/fctr)), interpolation=cv2.INTER_LANCZOS4)
        tmp1[tmp1<0] = 0
        tmp1[tmp1>255] = 255
        tmp1 = tmp1.astype(np.uint8)
        tmp1 = tmp1.astype(np.float32)
        tmp1 = cv2.resize(tmp1, (np.int32(slc[0].shape[1]), np.int32(slc[0].shape[0])), interpolation=utype[0])
        tmp1[tmp1<0] = 0
        tmp1[tmp1>255] = 255
        tmp1 = tmp1.astype(np.uint8)
        tmp1 = tmp1.astype(np.float32)

        tmp2 = cv2.resize(slc[1], (np.int32(slc[1].shape[1]/fctr), np.int32(slc[1].shape[0]/fctr)), interpolation=cv2.INTER_LANCZOS4)
        tmp2[tmp2<0] = 0
        tmp2[tmp2>255] = 255
        tmp2 = tmp2.astype(np.uint8)
        tmp2 = tmp2.astype(np.float32)
        tmp2 = cv2.resize(tmp2, (np.int32(slc[1].shape[1]), np.int32(slc[1].shape[0])), interpolation=utype[0])
        tmp2[tmp2<0] = 0
        tmp2[tmp2>255] = 255
        tmp2 = tmp2.astype(np.uint8)
        tmp2 = tmp2.astype(np.float32)

        # now use selection criteria
        tmp1 = tmp1[y:y+256, x:x+256]
        tmp2 = tmp2[y:y+256, x:x+256]

        if np.std(tmp1) < 20:
          continue
        if np.std(tmp2) < 20:
          continue

        #print diff
        #skimage.io.imsave("extract/test_%d.png" % (idx,), np.hstack((goodpatch[0].astype(np.uint8), badpatch[0].astype(np.uint8))))

        #preprocess = preprocess[:, 5:-5, 5:-5]
        hdf5_im[idx, 0] = tmp1
        hdf5_im[idx, 1] = tmp2
        hdf5_lab[idx] = 1
        hdf5_trainset[idx] = traintest
        idx += 1
        limit1 += 1
      else:
        if limit2 >= 25:
          continue

        slc = vid_pris[t:t+2, :, :, 0].astype(np.float32)
        # take the whole frame, shrink using lanczos4, expand it using an upscaling type
        tmp1 = cv2.resize(slc[0], (np.int32(slc[0].shape[1]*fctr), np.int32(slc[0].shape[0]*fctr)), interpolation=utype[0])
        tmp1[tmp1<0] = 0
        tmp1[tmp1>255] = 255
        tmp1 = tmp1.astype(np.uint8)
        tmp1 = tmp1.astype(np.float32)

        tmp2 = cv2.resize(slc[1], (np.int32(slc[1].shape[1]*fctr), np.int32(slc[1].shape[0]*fctr)), interpolation=utype[0])
        tmp2[tmp2<0] = 0
        tmp2[tmp2>255] = 255
        tmp2 = tmp2.astype(np.uint8)
        tmp2 = tmp2.astype(np.float32)

        H2, W2 = tmp1.shape

        adj_h2 = H2 - 256
        adj_w2 = W2 - 256
        iv2, jv2 = np.meshgrid(np.arange(adj_h2), np.arange(adj_w2), sparse=False, indexing='ij')
        iv2 = iv2.reshape(-1)
        jv2 = jv2.reshape(-1)

        jdx2 = np.random.permutation(adj_h2*adj_w2)

        iv2 = iv2[jdx2]
        jv2 = jv2[jdx2]

        tmp1 = tmp1[iv2[0]:iv2[0]+256, jv2[0]:jv2[0]+256]
        tmp2 = tmp2[iv2[0]:iv2[0]+256, jv2[0]:jv2[0]+256]
        if np.std(tmp1) < 20:
          continue
        if np.std(tmp2) < 20:
          continue

        hdf5_im[idx, 0] = tmp1
        hdf5_im[idx, 1] = tmp2
        hdf5_lab[idx] = 1
        hdf5_trainset[idx] = traintest
        idx += 1
        limit2 += 1

      #skimage.io.imsave("extract/%d.png" % (idx,), patch)
      if (limit1 >= 25) and (limit2 >= 25):
        break
    print limit1, limit2, idx

  return idx

def get_negtrainpatches(image_patches, labels, trainset, idx=0, traintest=0):
  return genericnegpatcher(image_patches, labels, trainset, idx=idx, traintest=traintest)

def genericnegpatcher(hdf5_im, hdf5_lab, hdf5_trainset, idx=0, traintest=0):
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
    jdx = jdx[:1000]

    iv = iv[jdx]
    jv = jv[jdx]
    tv = np.arange(0, T-1)

    limit1 = 0
    limit2 = 0
    for (y, x) in zip(iv, jv):
      np.random.shuffle(tv)
      t = tv[0]

      # if 1080p, we can randomly decide to downsample first
      mmethod = np.array([0, 1])
      np.random.shuffle(mmethod)

      fctr = np.random.random()*4.75 + 1.25
      if mmethod[0] == 0:
        if limit1 >= 25:
          continue
        slc = vid_pris[t:t+2, :, :, 0]
        tmp1 = cv2.resize(slc[0], (np.int32(slc[1].shape[1]/fctr), np.int32(slc[0].shape[0]/fctr)), interpolation=cv2.INTER_LANCZOS4)
        tmp1[tmp1<0] = 0
        tmp1[tmp1>255] = 255
        tmp1 = tmp1.astype(np.uint8)
        tmp1 = tmp1.astype(np.float32)
        tmp2 = cv2.resize(slc[1], (np.int32(slc[1].shape[1]/fctr), np.int32(slc[0].shape[0]/fctr)), interpolation=cv2.INTER_LANCZOS4)
        tmp2[tmp2<0] = 0
        tmp2[tmp2>255] = 255
        tmp2 = tmp2.astype(np.uint8)
        tmp2 = tmp2.astype(np.float32)
        if (tmp1.shape[0] < 256+1) or (tmp1.shape[1] < 256+1):
          continue
        H2, W2 = tmp1.shape

        adj_h2 = H2 - 256
        adj_w2 = W2 - 256
        iv2, jv2 = np.meshgrid(np.arange(adj_h2), np.arange(adj_w2), sparse=False, indexing='ij')
        iv2 = iv2.reshape(-1)
        jv2 = jv2.reshape(-1)

        jdx2 = np.random.permutation(adj_h2*adj_w2)

        iv2 = iv2[jdx2]
        jv2 = jv2[jdx2]

        tmp1 = tmp1[iv2[0]:iv2[0]+256, jv2[0]:jv2[0]+256]
        tmp2 = tmp2[iv2[0]:iv2[0]+256, jv2[0]:jv2[0]+256]

        #if np.std(tmp1) < 20:
        #  continue
        #if np.std(tmp2) < 20:
        #  continue

        hdf5_im[idx, 0] = tmp1
        hdf5_im[idx, 1] = tmp2
        hdf5_lab[idx] = 0
        hdf5_trainset[idx] = traintest

        idx += 1
        limit1 += 1
      else:
        if limit2 >= 25:
          continue
        # randomly downsample with Lanczos-4
        goodpatch = vid_pris[t:t+2, y:y+height, x:x+width, 0]

        #preprocess = preprocess[:, 5:-5, 5:-5]
        #if np.std(goodpatch[0]) < 20:
        #  continue
        #if np.std(goodpatch[1]) < 20:
        #  continue

        hdf5_im[idx] = goodpatch
        hdf5_lab[idx] = 0
        hdf5_trainset[idx] = traintest

      #skimage.io.imsave("extract/%d.png" % (idx,), patch)
        idx += 1
        limit2 += 1
      if (limit2 >= 25) and (limit1 >= 25):
        break
    print limit1, limit2, idx

  return idx

# get the number of patches
np.random.seed(12345)

n_total_images = 131000#12000
patch_height = 256
patch_width = 256
n_channels = 2

# sf = single frame
# fd = frame diff
f = h5py.File('/mnt/hd2/upscalingdataset_sf.hdf5', mode='w')

image_patches = f.create_dataset('image_patches', (n_total_images, n_channels, patch_height, patch_width), dtype='float')
image_patches.dims[0].label = 'batch'
image_patches.dims[1].label = 'channel'
image_patches.dims[2].label = 'height'
image_patches.dims[3].label = 'width'

labels = f.create_dataset('labels', (n_total_images,), dtype='uint8')
trainset = f.create_dataset('set', (n_total_images,), dtype='uint8')

n_idx = 0
n_idx = get_negtrainpatches(image_patches, labels, trainset, n_idx, 0)
n_idx = get_postrainpatches(image_patches, labels, trainset, n_idx, 0)
n_idx = get_postrainpatches(image_patches, labels, trainset, n_idx, 1)
n_idx = get_negtrainpatches(image_patches, labels, trainset, n_idx, 1)

print n_idx, n_total_images

f.flush()
f.close()
