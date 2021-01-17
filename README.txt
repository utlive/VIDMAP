The Video Impairment Detection Mapper (VIDMAP)

========================================================================

-----------COPYRIGHT NOTICE STARTS WITH THIS LINE------------
Copyright (c) 2018 The University of Texas at Austin
All rights reserved.

Permission is hereby granted, without written agreement and without license or royalty fees, to use, copy, 
modify, and distribute this code (the source files) and its documentation for
any purpose, provided that the copyright notice in its entirety appear in all copies of this code, and the 
original source of this code, Laboratory for Image and Video Engineering (LIVE, http://live.ece.utexas.edu)
at the University of Texas at Austin (UT Austin, http://www.utexas.edu), is acknowledged in any publication 
that reports research using this code. The research is to be cited in the bibliography as:

1) T. Goodall and A. C. Bovik, "VIDMAP Software Release", 
URL: http://live.ece.utexas.edu/research/quality/VIDMAP_release.zip, 2018

2) T. Goodall and A. C. Bovik, "Detecting and Mapping Video Impairments" 2018
submitted

3) T. Goodall and A. C. Bovik, "Artifact Detection Maps Learned using Shallow 
Convolutional Networks" SSIAI 2017

IN NO EVENT SHALL THE UNIVERSITY OF TEXAS AT AUSTIN BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, 
OR CONSEQUENTIAL DAMAGES ARISING OUT OF THE USE OF THIS DATABASE AND ITS DOCUMENTATION, EVEN IF THE UNIVERSITY OF TEXAS
AT AUSTIN HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

THE UNIVERSITY OF TEXAS AT AUSTIN SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE DATABASE PROVIDED HEREUNDER IS ON AN "AS IS" BASIS,
AND THE UNIVERSITY OF TEXAS AT AUSTIN HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.

-----------COPYRIGHT NOTICE ENDS WITH THIS LINE------------%

Author  : Todd Goodall
Version : 1.1

The authors are with the Laboratory for Image and Video Engineering
(LIVE), Department of Electrical and Computer Engineering, The
University of Texas at Austin, Austin, TX.

Kindly report any suggestions or corrections to tgoodall@utexas.edu

========================================================================

This is a demonstration of the the Video Impairment Detection Mapper (VIDMAP).
The algorithm is described in:

T. Goodall and A. C. Bovik, "Detecting and Mapping Video Impairments"

You can change this program as you like and use it anywhere, but please
refer to its original source (cite our paper and our web page at
http://live.ece.utexas.edu/research/quality/VIDMAP_release.zip).

========================================================================

VIDMAP is a state-of-the-art detector of many video distortion types and is 
capable of detecting and localizing artifacts without a reference video. 
The same network architecture has been tested with each of the following
distortion types: 
    - Upscaling (Bilinear, Bicubic, Lanczos-4, and Nearest Neighbor)
    - Video Hits (H.264)
    - Video Hits (MPEG2)
    - "False Contours"/Banding
    - H.264 Compression
    - Aliasing
    - Interlacing/Combing
    - Quantization
    - Dropped Frames

Trained weights are included in this archive for full VIDMAP, VIDMAP (2 layer),
and VIDMAP-D (VIDMAP configured to use frame-difference input).

Archive contents and brief description:
    - VIDMAP_train.py         VIDMAP code for training models using
                              generated the hdf5 databases. 

    - VIDMAP_test.py          VIDMAP code for testing trained models 
                              against the hdf5 databases. 

    - VIDMAP_test_whole.py    VIDMAP code for testing trained models 
                              on images and video against the database. 
                              This script outputs a probability of prediction
                              per frame along with a video of predicted frames.

    - single/                 Pretrained data files that correspond to 
                              full VIDMAP version with 3 layers that works
                              on single frames.

    - framediff/              Pretrained data files that correspond to 
                              full VIDMAP version with 3 layers that works
                              on frame differences.

    - 2layer/                 Pretrained data files that correspond to 
                              reduced VIDMAP version with 2 layers that works
                              on single frames (also includes droppedFrames, which
                              required frame differences).

    - datageneration/         Data generation scripts used in VIDMAP paper.

By running each python file without arguments, you will see a printout of 
available options.

Running VIDMAP requires python and the following packages:
    - tensorflow >= 1.10
    - numpy
    - sklearn (for f1 score and mcc metrics)
    - scikit-video
    - h5py (only if you use the generation scripts) 
