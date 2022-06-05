########## A.L.P.H.A. ########
#
# Author: Julián Moreno
# Date: 05/06/22
# Description: 
# This programm is an installation usign gpt-3 and tensorflow lite, creates poetry based 
# on what it sees. 
# 
# This code is based off the TensorFlow Lite image classification example at:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py
#

# Import packages
import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util
