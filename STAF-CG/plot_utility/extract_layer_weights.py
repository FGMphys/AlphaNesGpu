#!/home/francegm/miniconda3/envs/fsenv/bin/python3
#IMPORT NN and math libraries
import tensorflow as tf
import argparse
#import sys
#from tensorflow.keras.layers import Dense
#from tensorflow.keras import Input
import glob as gl
import numpy as np
import os
#import shutil as sh
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
####ISTRUZIONI DA RIGA DI COMANDO
parser = argparse.ArgumentParser()
#########DATASET DIMENSION PARAMETERS##########
parser.add_argument("-imodel", help="model to save")

##Read shape parameter for keras model
args = parser.parse_args()
input_model=args.imodel



with tf.device('/cpu:0'):
     model=tf.keras.models.load_model(input_model)
count=0
for layer in model.layers:
    cc=0
    for ll in layer.weights:
        np.savetxt("param_"+str(cc)+"_layer_"+str(count),ll.numpy())
        cc=cc+1
    count=count+1

