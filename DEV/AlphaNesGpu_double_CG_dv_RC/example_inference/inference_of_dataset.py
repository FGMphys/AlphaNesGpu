
import numpy as np
import tensorflow as tf
import os
import sys
abspath=os.path.abspath('PUT HERE ABSOLUTE PATH TO PROGRAM FOLDER')
sys.path.append(abspath)
from alphanes_models.mixture.alpha_nes_model_inference import alpha_nes_full_inference

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)


Model=alpha_nes_full_inference(sys.argv[1])

Pos=np.load("dataset_281K/dataset/training/pos.npy")
nf=Pos.shape[0]
Pos=Pos.reshape((nf,-1*3)) #np.random.randn(768*3).reshape((1,-1,3))
Box=np.load("dataset_281K/dataset/training/box.npy").reshape((-1,6))
batch_size=40
num_batch=int(np.ceil(nf/batch_size))
print(nf,num_batch,num_batch*batch_size)
force_list=[]
for k in range(num_batch):
    pos_actual=Pos[k*batch_size:(k+1)*batch_size]
    box_actual=Box[k*batch_size:(k+1)*batch_size]
    fr_actual=pos_actual.shape[0]
    output=Model.full_test(pos_actual.reshape((fr_actual,-1)),box_actual.reshape((fr_actual,-1)))
    force_list.append(output[1])
    print("DONE ",k*batch_size," over ",nf)
print(force_list)
np.save("force_training.npy",np.concatenate(force_list,axis=0))
