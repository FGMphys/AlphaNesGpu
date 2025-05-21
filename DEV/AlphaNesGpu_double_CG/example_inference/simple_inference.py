
import numpy as np
import tensorflow as tf
import os
import sys
abspath=os.path.abspath('/home/francegm/AlphaNesGpu_local')
sys.path.append(abspath)
from alphanes_models.mixture.alpha_nes_model_inference_full import alpha_nes_full_inference

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

Pos=np.loadtxt("pos_0",dtype='float64').reshape((1,-1*3))#np.random.randn(768*3).reshape((1,-1,3))
Box=np.loadtxt("box_0",dtype='float64').reshape((1,6))#np.zeros(6).reshape((1,6))

output=Model.full_test(Pos,Box)
output2=Model.full_test(Pos,Box)
breakpoint()
