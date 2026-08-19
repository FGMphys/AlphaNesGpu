#!/usr/bin/env python3
"""Export a double-precision trained STAF-CG checkpoint as a float32 inference model.

Usage:
  python save_model_in_float.py -imodel /path/to/model_logN -modelname /path/to/out_float
"""
import argparse
import os
import shutil as sh

import numpy as np
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

parser = argparse.ArgumentParser()
parser.add_argument("-imodel", help="model to save", required=True)
parser.add_argument("-modelname", help="path/name for the exported float32 model", required=True)

tf.keras.backend.set_floatx("float32")


class TestModel(tf.Module):
    def __init__(self, alphamu, num_AFs, newmodel):
        super(TestModel, self).__init__()
        self.alphamu = tf.Variable(alphamu, dtype=tf.float32)
        self.num_AFs = num_AFs
        self.newmodel = newmodel

    @tf.function()
    def testmodel(self, des):
        self.des = des
        self.N = tf.shape(self.des)[1]
        self.logdes = tf.math.log(self.des + 10 ** (-3)) - self.alphamu
        self.Energies = self.newmodel(self.logdes)
        self.gradEn = tf.reshape(
            tf.gradients(self.Energies, self.des), shape=(-1, self.N, self.num_AFs)
        )
        self.energy = tf.math.reduce_sum(self.Energies, axis=1) * 0.5
        return tf.cast(self.energy, dtype="float32"), self.gradEn


args = parser.parse_args()
namemodel = args.modelname
os.mkdir(namemodel)
input_model = args.imodel

nt = 0
for guess in range(100):
    if os.path.exists(input_model + "/net_model_type" + str(guess)):
        nt = nt + 1
    else:
        break
print("STAF-CG: detected ", nt, " NN")

mean = [
    np.loadtxt(input_model + "/type" + str(k) + "_alpha_mu.dat", dtype="float32")
    for k in range(nt)
]
nAFs = [mean[k].shape[0] for k in range(nt)]

with tf.device("/cpu:0"):
    model = [
        tf.keras.models.load_model(input_model + "/net_model_type" + str(k))
        for k in range(nt)
    ]

tf.keras.backend.set_floatx("float32")
newmodel = [tf.keras.Sequential() for k in range(nt)]
for k in range(nt):
    newmodel[k].add(tf.keras.Input(shape=(nAFs[k],), dtype=tf.float32))
for num, typemodel in enumerate(newmodel):
    for el in model[num].layers:
        if isinstance(el, tf.keras.layers.InputLayer):
            continue
        cfg = el.get_config()
        if "dtype" in cfg:
            cfg["dtype"] = "float32"
        new_el = el.__class__.from_config(cfg)
        typemodel.add(new_el)
        weights = [np.asarray(w, dtype=np.float32) for w in el.get_weights()]
        if weights:
            new_el.set_weights(weights)

toexport = [TestModel(mean[k], nAFs[k], newmodel[k]) for k in range(nt)]
call = [
    toexport[k].testmodel.get_concrete_function(
        tf.TensorSpec([None, None, nAFs[k]], tf.float32)
    )
    for k in range(nt)
]

for aux in (
    "model_error",
    "cutoff_info",
    "number_of_nn.dat",
    "color_type_map.dat",
    "map_color_interaction.dat",
    "map_intra.dat",
):
    src = os.path.join(input_model, aux)
    if os.path.exists(src):
        sh.copy(src, namemodel)

for k in range(nt):
    sh.copy(input_model + "/type" + str(k) + "_alpha_2body.dat", namemodel)
    sh.copy(input_model + "/type" + str(k) + "_alpha_3body.dat", namemodel)
    tf.saved_model.save(toexport[k], namemodel + "/model_type" + str(k), signatures=call[k])
    for emb in (
        "/type" + str(k) + "_type_emb_2b_sq.dat",
        "/type" + str(k) + "_type_emb_3b_sq.dat",
        "/type" + str(k) + "_type_emb_2b.dat",
        "/type" + str(k) + "_type_emb_3b.dat",
    ):
        src = input_model + emb
        if os.path.exists(src):
            sh.copy(src, namemodel)

with open(os.path.join(namemodel, "number_of_nn.dat"), "w") as fh:
    fh.write(str(nt) + "\n")

print("STAF-CG: float32 inference model written to", namemodel)
