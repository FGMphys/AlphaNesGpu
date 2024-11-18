#!/home/francegm/miniconda3/envs/fsenv/bin/python3
#IMPORT NN and math libraries
import tensorflow as tf
import argparse
import sys
from tensorflow.keras.layers import Dense
from tensorflow.keras import Input
import numpy as np
import os
import shutil as sh
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
####ISTRUZIONI DA RIGA DI COMANDO
parser = argparse.ArgumentParser()
#########DATASET DIMENSION PARAMETERS##########
parser.add_argument("-imodel", help="model to save")
###OUTPUT PARAMETER
parser.add_argument("-modelname", help="indicate a path/name for the exported model (e.g. folder/folder2/namemodel")

##Build class for graph process execution
class TestModel(tf.Module):
    def __init__(self,alphamu,num_AFs,newmodel):
        super(TestModel, self).__init__()
        self.alphamu = tf.Variable(alphamu)
        self.num_AFs=num_AFs
        self.newmodel=newmodel
    @tf.function()#input_signature=[tf.TensorSpec(name='input1',shape=(1,256,10),dtype=tf.float64)])
    def testmodel(self,des):
        self.des=des
        self.N=tf.shape(self.des)[1]
        self.logdes=tf.math.log(self.des+10**(-3))-self.alphamu
        self.Energies=self.newmodel(self.logdes)
        self.gradEn=tf.reshape(tf.gradients(self.Energies,self.des),shape=(-1,self.N,self.num_AFs))
        self.energy=tf.math.reduce_sum(self.Energies,axis=1)*0.5
        return  self.energy,self.gradEn;


##Read shape parameter for keras model
args = parser.parse_args()
namemodel=args.modelname
os.mkdir(namemodel)
input_model=args.imodel

nt=0
for guess in range(100):
    exist = os.path.exists(input_model+'/net_model_type'+str(guess))
    if exist==True:
       nt=nt+1
    else:
       guess=100
print("Alphanes: detected ",nt," atom species system")


mean=[np.loadtxt(input_model+'/type'+str(k)+'_alpha_mu.dat',dtype='float32') for k in range(nt)]
nAFs=[mean[k].shape[0] for k in range(nt)]

with tf.device('/cpu:0'):
     model=[tf.keras.models.load_model(input_model+'/net_model_type'+str(k)) for k in range(nt)]
newmodel=[tf.keras.Sequential() for k in range(nt)]
res=[newmodel[k].add(Input(shape=(nAFs[k]))) for k in range(nt)]
for num,typemodel in enumerate(newmodel):
    for el in model[num].layers:
        typemodel.add(el)



##Call Class to build the graph
toexport=[TestModel(mean[k],nAFs[k],newmodel[k]) for k in range(nt)]
#out=[toexport[k].testmodel(tf.zeros((tipos[k],nAFs[k]),dtype='float64')) for k in range(nt)]
call = [toexport[k].testmodel.get_concrete_function(tf.TensorSpec([None,None,nAFs[k]], tf.float32)) for k in range(nt)]
##Compile a models


###Save the model
sh.copy(input_model+'/model_error',namemodel)
for k in range(nt):
    sh.copy(input_model+'/type'+str(k)+'_alpha_2body.dat',namemodel)
    sh.copy(input_model+'/type'+str(k)+'_alpha_3body.dat',namemodel)
    tf.saved_model.save(toexport[k], namemodel+'/model_type'+str(k),signatures=call[k])
    if nt>1:
        sh.copy(input_model+'/type'+str(k)+'_type_emb_2b_sq.dat',namemodel)
        sh.copy(input_model+'/type'+str(k)+'_type_emb_3b_sq.dat',namemodel)
        sh.copy(input_model+'/type'+str(k)+'_type_emb_2b.dat',namemodel)
        sh.copy(input_model+'/type'+str(k)+'_type_emb_3b.dat',namemodel)
