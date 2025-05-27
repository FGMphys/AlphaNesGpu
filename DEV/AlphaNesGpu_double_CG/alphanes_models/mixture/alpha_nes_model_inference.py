import tensorflow as tf
import numpy as np
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense

from source_routine.mixture.physics_layer_mod import physics_layer
from source_routine.mixture.physics_layer_mod import lognorm_layer
from source_routine.descriptor_builder import descriptor_layer
from source_routine.mixture.force_layer_mod import force_layer
print("alpha_nes: Inference of NEW CG model")


class alpha_nes_full_inference(tf.Module):
      def __init__(self,modelname):
          super(alpha_nes_full_inference, self).__init__()
          self.max_batch=1
          self.number_of_NN=np.loadtxt(modelname+'/number_of_nn.dat',dtype='int32')
          self.color_type_map=np.loadtxt(modelname+'/color_type_map.dat',dtype='int32')
          self.map_color_interaction=np.loadtxt(modelname+'/map_color_interaction.dat',dtype='int32')
          self.map_intra=np.loadtxt(modelname+'/map_intra.dat',dtype='int32')
          self.cutoff_info=np.loadtxt(modelname+'/cutoff_info')
          self.rc=float(self.cutoff_info[0,0])
          self.rad_buff=int(self.cutoff_info[0,1])
          self.rc_ang=float(self.cutoff_info[1,0])
          self.ang_buff=int(self.cutoff_info[1,1])
          self.rs=self.cutoff_info[2,0]
          self.boxinit=np.array([12.,0.,0.,12.,0.,12.],dtype='float64')
          self.N=len(self.color_type_map)
          print("Model 4")
          print("Alpha_inference: Found ",len(self.map_color_interaction)," colors of atoms")
          print("Alpha_inference: Found ",self.N," atoms")
          print("Alpha_inference: Found ",self.rad_buff," for radial buffer")
          print("Alpha_inference: Found ",self.ang_buff," for angular buffer")
          print("Alpha_inference: Found ",self.rc," for cutoff 2body")
          print("Alpha_inference: Found ",self.rc_ang," for cutoff 3body")
          print("Alpha_inference: Found ",self.rs," for hard cutoff")

          self.descriptor_layer=descriptor_layer(self.rc,self.rad_buff,self.rc_ang,self.ang_buff,self.N,self.boxinit,self.rs,self.max_batch)

          init_alpha2b=[np.loadtxt(modelname+'/type'+str(k)+'_alpha_2body.dat',dtype='float64').reshape((self.ntipos,-1)) for k in range(self.number_of_NN)]
          init_alpha3b=[np.loadtxt(modelname+'/type'+str(k)+'_alpha_3body.dat',dtype='float64').reshape((self.nt_couple,-1)) for k in range(self.number_of_NN)]

          initial_type_emb2b=[np.loadtxt(modelname+'/type'+str(k)+'_type_emb_2b.dat',dtype='float64') for k in range(self.number_of_NN)]
          initial_type_emb3b=[np.loadtxt(modelname+'/type'+str(k)+'_type_emb_3b.dat',dtype='float64') for k in range(self.number_of_NN)]
          initial_type_emb=[[initial_type_emb2b[k],initial_type_emb3b[k]] for k in range(self.number_of_NN)]
          self.physics_layer=[physics_layer(init_alpha2b[k],init_alpha3b[k],
                       initial_type_emb[k]) for k in range(self.number_of_NN)]

          self.nets=[tf.saved_model.load(modelname+'/model_type'+str(k))
                          for k in range(self.number_of_NN)]
          self.force_layer=force_layer(self.rad_buff,self.ang_buff)





      #@tf.function()
      def full_test(self,pos,box):

          [x1,x2,x3bsupp,
        int2b,int3b,intder2b,
        intder3b,intder3bsupp,numtriplet]=self.descriptor_layer(pos,box)

          self.x2b = x1
          self.x3b = x2
          self.x3bsupp = x3bsupp
          self.int2b = int2b
          self.int3b = int3b
          self.intder2b = intder2b
          self.intder3b = intder3b
          self.intder3bsupp = intder3bsupp
          self.numtriplet = numtriplet

          number_of_NN=self.number_of_NN
          self.fingerprint=[self.physics_layer[k](self.x2b,self.x3bsupp,
          self.int2b,self.x3b,self.int3b,self.numtriplet,self.color_type_map,self.map_color_interaction,self.map_intra)
                      for k in range(number_of_NN)]
          self.log_norm_projdes=[self.lognorm_layer[k](finger)
                           for k,finger in enumerate(self.fingerprint)]
          self.energy=[self.nets[k](cp) for k,cp in enumerate(self.log_norm_projdes)]
          self.grad_ene=[tf.gradients(self.energy[k],cp) for k,cp in enumerate(self.fingerprint)]


          self.totene=tf.concat(self.energy,axis=1)
          self.totenergy=tf.reduce_mean(self.totene,axis=(-1,-2))*0.5

          self.grad_listed=[tf.split(self.grad_ene[k][0],[self.physics_layer[k].nalpha_r,
                                      self.physics_layer[k].nalpha_a],axis=2) for k in range(number_of_NN)]

          self.force_list=[self.force_layer(self.grad_listed[k][0],self.x2b,
                                   self.intder2b,self.int2b,
                                   self.physics_layer[k].alpha2b,
                                   self.grad_listed[k][1],self.x3b,self.x3bsupp,
                                   self.intder3b,self.intder3bsupp,self.int3b,
                                   self.numtriplet,
                                   self.physics_layer[k].alpha3b,
                                   self.physics_layer[k].type_emb_2b,
                                   self.physics_layer[k].type_emb_3b,
                                   self.color_type_map,self.map_color_interaction,
                                   k,self.map_intra) for k in range(number_of_NN)]

          self.force=tf.math.add_n(self.force_list)


          return self.totenergy,self.force
