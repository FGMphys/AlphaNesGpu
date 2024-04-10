import tensorflow as tf
import numpy as np
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense

from source_routine.mixture.physics_layer_mod import physics_layer
from source_routine.mixture.physics_layer_mod import lognorm_layer
from source_routine.descriptor_builder import descriptor_layer
from source_routine.mixture.force_layer_mod import force_layer
print("alpha_nes: Inference of v4 model")
def make_typemap(tipos):
    num=0
    list_tmap=[]
    try:
        res=len(tipos)
    except:
        tipos=[tipos]
    for el in tipos:
        for k in range(el):
            list_tmap.append(num)
        num=num+1
    return list_tmap

class alpha_nes_full_inference(tf.Module):
      def __init__(self,modelname):
          super(alpha_nes_full_inference, self).__init__()
          self.max_batch=1
          self.tipos=tf.constant(np.loadtxt(modelname+'/type.dat').reshape((-1,)),dtype='int32')
          try:
              self.ntipos=len(self.tipos)
          except:
              self.ntipos=1
          self.type_map=tf.constant(make_typemap(np.loadtxt(modelname+'/type.dat',dtype='int32')))
          self.cutoff_info=np.loadtxt(modelname+'/cutoff_info')
          self.rc=float(self.cutoff_info[0,0])
          self.rad_buff=int(self.cutoff_info[0,1])
          self.rc_ang=float(self.cutoff_info[1,0])
          self.ang_buff=int(self.cutoff_info[1,1])
          self.rs=self.cutoff_info[2,0]
          self.boxinit=np.array([12.,0.,0.,12.,0.,12.],dtype='float64')
          self.N=len(self.type_map)
          if self.ntipos==1:
             self.nt_couple=1
          else:
             self.nt_couple=int(self.ntipos*(self.ntipos-1)/2)
          print("Model 4")
          print("Alpha_inference: Found ",self.ntipos," types of atoms")
          print("Alpha_inference: Found ",self.N," atoms")
          print("Alpha_inference: Found ",self.rad_buff," for radial buffer")
          print("Alpha_inference: Found ",self.ang_buff," for angular buffer")
          print("Alpha_inference: Found ",self.rc," for cutoff 2body")
          print("Alpha_inference: Found ",self.rc_ang," for cutoff 3body")
          print("Alpha_inference: Found ",self.rs," for hard cutoff")

          self.descriptor_layer=descriptor_layer(self.rc,self.rad_buff,self.rc_ang,self.ang_buff,self.N,self.boxinit,self.rs,self.max_batch)

          init_alpha2b=[np.loadtxt(modelname+'/type'+str(k)+'_alpha_2body.dat',dtype='float64').reshape((self.ntipos,-1)) for k in range(self.ntipos)]
          init_alpha3b=[np.loadtxt(modelname+'/type'+str(k)+'_alpha_3body.dat',dtype='float64').reshape((self.nt_couple,-1)) for k in range(self.ntipos)]
          if self.ntipos==1:
               num_finger_rad=init_alpha2b[0].shape[1]
               num_finger_ang=init_alpha3b[0].shape[1]
               initial_type_emb2b=np.ones(num_finger_rad,dtype='float64')
               initial_type_emb3b=np.ones(num_finger_ang,dtype='float64')
               initial_type_emb=[initial_type_emb2b,initial_type_emb3b]
          else:
               initial_type_emb2b=[np.loadtxt(modelname+'/type'+str(k)+'_type_emb_2b.dat',dtype='float64') for k in range(self.ntipos)]
               initial_type_emb3b=[np.loadtxt(modelname+'/type'+str(k)+'_type_emb_3b.dat',dtype='float64') for k in range(self.ntipos)]
               initial_type_emb=[[initial_type_emb2b[k],initial_type_emb3b[k]] for k in range(self.ntipos)]
          self.physics_layer=[physics_layer(init_alpha2b[k],init_alpha3b[k],
                       initial_type_emb[k]) for k in range(self.ntipos)]

          self.nets=[tf.saved_model.load(modelname+'/model_type'+str(k))
                          for k in range(self.ntipos)]
          self.force_layer=force_layer(self.rad_buff,self.ang_buff)





      #@tf.function()
      def full_test(self,pos,box):

          [x1,x2,x3bsupp,
        int2b,int3b,intder2b,
        intder3b,intder3bsupp,numtriplet]=self.descriptor_layer(pos,box)

          nt=self.ntipos
          self.x2b=tf.split(x1,self.tipos,axis=1)
          self.x3b=tf.split(x2,self.tipos,axis=1)
          self.x3bsupp=tf.split(x3bsupp,self.tipos,axis=1)
          self.int2b=tf.split(int2b,self.tipos,axis=1)
          self.int3b=tf.split(int3b,self.tipos,axis=1)
          self.numtriplet=tf.split(numtriplet,self.tipos,axis=1)
           
          self.intder2b=tf.split(intder2b,self.tipos,axis=1)
          self.intder3b=tf.split(intder3b,self.tipos,axis=1)
          self.intder3bsupp=tf.split(intder3bsupp,self.tipos,axis=1)
        
          self.fingerprint=[self.physics_layer[k](self.x2b[k],self.x3bsupp[k],
          self.int2b[k],self.x3b[k],self.int3b[k],self.numtriplet[k],self.type_map)                for k in range(nt)]
  

          self.outmodel=[self.nets[k].testmodel(fingers) for k,fingers in enumerate(self.fingerprint)]

          self.energy=[self.outmodel[k][0] for k in range(nt)]

          self.totene=tf.concat(self.energy,axis=1)
          self.totenergy=tf.reduce_sum(self.totene,axis=(-1))


          self.grad_listed=[tf.split(self.outmodel[k][1],[self.physics_layer[k].nalpha_r,
                                      self.physics_layer[k].nalpha_a],axis=-1) for k in range(nt)]

          
          self.force_list=[self.force_layer(self.grad_listed[k][0],self.x2b[k],
                                 self.intder2b[k],self.int2b[k],
                                 self.physics_layer[k].alpha2b,
                                 self.grad_listed[k][1],self.x3b[k],self.x3bsupp[k],
                                 self.intder3b[k],self.intder3bsupp[k],self.int3b[k],
                                 self.numtriplet[k],
                                 self.physics_layer[k].alpha3b,
                                 self.physics_layer[k].type_emb_2b,
                                 self.physics_layer[k].type_emb_3b,
                                 self.type_map,self.tipos,k) for k in range(nt)]

          self.force=tf.math.add_n(self.force_list)


          return self.totenergy,self.force
