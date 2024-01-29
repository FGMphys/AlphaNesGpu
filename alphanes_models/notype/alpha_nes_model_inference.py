import tensorflow as tf
import numpy as np
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense

from source_routine.notype.physics_layer_mod import physics_layer
from source_routine.notype.physics_layer_mod import lognorm_layer
from source_routine.descriptor_builder import descriptor_layer
from source_routine.notype.force_layer_mod import force_layer


class alpha_nes_full_inference(tf.Module):
      def __init__(self,modelname,N):
          super(alpha_nes_full_inference, self).__init__()

          print("alpha_nes: inference for only notype system")
          #self.N=int(np.loadtxt(modelname+'/type.dat'))
          self.N=N
          self.ntipos=1
          self.cutoff_info=np.loadtxt(modelname+'/cutoff_info')
          self.rc=float(self.cutoff_info[0,0])
          self.rad_buff=int(self.cutoff_info[0,1])
          self.rc_ang=float(self.cutoff_info[1,0])
          self.ang_buff=int(self.cutoff_info[1,1])
          self.rs=int(self.cutoff_info[2,0])
          self.boxinit=np.array([12.,0.,0.,12.,0.,12.],dtype='float64')

          print("Alpha_inference: Found ",self.ntipos," types of atoms")
          print("Alpha_inference: Found ",self.N," atoms")
          print("Alpha_inference: Found ",self.rad_buff," for radial buffer")
          print("Alpha_inference: Found ",self.ang_buff," for angular buffer")
          print("Alpha_inference: Found ",self.rc," for cutoff 2body")
          print("Alpha_inference: Found ",self.rc_ang," for cutoff 3body")


          self.descriptor_layer=descriptor_layer(self.rc,self.rad_buff,self.rc_ang,
                                                 self.ang_buff,self.N,self.boxinit,self.rs)

          init_alpha2b=[np.loadtxt(modelname+'/type'+str(k)+'_alpha_2body.dat') for k in range(self.ntipos)]
          init_alpha3b=[np.loadtxt(modelname+'/type'+str(k)+'_alpha_3body.dat') for k in range(self.ntipos)]
          self.physics_layer=[physics_layer(init_alpha2b[k],init_alpha3b[k]) for k in range(self.ntipos)]

          self.nets = [tf.saved_model.load(modelname+'/model_type'+str(k)) for k in range(self.ntipos)]

          #init_mu=[np.loadtxt(modelname+'/type'+str(k)+'_alpha_mu.dat') for k in range(self.ntipos)]
          #self.lognorm_layer=[lognorm_layer(init_mu[k]) for k in range(self.ntipos)]

          #self.nets = [tf.keras.models.load_model(modelname+'/net_model_type'+str(k))
            #          for k in range(self.ntipos)]

          self.force_layer=force_layer()





      @tf.function()
      def full_test(self,pos,box):

          [x,x3bsuppmap,int2bmap,int3bmap,intder2b,
          intder3b,intder3bsupp]=self.descriptor_layer(pos,box)



          nr=self.rad_buff
          na=self.ang_buff

          self.x2b=x[:,:,:nr]
          self.x3b=x[:,:,nr:]
          self.x3bsupp=x3bsuppmap
          self.int2b=int2bmap
          self.int3b=int3bmap
          self.intder2b=intder2b
          self.intder3b=intder3b
          self.intder3bsupp=intder3bsupp


          self.N=tf.shape(self.x2b)[1]
          dimbat=pos.shape[0]
          self.fingerprint=self.physics_layer[0](self.x2b,self.x3bsupp,nr,self.int2b,self.x3b,na,self.int3b,self.N,dimbat)
          #self.log_norm_projdes=self.lognorm_layer[0](self.fingerprint)
          [self.energy,self.grad_ene]=self.nets[0].testmodel(self.fingerprint)
          #self.grad_ene=tf.gradients(self.energy,self.fingerprint)

          self.totene=self.energy#tf.concat(self.energy,axis=1)
          self.totenergy=self.energy#tf.reduce_mean(self.totene,axis=(-1,-2))*0.5


          #self.grad_ene=tf.gradients(self.energy,self.fingerprint)
          [self.grad_r,self.grad_a]=tf.split(self.grad_ene,[self.physics_layer[0].nalpha_r,self.physics_layer[0].nalpha_a],axis=-1)

          self.force=self.force_layer(self.grad_r,self.x2b,nr,
                                 self.intder2b,self.int2b,
                                 self.physics_layer[0].nalpha_r,self.physics_layer[0].alpha2b,
                                 self.grad_a,self.x3b,self.x3bsupp,na,
                                 self.intder3b,self.intder3bsupp,self.int3b,
                                 self.physics_layer[0].nalpha_a,
                                 self.physics_layer[0].alpha3b,self.N,dimbat)

          return self.totenergy,self.force
