import tensorflow as tf
import numpy as np



root_path='/home/francegm/alphaGPU_done/alphanes_mixture_v4_Ck_notrain_tf11'
proj2b_sopath=root_path+'/src/notype/bin/op_2bAFs.so'
proj3b_sopath=root_path+'/src/notype/bin/op_3bAFs.so'


class physics_layer(tf.Module):
      def __init__(self,init_alpha2b,init_alpha3b):
          super(physics_layer, self).__init__()
          self.proj2b=tf.load_op_library(proj2b_sopath)
          self.proj3b=tf.load_op_library(proj3b_sopath)

          self.alpha2b=tf.Variable(init_alpha2b)
          self.alpha3b=tf.Variable(init_alpha3b)


          self.nalpha_r=int(init_alpha2b.shape[0])
          self.nalpha_a=int(init_alpha3b.shape[0])

          self.output_dim=self.nalpha_r+self.nalpha_a




      @tf.function()
      def __call__(self,x2b,x3bsupp,nr,intmap2b,x3b,na,intmap3b,N,dimbat):


          self.resproj2b=self.proj2b.compute_sort_proj(x2b,N,nr,dimbat,intmap2b,
                                                       self.alpha2b,self.nalpha_r)
          ###Da eliminare numneigh2b perchè non viene usato!!!
          self.resproj3b=self.proj3b.compute_sort_proj3body(x3b,x3bsupp,na,nr,
                                                           intmap3b,N,dimbat,
                                                           intmap2b,self.alpha3b,
                                                           self.nalpha_a)
          self.restot=tf.concat([self.resproj2b,self.resproj3b],axis=2)
          return self.restot
      def savealphas(self,folder_ou,prefix):
         np.savetxt(folder_ou+'/'+prefix+'alpha_2body.dat',self.alpha2b.numpy())
         np.savetxt(folder_ou+'/'+prefix+'alpha_3body.dat',self.alpha3b.numpy())

class lognorm_layer(tf.Module):
      def __init__(self,mu):
          super(lognorm_layer, self).__init__()
          self.mu=tf.Variable(mu,trainable=True)
      @tf.function()
      def __call__(self,restot):
          return tf.math.log(restot+10**(-3))-self.mu
      def savemu(self,folder_ou,prefix):
          np.savetxt(folder_ou+'/'+prefix+'alpha_mu.dat',self.mu.numpy())
