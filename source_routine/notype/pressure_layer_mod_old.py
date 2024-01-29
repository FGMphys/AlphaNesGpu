import tensorflow as tf


class pressure_layer(tf.Module):
      def __init__(self,press2b_sopath,press3b_sopath):
          self.press2b=tf.load_op_library(press2b_sopath)
          self.press3b=tf.load_op_library(press3b_sopath)

      @tf.function()
      def __call__(self,net_der_r,x2b,nr,intder2b,int2b,nalpha_r,alpha2b,net_der_a,
                   x3b,x3bsupp,na,intder3b,intder3bsupp,int3b,nalpha_a,alpha3b,
                   N,dimbat,pos,type_emb_2b,nt,type_emb_3b,nt_couple,type_map):

          [self.force_radial,self.press_radial]=self.press2b.compute_press_radial(net_der_r,intder2b,int2b,N,nr,dimbat,x2b,nalpha_r,alpha2b,pos,type_emb_2b,nt,type_map)

          [self.force_angular,self.press_angular]=self.press3b.compute_press_tripl(net_der_a,x3bsupp,x3b,intder3bsupp,intder3b,int2b,int3b,nr,na,N,dimbat,nalpha_a,alpha3b,pos,type_emb_3b,nt_couple,type_map)
          self.presstot=self.press_radial+self.press_angular
          self.forcetot=self.force_radial+self.force_angular
          return self.forcetot,self.presstot
