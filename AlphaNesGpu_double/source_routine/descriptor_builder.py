import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import yaml
import sys



root_path='/leonardo/pub/userexternal/fguidare/AlphaNesGpu_double'
descriptor_sopath=root_path+'/src/descriptor_builder/reforce.so'

class descriptor_layer(tf.Module):
      def __init__(self,rc,rad_buff,rc_ang,ang_buff,N,box_example,Rs,maxbatch):
          super(descriptor_layer, self).__init__()

          self.descriptor_op=tf.load_op_library(descriptor_sopath)
          self.Rs=Rs
          res=self.descriptor_op.construct_descriptors_light(rc,rad_buff,ang_buff,N,box_example,self.Rs,rc_ang,maxbatch)
          self.rc=rc
          self.rad_buff=rad_buff
          self.rc_ang=rc_ang
          self.ang_buff=ang_buff
          self.N=N
          self.box_example=box_example


      #@tf.function()
      def __call__(self,pos,box):

          [raddescr,angdescr,des3bsupp,intmap2b,intmap3b,
          intder2b,intder3b,
          intder3bsupp,numtriplet]=self.descriptor_op.compute_descriptors_light(pos,box)

          return [raddescr,angdescr,des3bsupp,intmap2b,intmap3b,intder2b,intder3b,intder3bsupp,numtriplet]
