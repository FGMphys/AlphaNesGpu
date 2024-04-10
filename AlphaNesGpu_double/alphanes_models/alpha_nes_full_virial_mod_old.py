import tensorflow as tf
import numpy as np
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LayerNormalization
from  tensorflow.keras.layers import BatchNormalization

class alpha_nes_full(tf.Module):
      def __init__(self,physics_layer,force_layer,num_layers,node_seq,actfun,
                   output_dim,lossfunction,val_loss,opt_net,opt_phys,
                   alpha_bound,lognorm_layer,pressure_layer,tipos,type_map):
          super(alpha_nes_full, self).__init__()

          self.nhl=num_layers
          self.node=node_seq
          self.actfun=actfun
          self.tipos=tipos
          self.ntipos=len(tipos)
          self.type_map=type_map

          self.physics_layer=physics_layer
          self.lognorm_layer=lognorm_layer
          self.force_layer=force_layer
          self.pressure_layer=pressure_layer

          self.nets = [tf.keras.Sequential() for el in range(self.ntipos)]
          for net in self.nets:
              net.add(Input(shape=(1000,self.physics_layer.output_dim,)))
              if self.nhl>0:
                 for k in self.node:
                     net.add(Dense(k, activation=self.actfun))
                 net.add(Dense(output_dim))

          self.lossfunction=lossfunction
          self.val_loss=val_loss

          self.relu_bound=layer=tf.keras.layers.ReLU(max_value=None, threshold=alpha_bound,negative_slope=0.0)

          self.opt_net=opt_net
          self.opt_phys=opt_phys

          self.global_step=0

      @tf.function()
      def __call__(self,x,int2b,int3b):
          N=x.shape[-2]
          dimbat=x.shape[0]
          nr=int2b.shape[-1]-1
          na=x.shape[-1]-nr
          x2b=x[:,:,:nr]
          x3b=x[:,:,nr:]

          self.projdes=self.physics_layer(x2b,nr,int2b[:,:,0],x3b,na,int3b[:,:,0],N,dimbat)

          return self.net(self.projdes)

      @tf.function()
      def full_train_e_f(self,x,x3bsupp,int2b,intder2b,int3b,intder3b,intder3bsupp,
                         etrue,ftrue,pe,pf):

          nalpha_r=self.physics_layer.nalpha_r
          nalpha_a=self.physics_layer.nalpha_a
          nt=self.physics_layer.nt
          nt_couple=self.physics_layer.nt_couple

          nr=int2b.shape[-1]-1
          na=x.shape[-1]-nr
          x2b=x[:,:,:nr]
          x3b=x[:,:,nr:]

          N=x.shape[-2]
          dimbat=x.shape[0]

          self.restot=self.physics_layer.train(x2b,x3bsupp,nr,int2b[:,:,0],x3b,
                                               na,int3b[:,:,0],N,dimbat,
                                               self.type_map)
          self.log_norm_projdes=self.lognorm_layer.train(self.restot)
          self.chemical_proj=tf.split(self.log_norm_projdes,self.tipos,axis=1)

          self.energy=[self.nets[k](cp) for k,cp in enumerate(self.chemical_proj)]
          self.grad_ene=[tf.gradients(self.energy[k],cp) for k,cp in enumerate(self.chemical_proj)]

          self.totgrad=tf.concat(self.grad_ene,axis=2)
          self.totene=tf.concat( self.energy,axis=1)
          self.totenergy=tf.reduce_mean(self.totene,axis=(-1,-2))

          [self.net_der_r,self.net_der_a]=tf.split(self.totgrad,[nalpha_r,nalpha_a],axis=3)


          self.force=self.force_layer(self.net_der_r,x2b,nr,intder2b,int2b,
                                       nalpha_r,self.physics_layer.alpha2b,
                                       self.net_der_a,x3b,x3bsupp,na,
                                       intder3b,intder3bsupp,int3b,nalpha_a,
                                       self.physics_layer.alpha3b,N,dimbat,
                                       self.physics_layer.type_emb_2,nt,
                                       self.physics_layer.type_emb_3b,nt_couple,
                                       self.type_map)

          loss_energy=self.lossfunction(self.totenergy,etrue)
          loss_force=self.lossfunction(self.force,ftrue)

          loss_bound=tf.math.reduce_sum(self.relu_bound(self.physics_layer.alpha2b))+tf.math.reduce_sum(self.relu_bound(self.physics_layer.alpha3b))

          loss=pe*loss_energy+pf*loss_force+loss_bound

          grad_w=[tf.gradients(loss,net.trainable_variables) for net  in  self.nets]
          grad_2b=tf.gradients(loss,self.physics_layer.alpha2b)
          grad_3b=tf.gradients(loss,self.physics_layer.alpha3b)
          grad_mu=tf.gradients(loss,self.lognorm_layer.mu)
          grad_tem_2b=tf.gradients(loss,self.physics_layer.type_emb_2b)
          grad_tem_3b=tf.gradients(loss,self.physics_layer.type_emb_3b)


          for k,grad_per_type_w in  enumerate(grad_w):
              self.opt_net.apply_gradients((grad, var)
                        for (grad, var) in zip(grad_per_type_w, self.nets[k].trainable_variables))

          self.opt_phys.apply_gradients(zip(grad_2b, [self.physics_layer.alpha2b]))
          self.opt_phys.apply_gradients(zip(grad_3b, [self.physics_layer.alpha3b]))
          self.opt_phys.apply_gradients(zip(grad_mu,[self.lognorm_layer.mu]))
          self.opt_phys.apply_gradients(zip(grad_tem_2b, [self.physics_layer.type_emb_2b]))
          self.opt_phys.apply_gradients(zip(grad_tem_3b, [self.physics_layer.type_emb_3b]))

          self.global_step=self.global_step+1

          return loss_force+loss_energy,loss_force,loss_energy

      @tf.function()
      def full_test_e_f(self,x,x3bsupp,int2b,intder2b,int3b,intder3b,intder3bsupp,etrue,ftrue,pos,ptrue):

          nalpha_r=self.physics_layer.nalpha_r
          nalpha_a=self.physics_layer.nalpha_a
          nt=self.physics_layer.nt
          nt_couple=self.physics_layer.nt_couple


          nr=int2b.shape[-1]-1
          na=x.shape[-1]-nr
          dimbat=x.shape[0]
          x2b=x[:,:,:nr]
          x3b=x[:,:,nr:]
          N=x.shape[-2]
          self.restot=self.physics_layer.test(x2b,x3bsupp,nr,int2b[:,:,0],x3b,na,
                                               int3b[:,:,0],N,dimbat,
                                               self.type_map)
          self.log_norm_projdes=self.lognorm_layer.test(self.restot)
          self.chemical_proj=tf.split(self.log_norm_projdes,self.tipos,axis=1)

          self.energy=[self.nets[k](cp) for k,cp in enumerate(self.chemical_proj)]
          self.grad_ene=[tf.gradients(self.energy[k],cp) for k,cp in enumerate(self.chemical_proj)]

          self.totgrad=tf.concat(self.grad_ene,axis=2)
          self.totene=tf.concat(self.energy,axis=1)
          self.totenergy=tf.reduce_mean(self.totene,axis=(-1,-2))

          [self.net_der_r,self.net_der_a]=tf.split(self.totgrad,[nalpha_r,nalpha_a],axis=3)


          [self.force,self.pressure]=self.pressure_layer(self.net_der_r,x2b,nr,intder2b,int2b,
                                     nalpha_r,self.physics_layer.alpha2b,self.net_der_a,x3b,x3bsupp,na,
                                     intder3b,intder3bsupp,int3b,nalpha_a,
                                     self.physics_layer.alpha3b,N,dimbat,pos,
                                     self.physics_layer.type_emb_2b,nt,
                                     self.physics_layer.type_emb_3b,nt_couple,
                                     self.type_map)

          loss_energy=self.val_loss(self.totenergy,etrue)
          loss_force=self.val_loss(self.force,ftrue)
          loss_pressure=self.val_loss(self.pressure,ptrue)
          loss=loss_energy+loss_force

          return loss, loss_force,loss_energy,loss_pressure


      def save_model(self,folder_ou):
          for k,net in enumerate(self.nets):
              net.save(folder_ou+'/net_model_type'+str(k),overwrite=True)
          np.savetxt(folder_ou+'/alpha_2body.dat',self.physics_layer.alpha2b.numpy())
          np.savetxt(folder_ou+'/alpha_3body.dat',self.physics_layer.alpha3b.numpy())
          np.savetxt(folder_ou+'/type_emb_2b.dat',self.physics_layer.type_emb_2b.numpy())
          np.savetxt(folder_ou+'/type_emb_3b.dat',self.physics_layer.type_emb_3b.numpy())
          np.savetxt(folder_ou+'/alpha_mu.dat',self.lognorm_layer.mu.numpy())

      def restart(self,restart):
          if restart!='no':
             self.net=tf.keras.models.load_model(restart+'/model')
             self.physics_layer.alpha2b=tf.Variable(np.loadtxt(restart+"/alpha_2body.dat"))
             self.physics_layer.alpha3b=tf.Variable(np.loadtxt(restart+"/alpha_3body.dat"))
             self.physics_layer.type_emb_2b=tf.Variable(np.loadtxt(restart+"/type_emb_2b.dat"))
             self.physics_layer.type_emb_3b=tf.Variable(np.loadtxt(restart+"/type_emb_3b.dat"))
             self.lognorm_layer.mu=tf.Variable(np.loadtxt(restart+"/alpha_mu.dat"))

             print("alpha_nes: model has been loaded from previous run!",end='\n\n')
