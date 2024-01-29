import tensorflow as tf
import numpy as np
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense
import pickle

class alpha_nes_full(tf.Module):
    def __init__(self,physics_layer,force_layer,num_layers,node_seq,actfun,
               output_dim,lossfunction,val_loss,opt_net,opt_phys,alpha_bound,
               lognorm_layer,N,restart):
        super(alpha_nes_full, self).__init__()


        self.physics_layer=physics_layer[0]
        self.lognorm_layer=lognorm_layer[0]
        self.force_layer=force_layer

        if restart=='no':
            self.nhl=num_layers
            self.node=node_seq
            self.actfun=actfun
            self.net = tf.keras.Sequential()
            self.net.add(Input(shape=(N,self.physics_layer.output_dim,)))
            if self.nhl>0:
                for k in self.node:
                    self.net.add(Dense(k,activation=self.actfun))
                self.net.add(Dense(output_dim))
        else:
             self.net=tf.keras.models.load_model(restart+'/net_model_type0')
             with open(restart+'/opt_net_weights','rb') as source:
                  weight_net=pickle.load(source)
             self.opt_net_weights=weight_net
             with open(restart+'/opt_phys_weights','rb') as source:
                  weight_phys=pickle.load(source)
             self.opt_phys_weights=weight_phys

        self.lossfunction=lossfunction
        self.val_loss=val_loss

        self.relu_bound=tf.keras.layers.ReLU(max_value=None,
                                           threshold=alpha_bound,
                                           negative_slope=0.0)

        self.opt_net=opt_net
        self.opt_phys=opt_phys

        self.global_step=0.


    @tf.function()
    def full_train_e(self,x,x3bsupp,int2b,intder2b,int3b,intder3b,
                     intder3bsupp,etrue,ftrue,pe,pf):


        nr=int2b.shape[-1]-1
        na=x.shape[-1]-nr

        self.x2b=x[:,:,:nr]
        self.x3b=x[:,:,nr:]
        self.x3bsupp=x3bsupp
        self.int2b=int2b
        self.int3b=int3b

        self.N=int2b.shape[-2]


        dimbat=x.shape[0]
        self.fingerprint=self.physics_layer(self.x2b,self.x3bsupp,nr,
        self.int2b,self.x3b,na,self.int3b,self.N,dimbat)




        self.log_norm_projdes=self.lognorm_layer(self.fingerprint)



        self.energy=self.net(self.log_norm_projdes)
        self.totenergy=tf.reduce_mean(self.energy,axis=(-1,-2))*0.5

        loss_energy=self.lossfunction(self.totenergy,etrue)
        loss_force=tf.constant(0.,dtype='float64')

        loss_bound_2b=tf.math.reduce_sum(self.relu_bound(self.physics_layer.alpha2b))
        loss_bound_3b=tf.math.reduce_sum(self.relu_bound(self.physics_layer.alpha3b))
        loss_bound=loss_bound_2b+loss_bound_3b

        loss=pe*loss_energy+loss_bound

        grad_w=tf.gradients(loss,self.net.trainable_variables)
        grad_2b=tf.gradients(loss,self.physics_layer.alpha2b)
        grad_3b=tf.gradients(loss,self.physics_layer.alpha3b)
        grad_mu=tf.gradients(loss,self.lognorm_layer.mu)

        self.opt_net.apply_gradients((grad, var)
                for (grad, var) in zip(grad_w, self.net.trainable_variables))
        all_AFS_grad=[grad_2b[0],grad_3b[0],grad_mu[0]]
        all_AFs_param=[self.physics_layer.alpha2b,self.physics_layer.alpha3b,self.lognorm_layer.mu]
        self.opt_phys.apply_gradients(zip(all_AFS_grad,all_AFs_param))




        return loss_force+loss_energy,loss_force,loss_energy

    @tf.function()
    def full_test_e(self,x,x3bsupp,int2b,intder2b,int3b,intder3b,intder3bsupp,etrue,ftrue):

        nr=int2b.shape[-1]-1
        na=x.shape[-1]-nr

        self.x2b=x[:,:,:nr]
        self.x3b=x[:,:,nr:]
        self.x3bsupp=x3bsupp
        self.int2b=int2b
        self.int3b=int3b

        self.N=int2b.shape[1]

        dimbat=x.shape[0]
        self.fingerprint=self.physics_layer(self.x2b,self.x3bsupp,nr,
        self.int2b,self.x3b,na,self.int3b,self.N,dimbat)



        self.log_norm_projdes=self.lognorm_layer(self.fingerprint)



        self.energy=self.net(self.log_norm_projdes)
        self.totenergy=tf.reduce_mean(self.energy,axis=(-1,-2))*0.5

        loss_energy=self.val_loss(self.totenergy,etrue)
        loss_force=tf.constant(0.,dtype='float64')
        loss=loss_energy+loss_force

        return loss,loss_force,loss_energy
    @tf.function()
    def full_train_e_f(self,x,x3bsupp,int2b,intder2b,int3b,intder3b,
                     intder3bsupp,etrue,ftrue,pe,pf):
        nr=int2b.shape[-1]-1
        na=x.shape[-1]-nr

        self.N=int2b.shape[1]

        self.x2b=x[:,:,:nr]
        self.x3b=x[:,:,nr:]
        self.x3bsupp=x3bsupp
        self.int2b=int2b
        self.int3b=int3b
        self.intder2b=intder2b
        self.intder3b=intder3b
        self.intder3bsupp=intder3bsupp


        dimbat=x.shape[0]
        self.fingerprint=self.physics_layer(self.x2b,self.x3bsupp,nr,
        self.int2b,self.x3b,na,self.int3b,self.N,dimbat)



        self.log_norm_projdes=self.lognorm_layer(self.fingerprint)



        self.energy=self.net(self.log_norm_projdes)
        self.totenergy=tf.reduce_mean(self.energy,axis=(-1,-2))*0.5

        self.grad_ene=tf.gradients(self.energy,self.fingerprint)
        [self.grad_r,self.grad_a]=tf.split(self.grad_ene,[self.physics_layer.nalpha_r,self.physics_layer.nalpha_a],axis=-1)

        self.force=self.force_layer(self.grad_r,self.x2b,nr,
                                 self.intder2b,self.int2b,
                                 self.physics_layer.nalpha_r,self.physics_layer.alpha2b,
                                 self.grad_a,self.x3b,self.x3bsupp,na,
                                 self.intder3b,self.intder3bsupp,self.int3b,
                                 self.physics_layer.nalpha_a,
                                 self.physics_layer.alpha3b,self.N,dimbat)

        loss_energy=self.lossfunction(self.totenergy,etrue)
        loss_force=self.lossfunction(self.force,ftrue)

        loss_bound_2b=tf.math.reduce_sum(self.relu_bound(self.physics_layer.alpha2b))
        loss_bound_3b=tf.math.reduce_sum(self.relu_bound(self.physics_layer.alpha3b))
        loss_bound=loss_bound_2b+loss_bound_3b

        loss=pe*loss_energy+loss_bound+pf*loss_force

        grad_w=tf.gradients(loss,self.net.trainable_variables)
        grad_2b=tf.gradients(loss,self.physics_layer.alpha2b)
        grad_3b=tf.gradients(loss,self.physics_layer.alpha3b)
        grad_mu=tf.gradients(loss,self.lognorm_layer.mu)

        self.opt_net.apply_gradients((grad, var)
                for (grad, var) in zip(grad_w, self.net.trainable_variables))
        all_AFS_grad=[grad_2b[0],grad_3b[0],grad_mu[0]]
        all_AFs_param=[self.physics_layer.alpha2b,self.physics_layer.alpha3b,self.lognorm_layer.mu]
        self.opt_phys.apply_gradients(zip(all_AFS_grad,all_AFs_param))

        return loss,loss_force,loss_energy

    @tf.function()
    def compile(self,x,x3bsupp,int2b,intder2b,int3b,intder3b,
                     intder3bsupp,etrue,ftrue):
        nr=int2b.shape[-1]-1
        na=x.shape[-1]-nr

        self.N=int2b.shape[1]

        self.x2b=x[:,:,:nr]
        self.x3b=x[:,:,nr:]
        self.x3bsupp=x3bsupp
        self.int2b=int2b
        self.int3b=int3b
        self.intder2b=intder2b
        self.intder3b=intder3b
        self.intder3bsupp=intder3bsupp


        dimbat=x.shape[0]
        self.fingerprint=self.physics_layer(self.x2b,self.x3bsupp,nr,
                                            self.int2b,self.x3b,na,
                                            self.int3b,self.N,dimbat)

        self.log_norm_projdes=self.lognorm_layer(self.fingerprint)

        self.energy=self.net(self.log_norm_projdes)
        self.totenergy=tf.reduce_mean(self.energy,axis=(-1,-2))*0.5

        self.grad_ene=tf.gradients(self.energy,self.fingerprint)

        loss_energy=self.lossfunction(self.totenergy,etrue)
        loss_force=tf.constant(0.,dtype='float64')

        loss_bound_2b=tf.math.reduce_sum(self.relu_bound(self.physics_layer.alpha2b))
        loss_bound_3b=tf.math.reduce_sum(self.relu_bound(self.physics_layer.alpha3b))
        loss_bound=loss_bound_2b+loss_bound_3b

        loss=0.*loss_energy+0.*loss_bound

        grad_w=tf.gradients(loss,self.net.trainable_variables)
        grad_2b=tf.gradients(loss,self.physics_layer.alpha2b)
        grad_3b=tf.gradients(loss,self.physics_layer.alpha3b)
        grad_mu=tf.gradients(loss,self.lognorm_layer.mu)

        self.opt_net.apply_gradients((grad, var)
                for (grad, var) in zip(grad_w, self.net.trainable_variables))
        all_AFS_grad=[grad_2b[0],grad_3b[0],grad_mu[0]]
        all_AFs_param=[self.physics_layer.alpha2b,self.physics_layer.alpha3b,self.lognorm_layer.mu]
        self.opt_phys.apply_gradients(zip(all_AFS_grad,all_AFs_param))
        return 0.

    @tf.function()
    def full_test_e_f(self,x,x3bsupp,int2b,intder2b,int3b,intder3b,intder3bsupp,etrue,ftrue):

        nr=int2b.shape[-1]-1
        na=x.shape[-1]-nr

        self.x2b=x[:,:,:nr]
        self.x3b=x[:,:,nr:]
        self.x3bsupp=x3bsupp
        self.int2b=int2b
        self.int3b=int3b
        self.intder2b=intder2b
        self.intder3b=intder3b
        self.intder3bsupp=intder3bsupp

        self.N=int2b.shape[1]


        dimbat=x.shape[0]
        self.fingerprint=self.physics_layer(self.x2b,self.x3bsupp,nr,
        self.int2b,self.x3b,na,self.int3b,self.N,dimbat)



        self.log_norm_projdes=self.lognorm_layer(self.fingerprint)



        self.energy=self.net(self.log_norm_projdes)
        self.totenergy=tf.reduce_mean(self.energy,axis=(-1,-2))*0.5

        self.grad_ene=tf.gradients(self.energy,self.fingerprint)
        [self.grad_r,self.grad_a]=tf.split(self.grad_ene,[self.physics_layer.nalpha_r,self.physics_layer.nalpha_a],axis=-1)

        self.force=self.force_layer(self.grad_r,self.x2b,nr,
                                 self.intder2b,self.int2b,
                                 self.physics_layer.nalpha_r,self.physics_layer.alpha2b,
                                 self.grad_a,self.x3b,self.x3bsupp,na,
                                 self.intder3b,self.intder3bsupp,self.int3b,
                                 self.physics_layer.nalpha_a,
                                 self.physics_layer.alpha3b,self.N,dimbat)

        loss_energy=self.val_loss(self.totenergy,etrue)
        loss_force=self.val_loss(self.force,ftrue)


        loss=loss_energy+loss_force

        return loss,loss_force,loss_energy


    def save_model(self,folder_ou):
        k=0
        self.net.save(folder_ou+'/net_model_type'+str(k),overwrite=True)
        np.savetxt(folder_ou+'/type'+str(k)+'_alpha_2body.dat',self.physics_layer.alpha2b.numpy())
        np.savetxt(folder_ou+'/type'+str(k)+'_alpha_3body.dat',self.physics_layer.alpha3b.numpy())
        np.savetxt(folder_ou+'/type'+str(k)+'_alpha_mu.dat',self.lognorm_layer.mu.numpy())
        with open(folder_ou+'/opt_net_weights','wb') as dest:
             pickle.dump(self.opt_net.variables(),dest)
        with open(folder_ou+'/opt_phys_weights','wb') as dest:
             pickle.dump(self.opt_phys.variables(),dest)
        with open(folder_ou+'/opt_net_conf','wb') as dest:
             pickle.dump(self.opt_net.get_config(),dest)
        with open(folder_ou+'/opt_phys_conf','wb') as dest:
             pickle.dump(self.opt_phys.get_config(),dest)

    def set_opt_weight(self):
        self.opt_phys.set_weights(self.opt_phys_weights)
        self.opt_net.set_weights(self.opt_net_weights)
    def get_op_weigth(self):
        print(self.opt_phys.get_config())
        return self.opt_phys.variables(),self.opt_net.variables()
    def get_lrnet(self):
        return self.opt_net.learning_rate
    def get_lrphys(self):
        return self.opt_phys.learning_rate
