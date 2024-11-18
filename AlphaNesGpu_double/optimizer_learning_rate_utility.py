import sys

import numpy as np

import tensorflow as tf
import tensorflow.keras.optimizers.schedules as optsch
import tensorflow.keras.experimental as tfexp
import tensorflow.keras.optimizers as tfopt

tf.keras.backend.set_floatx('float64')

def build_learning_rate(param,ne,nb,buffer_stream_tr,name,num_call):
    if param[0]=='expdec':
       tf=ne*buffer_stream_tr
       try:
           initial_learning_rate=np.float64(float(param[1]))
       except:
           initial_learning_rate=np.float64(0.001)
       try:
           final_learning_rate=np.float64(float(param[2]))
       except:
           final_learning_rate=np.float64(10**(-7))
       if num_call==0:
           print("alpha_nes: ",name," learning rate decay is set to exponential decay.",sep=' ',end='\n')
           print("alpha_nes: ",name," initial learning rate set to",initial_learning_rate,sep=' ',end='\n')
           print("alpha_nes: ",name," final learning rate set to",final_learning_rate,sep=' ',end='\n')
       lr_built=optsch.ExponentialDecay(initial_learning_rate,1,(final_learning_rate/initial_learning_rate)**(1/tf),
                         staircase=False, name=None)

    elif param[0]=='cosann':
        try:
            initial_learning_rate=np.float64(float(param[1]))
        except:
            initial_learning_rate=np.float64(0.01)
        try:
            first_decay_steps=int(float(param[2])*nb*buffer_stream_tr)
        except:
            first_decay_steps=int(2*nb*buffer_stream_tr)
        try:
            t_mul=np.float64(float(param[3]))
        except:
            t_mul=np.float64(2.0)
        try:
            m_mul=np.float64(float(param[4]))
        except:
            m_mul=np.float64(1.0)
        try:
            alpha=np.float64(float(param[5]))
        except:
            alpha=np.float64(0.0)
        if num_call==0:
            print("alpha_nes: ",name," learning rate decay is set to cosine annealing.",sep=' ',end='\n')
            print("alpha_nes: ",name," initial learning rate set to",initial_learning_rate,sep=' ',end='\n')
            print("alpha_nes: ",name," first decay steps set to",first_decay_steps,sep=' ',end='\n')
            print("alpha_nes: ",name," t_mul set to ",t_mul,sep=' ',end='\n')
            print("alpha_nes: ",name," m_mul set to",m_mul,sep=' ',end='\n')
            print("alpha_nes: ",name," alpha set to",alpha,sep=' ',end='\n')
        lr_built=optsch.CosineDecayRestarts(initial_learning_rate,first_decay_steps,t_mul=t_mul,
                          m_mul=m_mul,alpha=alpha,name=None)

    else:
        sys.exit("alpha_nes: learning rate can be expdec or cosann.")
    return lr_built

def build_optimizer(param,lr_built,num_call):
    if param[0]=='adam':
       try:
           epsilon=float(param[1])
       except:
           epsilon=1e-07
       opt_built=tfopt.Adam(learning_rate=lr_built,epsilon=epsilon)
       if num_call==0:
           print("alpha_nes: The optimizer is Adam.",sep=' ',end='\n')
           print("alpha_nes: epsilon is set to ",epsilon,sep=' ',end='\n')

    elif param[0]=='sgd':
         try:
           momentum=float(param[1])
         except:
           epsilon=0.1
         opt_built=tfopt.SGD(learning_rate=lr_built,momentum=momentum)
         if num_call==0:
             print("alpha_nes: The optimizer is SGD.",sep=' ',end='\n')
             print("alpha_nes: momentum is set to ",momentum,sep=' ',end='\n')
    else:
        sys.exit("alpha_nes: optimizer can be only adam or sgd.")

    return opt_built
