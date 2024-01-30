import sys
import numpy as np
import matplotlib.pyplot as plt


def explr(lr0,lrf,tf,x):
    return lr0*(lrf/lr0)**(x/tf)

def cosannlr(x,eta0,m_mul,alpha,t_mul,first_dec,num_step):
    lrt=[]
    t_vec=[]
    #tot_step=first_dec*(1-t_mul**(n_cycl+1))/(1-t_mul)
    n_cycl=int(np.ceil(np.log(1-(1-t_mul)*num_step/first_dec)/np.log(t_mul)-1))
    print("For a total number of steps of ",num_step," we must run ",n_cycl," cycles") 
    for l in range(0,n_cycl+1):
        Al=eta0*m_mul**l
        Tl=first_dec*t_mul**l
        t=np.linspace(0,Tl,10)
        lrt_l= Al*((1-alpha)/2*(1+np.cos(np.pi*t/Tl))+alpha)
        lrt.append(lrt_l)
        cyc_shift=first_dec*(1-t_mul**(l-1+1))/(1-t_mul)
        t_vec.append(t+cyc_shift)
    return t_vec,lrt


if sys.argv[1]=='expdec':
   lr0=float(sys.argv[2])
   lrf=float(sys.argv[3])
   num_step=float(sys.argv[4])
   x=np.linspace(0,num_step)
   lrt=explr(lr0,lrf,num_step,x)
elif sys.argv[1]=='cosann':
   eta0=float(sys.argv[2])
   m_mul=float(sys.argv[3])
   print("m_mul",m_mul)
   alpha=float(sys.argv[4])
   t_mul=float(sys.argv[5])
   first_dec=float(sys.argv[6])
   num_step=float(sys.argv[7])
   x=np.linspace(0,num_step)
   
   [t_vec,lrt]=cosannlr(x,eta0,m_mul,alpha,t_mul,first_dec,num_step)
   lrt=np.concatenate(lrt)
   x=np.concatenate(t_vec)/(2304/4)

        
plt.plot(x,lrt)
plt.show()




