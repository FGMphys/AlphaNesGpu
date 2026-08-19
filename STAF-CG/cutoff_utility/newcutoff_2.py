import numpy as np
import matplotlib.pyplot as plt
import sys

def coeffC(A,B,f):
    return f-A-B

def coeffB(alpha,beta,rs,f1,f2):
    return (f1*rs+f2*rs**2/(alpha+1))*(alpha+1)/beta/(beta-alpha)

def coeffA(alpha,beta,rs,f1,f2,B):
    return (f2*rs**2-B*beta*(beta+1))/(alpha*(alpha+1))

def check_beta_fun(c1,c2,rs):
    Rstar=(-c2/c1)**(1/(beta-alpha))*rs
    check=True
    if Rstar>rs:
      check=False
    return check

class flexcutoff():
      def __init__(self,alpha,beta,rc,rs,f,f1,f2):
          super(flexcutoff,self).__init__()
          self.B=coeffB(alpha,beta,rs,f1,f2)
          self.A=coeffA(alpha,beta,rs,f1,f2,self.B)
          self.C=coeffC(self.A,self.B,f)
          self.alpha=alpha
          self.beta=beta
          self.rs=rs
          self.rc=rc
          c2=self.B*beta*(beta+1)
          c1=self.A*alpha*(alpha+1)
          if rs<rc/2:
             check_beta=check_beta_fun(c1,c2,rs)
             if check_beta==True:
                print("Flesso possibile in ",(-c2/c1)**(1/(beta-alpha))*rs)
             elif check_beta==False:
                print("Flesso non possibile con beta selezionato ",(-c2/c1)**(1/(beta-alpha))*rs)
                sys.exit("")
             
      def compute(self,x):
          if x<self.rs:
             res=self.A*(x/self.rs)**(-self.alpha)+self.B*(x/self.rs)**(-self.beta)+self.C
          else:
             res=0.5*(1+np.cos(np.pi*x/self.rc))
          return res
      def get_param(self):
          return self.A,self.alpha,self.B,self.beta,self.C

def getbeta(alpha,rs):
    if alpha-rs>0:
       res=alpha-rs+rs+0.01
    else:
       res=alpha+0.01
    return res
#rc=8.0
#x=np.arange(0.2,rc,0.001)


rc_vec=[50]
alpha_vec=[1]
beta=-3
for alpha in alpha_vec:
    for rc in rc_vec:
        rs_vec=[15,20,25,30,35]#np.arange(0.1,rc/2.+0.01,0.1)
        for k,rs in enumerate(rs_vec):
            x=np.arange(0.4,rc,0.0001)
            f2=0.5*(np.pi/rc)**2*(-np.cos(np.pi*rs/rc))
            f1=0.5*np.pi/rc*(-np.sin(np.pi*rs/rc))
            f=0.5*(1+np.cos(np.pi*rs/rc))
            print("Alpha Beta rs",alpha,beta,rs)
            Flexcutoff=flexcutoff(alpha,beta,rc,rs,f,f1,f2)
            params=Flexcutoff.get_param()
            print(f2,params[0]*params[1]*(params[1]+1)/rs**2+params[2]*params[3]*(params[3]+1)/rs**2)
            print(f1,-params[0]*params[1]/rs-params[2]*params[3]/rs)
            print(f,params[0]+params[2]+params[4])
            #print(alpha,Flexcutoff.get_param()[-2])
            y_m=np.array([Flexcutoff.compute(xel) for xel in x])
            plt.plot(x,y_m,label='rs '+str(round(rs,3))+'alpha '+str(round(alpha,3))+'rc '+str(round(rc,3)))
            plt.legend()
        plt.show()
