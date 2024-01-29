import numpy as np
import matplotlib.pyplot as plt


def coeffC(alpha,beta,rs,f,f1,f2_red):
    gamma_red=1/(alpha-beta)*alpha-rs**alpha
    delta_red=1/(alpha-beta)*(f*(alpha-beta)-f1*rs-f*alpha)
    eta_red=-alpha/(alpha-beta)
    epsilon_red=1/(alpha-beta)*(rs*f1+alpha*f)
    c2_red=alpha*(alpha+1)*delta_red+beta*(beta+1)*epsilon_red
    c1_red=alpha*(alpha+1)*gamma_red+beta*(beta+1)*eta_red
    C=(f2_red-c2_red)/c1_red
    return C

def coeffB(C,alpha,beta,rs,f1,f):
    eta=-alpha*rs**beta/(alpha-beta)
    epsilon=rs**beta/(alpha-beta)*(rs*f1+alpha*f)
    B=eta*C+epsilon
    return B

def coeffA(C,alpha,beta,rs,f1,f):
    gamma=rs**(alpha)/(alpha-beta)*alpha-rs**alpha
    delta=rs**alpha/(alpha-beta)*(f*(alpha-beta)-f1*rs-f*alpha)
    A=gamma*C+delta
    return A


class flexcutoff():
      def __init__(self,alpha,beta,rc,rs,f,f1,f2_red):
          super(flexcutoff,self).__init__()
          self.C=coeffC(alpha,beta,rs,f,f1,f2_red)
          self.B=coeffB(self.C,alpha,beta,rs,f1,f)
          self.A=coeffA(self.C,alpha,beta,rs,f1,f)
          self.alpha=alpha
          self.beta=beta
          self.rs=rs
          self.rc=rc
      def compute(self,x):
          if x<self.rs:
             res=self.A*x**(-self.alpha)+self.B*x**(-self.beta)+self.C
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


rc_vec=[4.5]
alpha_vec=[0.5]
beta=-30
for alpha in alpha_vec:
    for rc in rc_vec:
        rs_vec=[rc/2.]#np.arange(0.1,rc/2.+0.01,0.1)
        for k,rs in enumerate(rs_vec):
            x=np.arange(0.02,rc,0.001)
            f2_red=0.5*(np.pi/rc)**2*(-np.cos(np.pi*rs/rc))*rs**2
            f1=0.5*np.pi/rc*(-np.sin(np.pi*rs/rc))
            f=0.5*(1+np.cos(np.pi*rs/rc))
            Flexcutoff=flexcutoff(alpha,beta,rc,rs,f,f1,f2_red)
            #print(alpha,Flexcutoff.get_param()[-2])
            y_m=np.array([Flexcutoff.compute(xel) for xel in x])
            plt.plot(x,y_m,label='rs '+str(round(rs,3))+'alpha '+str(round(alpha,3))+'rc '+str(round(rc,3)))
            plt.legend()
        plt.show()
