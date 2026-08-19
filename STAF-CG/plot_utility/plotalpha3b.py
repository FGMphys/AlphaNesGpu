import numpy as np
import matplotlib.pyplot as plt 
import sys 

data=np.loadtxt(sys.argv[1])

plt.plot(data[:,0],marker='o',color='r')
plt.plot(data[:,1],marker='o',color='y')
plt.plot(data[:,2],marker='o',color='b',label='beta')
plt.legend()
plt.show()
