def AND(x1,x2):
    w1,w2,theta=0.5,0.5,0.7
    tmp=x1*w1+x2*w2
    if tmp<=theta:
        return 0
    elif tmp>theta:
        return 1

AND(1,1)
import numpy as np
x=np.array([1,1])
