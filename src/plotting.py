import numpy as np
import matplotlib.pyplot as plt

def faststem(x, y, linewidth=1., *args, **kwargs):
    ord = np.argsort(x)
    x = x[ord]
    y = y[ord]
    x = np.array([x,x,x]).T.ravel()
    y = np.array([0*y,y,np.nan*y]).T.ravel()
    plt.gca().plot(x,y,*args,**kwargs)