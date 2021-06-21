import numpy as np

def jacobian(f):
    """
    Returns the jacobian matrix of a function f at a given point x
    usage :
    
    def myfunction(x): #x has to be an (n,1) array
        return np.concatenate([x[0:1,:]**2 + x[1:2,:]**2,x[0:1,:]**2 - x[1:2,:]**2])
    
    x = np.array([[0.],[0.]])
    myjacobian = jacobian(myfunction)(x)
    """ 
    def fbis(x,eps=1.e-4):
        m = f(x).shape[0]
        n = x.shape[0]
        jacob = np.zeros((m,n))
        for k in range(m):
            for l in range(n):
                h = np.zeros((n,1))
                h[l:l+1,:] = eps
                jacob[k:k+1,l:l+1]=((f(x+h)[k:k+1,:]-f(x-h)[k:k+1,:])/(2.*eps))
        return jacob
    return fbis

def hessian(f):
    """
    Returns the hessian matrix of a function f at a given point x
    usage :
    
    def myfunction(x): #x has to be an (n,1) array
        return x[0:1,:]**2 + x[1:2,:]**2
    
    x = np.array([[0.],[0.]])
    myhessian = hessian(myfunction)(x)
    """
    def fbis(x,eps=1.e-4):
        dim = x.shape[0]
        hess = np.zeros((dim,dim))
        for n in range(dim):
            h = np.zeros((dim,1))
            h[n:n+1,:]=eps
            hess[n:n+1,n:n+1] = (f(x+h)+f(x-h)-2*f(x))/(eps**2)
        for n in range(dim):
            for m in range(n+1,dim):
                h = np.zeros((dim,1))
                h[n:n+1,:]=eps
                h[m:m+1,:]=eps
                hess[n:n+1,m:m+1]=0.5*((f(x+h)+f(x-h)-2*f(x))/(eps**2)-hess[n:n+1,n:n+1]-hess[m:m+1,m:m+1])
                hess[m:m+1,n:n+1]=hess[n:n+1,m:m+1]
        return hess
    return fbis