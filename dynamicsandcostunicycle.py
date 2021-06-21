from calculdiff import *

n = 3
m = 2
dt = 0.01
xweight = 10000.
uweight = 1.
T = 100
x0 = np.array([[1.],[0.],[0]])
xtarg = np.array([[0.],[0.],[0.]])
xterm = np.array([[0.],[0.],[0.]])
x = np.tile(x0,(1,T+1))
x[:,-1:] = xterm
u = np.zeros((m,T+1))

def next_state(x,u):
    return next_state_warp(np.concatenate([x,u]))
    
def next_state_warp(x):
    """
    Dynamique unicycle
    x = (x,y,theta) en m,m,radians
    u = (vitesse lin,vitesse ang) en m/s,rad/s
    """
    x1 = x[0:1,:]+x[n:n+1,:]*dt*np.cos(x[2:3,:])
    y1 = x[1:2,:]+x[n:n+1,:]*dt*np.sin(x[2:3,:])
    theta1 = x[2:3,:]+x[n+1:n+2,:]*dt
    return np.concatenate([x1,y1,theta1])

def Fx(x,u):
    return jacobian(next_state_warp)(np.concatenate([x,u]))[:,:n]

def Fu(x,u):
    return jacobian(next_state_warp)(np.concatenate([x,u]))[:,n:]

def f1(x,u):
    return next_state(x,u)-Fx(x,u)@x-Fu(x,u)@u

def cost(x,u):
    return cost_warp(np.concatenate([x,u]))

def cost_warp(x):
    """
    Unicycle cost function
    x = (x,y,theta,lin. speed, rotat. speed)
    """
    xtarg = np.array([[0.],[0.],[0.]])
    Cx = xweight*np.eye(n)
    Cx[2:3,2:3] = 0.
    Cu = uweight*np.eye(m)
    return 0.5*(x[:n,:]-xtarg).T@Cx@(x[:n,:]-xtarg) + 0.5*x[n:,:].T@Cu@x[n:,:]

def qx(x,u):
    return jacobian(cost_warp)(np.concatenate([x,u]))[:,:n].T

def qu(x,u):
    return jacobian(cost_warp)(np.concatenate([x,u]))[:,n:].T

def Qxx(x,u):
    return hessian(cost_warp)(np.concatenate([x,u]))[:n,:n]

def Quu(x,u):
    return hessian(cost_warp)(np.concatenate([x,u]))[n:,n:]

def Qux(x,u):
    return hessian(cost_warp)(np.concatenate([x,u]))[n:,:n]