import numpy as np
from scipy.optimize import minimize

# X is the matrix of input parameters. 2 dimensional np array
# y is the corresponding label vector
# r is the lambda coefficient for regularization, if you wish to make sure your model don't end up with parameter values like 34962349252230, 3429253945340, 9876540e+012319519234003549, etc.
# this uses l2 regularization if you didn't notice
# returns w0: the offset of the line, and w: the slope of the line
def L2Linear(X:np.ndarray, y:np.ndarray, r=0):
  n,d = X.shape
  if n < d:
    raise ValueError("Input matrix has more dimensions than datapoints... you sure you didn't get the np shape backwards? " \
    "Anyways screw you I'm terminating your program")
  params = np.concat([np.ones([n,1]), X], axis=1)
  w = np.linalg.inv(params.T @ params + r*np.eye(d+1)) @ params.T @ y
  w0 = w[0,:]
  return w0, w[1:,:]

# X is matrix of input parameters
# y is corresponding label vector
# r is the lambda coefficient for regularization
# literally same as the previous function but it now gives a big clunky kernel, alpha coefficient, and a w0 offset instead.
# why would anyone use this over the previous, you might ask; when n < d, we win
# returns w0: offset of the line, K: the kernel, a: the alpha value
def L2LinearButKernel(X:np.ndarray, y:np.ndarray, r=0):
  n = X.shape[0]
  K = X @ X.T
  def L2loss(u):
    a = u[:-1][:,None]
    w0 = u[-1]
    return np.logaddexp(0.5*((((K @ a) + w0) - y).T @ (((K @ a) + w0) - y)), 0.5*r*(a.T @ K @ a))[0][0]
  
  w = np.zeros(n+1)
  sol = minimize(L2loss, w)

  a = sol['x'][:-1]
  w0 = sol['x'][-1]

  return a, w0

