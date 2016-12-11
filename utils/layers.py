import numpy as np
from numpy import unravel_index

def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
  examples, where each example x[i] has shape (d_1, ..., d_k). We will
  reshape each input into a vector of dimension D = d_1 * ... * d_k, and
  then transform it to an output vector of dimension M.

  Inputs:
  - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
  - w: A numpy array of weights, of shape (D, M)
  - b: A numpy array of biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  N=x.shape[0]
  M= len(b)
  out = np.zeros((N,M))
  #############################################################################
  for row_number in range(N):
    row=x[row_number]
    row_flat=np.ravel(row)
    out[row_number,:] =row_flat.dot(w) + b
  #############################################################################
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b)
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  N = x.shape[0]
  dw=np.zeros(w.shape)
  #############################################################################
  #############################################################################
  dx = dout.dot(w.T).reshape(x.shape)
  db=np.sum(dout,axis=0)
  for row_number in range(N):
    row_flat = np.ravel(x[row_number])[np.newaxis,:]
    dw += row_flat.T.dot(dout[row_number][np.newaxis,:])
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  #############################################################################
  out=np.maximum(0,x)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = x
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  dx, x = None, cache

  #############################################################################
  #############################################################################
  dx=dout*(x>0)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def batchnorm_forward(x, gamma, beta, bn_param):
  """
  Forward pass for batch normalization.
  
  During training the sample mean and (uncorrected) sample variance are
  computed from minibatch statistics and used to normalize the incoming data.
  During training we also keep an exponentially decaying running mean of the mean
  and variance of each feature, and these averages are used to normalize data
  at test-time.

  At each timestep we update the running averages for mean and variance using
  an exponential decay based on the momentum parameter:

  running_mean = momentum * running_mean + (1 - momentum) * sample_mean
  running_var = momentum * running_var + (1 - momentum) * sample_var

  Note that the batch normalization paper suggests a different test-time
  behavior: they compute sample mean and variance for each feature using a
  large number of training images rather than using a running average. For
  this implementation we have chosen to use running averages instead since
  they do not require an additional estimation step; the torch7 implementation
  of batch normalization also uses running averages.

  Input:
  - x: Data of shape (N, D)
  - gamma: Scale parameter of shape (D,)
  - beta: Shift parameter of shape (D,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

  Returns a tuple of:
  - out: of shape (N, D)
  - cache: A tuple of values needed in the backward pass
  """
  mode = bn_param['mode']
  eps = bn_param.get('eps', 1e-5)
  momentum = bn_param.get('momentum', 0.9)

  N, D = x.shape
  running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
  running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))
  updated_sigma = None
  out, cache = None, None
  sample_mean = np.mean(x, axis=0)
  if mode == 'train':
    #############################################################################
    # Implement the training-time forward pass for batch normalization.         #
    # Use minibatch statistics to compute the mean and variance, use these      #
    # statistics to normalize the incoming data, and scale and shift the        #
    # normalized data using gamma and beta.                                     #
    #                                                                           #
    # You should store the output in the variable out. Any intermediates that   #
    # you need for the backward pass should be stored in the cache variable.    #
    #                                                                           #
    # You should also use your computed sample mean and variance together with  #
    # the momentum variable to update the running mean and running variance,    #
    # storing your result in the running_mean and running_var variables.        #
    #############################################################################
    sample_var = np.var(x,axis=0)
    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var
    updated_sigma=np.sqrt(sample_var + eps)
    x_normalized= (x - sample_mean)/updated_sigma
    out = gamma * x_normalized + beta
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  elif mode == 'test':
    #############################################################################
    # Implement the test-time forward pass for batch normalization. Use         #
    # the running mean and variance to normalize the incoming data, then scale  #
    # and shift the normalized data using gamma and beta. Store the result in   #
    # the out variable.                                                         #
    #############################################################################
    x_normalized = (x - running_mean) / np.sqrt(running_var + eps)
    out = gamma * x_normalized + beta
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  else:
    raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

  # Store the updated running means back into bn_param
  bn_param['running_mean'] = running_mean
  bn_param['running_var'] = running_var
  cache  = (x_normalized,updated_sigma,sample_mean,gamma)
  return out, cache


def batchnorm_backward(dout, cache):
  """
  Backward pass for batch normalization.
  
  For this implementation, you should write out a computation graph for
  batch normalization on paper and propagate gradients backward through
  intermediate nodes.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, D)
  - cache: Variable of intermediates from batchnorm_forward.
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs x, of shape (N, D)
  - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
  - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
  """
  dx, dgamma, dbeta = None, None, None
  N,D =dout.shape
  (x_normalized, updated_sigma, sample_mean, gamma) = cache
  #############################################################################
  # Implement the backward pass for batch normalization. Store the            #
  # results in the dx, dgamma, and dbeta variables.                           #
  #############################################################################
  dbeta = np.sum(dout,axis=0)
  dgamma = np.sum(dout*x_normalized,axis=0)
  dx_normalized = gamma * dout
  d_var= - 0.5  * np.sum(x_normalized*dx_normalized,axis=0) / (updated_sigma**2)
  d_sample_mean = -1.0*np.sum(dx_normalized /updated_sigma,axis=0)
  dx = (dx_normalized / updated_sigma) + 2*(d_var * x_normalized*updated_sigma)/N + d_sample_mean/N
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
  """
  Alternative backward pass for batch normalization.
  
  For this implementation you should work out the derivatives for the batch
  normalizaton backward pass on paper and simplify as much as possible. You
  should be able to derive a simple expression for the backward pass.
  
  Note: This implementation should expect to receive the same cache variable
  as batchnorm_backward, but might not use all of the values in the cache.
  
  Inputs / outputs: Same as batchnorm_backward
  """
  (x_normalized, updated_sigma, sample_mean, gamma) = cache
  #############################################################################
  # Implement the backward pass for batch normalization. Store the            #
  # results in the dx, dgamma, and dbeta variables.                           #
  #                                                                           #
  # After computing the gradient with respect to the centered inputs, you     #
  # should be able to compute gradients with respect to the inputs in a       #
  # single statement; our implementation fits on a single 80-character line.  #
  #############################################################################
  dbeta = np.sum(dout,axis=0)
  dgamma = np.sum(dout*x_normalized,axis=0)
  dx = gamma*(dout - x_normalized*np.mean(dout*x_normalized,axis=0) - np.mean(dout,axis=0))/updated_sigma
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
  """
  Performs the forward pass for (inverted) dropout.

  Inputs:
  - x: Input data, of any shape
  - dropout_param: A dictionary with the following keys:
    - p: Dropout parameter. We drop each neuron output with probability p.
    - mode: 'test' or 'train'. If the mode is train, then perform dropout;
      if the mode is test, then just return the input.
    - seed: Seed for the random number generator. Passing seed makes this
      function deterministic, which is needed for gradient checking but not in
      real networks.

  Outputs:
  - out: Array of the same shape as x.
  - cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
    mask that was used to multiply the input; in test mode, mask is None.
  """
  p, mode = dropout_param['p'], dropout_param['mode']
  if 'seed' in dropout_param:
    np.random.seed(dropout_param['seed'])

  mask = None
  out = None

  if mode == 'train':
    ###########################################################################
    # Implement the training phase forward pass for inverted dropout.         #
    # Store the dropout mask in the mask variable.                            #
    ###########################################################################
    mask = (np.random.rand(*x.shape) < p) / p
    out = x*mask
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    ###########################################################################
    # Implement the test phase forward pass for inverted dropout.             #
    ###########################################################################
    out=x
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################

  cache = (dropout_param, mask)
  out = out.astype(x.dtype, copy=False)

  return out, cache


def dropout_backward(dout, cache):
  """
  Perform the backward pass for (inverted) dropout.

  Inputs:
  - dout: Upstream derivatives, of any shape
  - cache: (dropout_param, mask) from dropout_forward.
  """
  dropout_param, mask = cache
  mode = dropout_param['mode']
  dx = None
  if mode == 'train':
    ###########################################################################
    # Implement the training phase backward pass for inverted dropout.        #
    ###########################################################################
    dx=mask*dout
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    dx = dout
  return dx


def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  #############################################################################
  # Implement the convolutional forward pass.                                 #
  # Hint: you can use the function np.pad for padding.                        #
  #############################################################################
  (N, C, H, W) = x.shape
  (F, C, HH, WW) = w.shape
  stride = conv_param['stride']
  pad = conv_param['pad']
  H_out = 1 + (H + 2 * pad - HH) / stride
  W_out = 1 + (W + 2 * pad - WW) / stride
  out= np.zeros((N, F, H_out, W_out))
  for n in range(N):
        padded_frame = np.pad(x[n],((0,0),(pad,pad),(pad,pad)),'constant',constant_values=0)
        for out_h in range(H_out):
          for out_w in range(W_out):
            for f in range(F):
              x_start=out_h*stride
              y_start=out_w*stride
              local_slice=padded_frame[:,x_start:x_start+HH,y_start:y_start+WW]
              out[n][f][out_h][out_w]= np.sum(local_slice * w[f,:,:]) + b[f]
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives of shape (N, F, H', W')
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  (x, w, b, conv_param) = cache
  (N, C, H, W) = x.shape
  (F, C, HH, WW) = w.shape
  stride = conv_param['stride']
  pad = conv_param['pad']
  (N,F,H_out,W_out)= dout.shape
  dw=np.zeros(w.shape)
  dx= np.zeros(x.shape)
  for n in range(N):
    padded_frame = np.pad(x[n], ((0, 0), (pad, pad), (pad, pad)), 'constant', constant_values=0)
    for out_h in range(H_out):
      for out_w in range(W_out):
        x_start = out_h * stride
        y_start = out_w * stride
        x_end = x_start +HH
        y_end = y_start +WW
        local_slice = padded_frame[:, x_start:x_end, y_start:y_end]
        for f in range(F):
          dw[f] += local_slice * (dout[n,f,out_h,out_w])
          weight_window = w[f,:,max(0,pad-x_start):min(HH,H+HH-x_end+pad),max(0,pad-y_start):min(WW,W+WW-y_end+pad)]*dout[n,f,out_h,out_w]
          x1=max(0,x_start-pad)
          x2 = x1 + min(HH,H+HH-x_end+pad) - max(0,pad-x_start)
          y1 = max(0,y_start - pad)
          y2 = y1 + min(WW,W+WW-y_end+pad) - max(0,pad-y_start)
          dx[n,:,x1:x2,y1:y2] += weight_window
  db = np.sum(dout, axis=(0, 2, 3))
  #############################################################################
  # Implement the convolutional backward pass.                                #
  #############################################################################
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """

  #############################################################################
  # Implement the max pooling forward pass                                    #
  #############################################################################
  (N,C,H,W) = x.shape
  HH=pool_param['pool_height']
  WW =pool_param['pool_width']
  S = pool_param['stride']
  H_out = (H-HH)/S + 1
  W_out = (W-WW)/S + 1
  out = np.zeros((N,C,H_out,W_out))
  for n in range(N):
    for out_h in range(H_out):
      for out_w in range(W_out):
          sliced_input = x[n, :, out_h * S:out_h * S + HH, out_w * S:out_w * S + WW]
          out[n,:,out_h,out_w]  =  np.max(sliced_input,axis=(1,2))
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, pool_param)
  return out, cache


def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives N,C,H',W'
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  x,pool_param = cache
  (N, C, H, W) = x.shape
  dx = np.zeros(x.shape)
  #############################################################################
  # Implement the max pooling backward pass                                   #
  #############################################################################
  HH = pool_param['pool_height']
  WW = pool_param['pool_width']
  S =  pool_param['stride']
  H_out = (H-HH)/S + 1
  W_out = (W-WW)/S + 1
  for n in range(N):
    for out_h in range(H_out):
      for out_w in range(W_out):
        sliced_input = x[n, :, out_h * S:out_h * S + HH, out_w * S:out_w * S + WW]
        for c in range(C):
          max_index_h,max_index_w =unravel_index(sliced_input[c].argmax(),sliced_input[c].shape)
          dx[n,c,out_h*S+max_index_h,out_w * S+max_index_w] = dout[n,c,out_h,out_w]
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.
  
  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features
    
  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  out, cache = np.zeros(x.shape), []
  (N, C, H, W) = x.shape
  #############################################################################
  # Implement the forward pass for spatial batch normalization.               #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################
  #batchnorm_forward(x, gamma, beta, bn_param)

  for c in range(C):
    flattened_layer= x[:,c,:,:].reshape(N,H*W)
    channel_out,channel_cache = batchnorm_forward(flattened_layer,gamma[c]*np.ones(H*W),beta[c]*np.ones(H*W),bn_param)
    cache.append(channel_cache)
    out[:,c,:,:] = channel_out.reshape((N,H,W))
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  (N, C, H, W) =  dout.shape
  dx, dgamma, dbeta = np.zeros(dout.shape), np.zeros(C),np.zeros(C)
  #############################################################################
  # Implement the backward pass for spatial batch normalization.              #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################
  for c in range(C):
    channel_cache = cache[c]
    flattened_layer = dout[:, c, :, :].reshape(N, H * W)
    channel_dx, channel_dgamma, channel_dbeta = batchnorm_backward_alt(flattened_layer,channel_cache)
    dgamma[c] = np.sum(channel_dgamma)
    dbeta[c] = np.sum(channel_dbeta)
    dx[:,c,:,:] = channel_dx.reshape(N,H,W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dgamma, dbeta
  

def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx