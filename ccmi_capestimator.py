# -*- coding: utf-8 -*-

# Commented out IPython magic to ensure Python compatibility.
# import tensorflow as tf
import os
import scipy
import h5py
import glob, os
from scipy.io import loadmat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
from keras import regularizers
from numpy import mean
from numpy import std
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import torch
import torch.nn as nn
from tqdm.auto import tqdm, trange
from numpy.random import default_rng
import torch.nn.functional as F
from google.colab import drive
import gc
#gc.collect()

#drive.mount('/content/drive')
#print(np.version.version)

class MI_Est_Losses():
  def __init__(self, estimator, device):
    """Estimate variational lower bounds on mutual information based on
      a function T(X,Y) represented by a NN using variational lower bounds.
    Args:
      estimator: string specifying estimator, one of:
        'smile', 'nwj', 'infonce', 'tuba', 'js', 'interpolated'
      device: the device to use (CPU or GPU)
    """    
    self.device = device
    self.estimator = estimator



  def logmeanexp_diag(self, x):
    """Compute logmeanexp over the diagonal elements of x."""
    batch_size = x.size(0)

    logsumexp = torch.logsumexp(x.diag(), dim=(0,))
    num_elem = batch_size

    return logsumexp - torch.log(torch.tensor(num_elem).float()).to(self.device)


  def logmeanexp_nodiag(self, x, dim=None):
      batch_size = x.size(0)
      if dim is None:
          dim = (0, 1)

      logsumexp = torch.logsumexp(
          x - torch.diag(np.inf * torch.ones(batch_size).to(self.device)), dim=dim)

      try:
          if len(dim) == 1:
              num_elem = batch_size - 1.
          else:
              num_elem = batch_size * (batch_size - 1.)
      except ValueError:
          num_elem = batch_size - 1
      return logsumexp - torch.log(torch.tensor(num_elem)).to(self.device)


  def tuba_lower_bound(self, scores, log_baseline=None):
      if log_baseline is not None:
          scores -= log_baseline[:, None]

      # First term is an expectation over samples from the joint,
      # which are the diagonal elmements of the scores matrix.
      joint_term = scores.diag().mean()

      # Second term is an expectation over samples from the marginal,
      # which are the off-diagonal elements of the scores matrix.
      marg_term = self.logmeanexp_nodiag(scores).exp()
      return 1. + joint_term - marg_term


  def nwj_lower_bound(self, scores):
      return self.tuba_lower_bound(scores - 1.)


  def infonce_lower_bound(scores):
      nll = scores.diag().mean() - scores.logsumexp(dim=1)
      # Alternative implementation:
      # nll = -tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores, labels=tf.range(batch_size))
      mi = torch.tensor(scores.size(0)).float().log() + nll
      mi = mi.mean()
      return mi


  def js_fgan_lower_bound(self, f):
      """Lower bound on Jensen-Shannon divergence from Nowozin et al. (2016)."""
      f_diag = f.diag()
      first_term = -F.softplus(-f_diag).mean()
      n = f.size(0)
      second_term = (torch.sum(F.softplus(f)) -
                    torch.sum(F.softplus(f_diag))) / (n * (n - 1.))
      return first_term - second_term


  def js_lower_bound(self, f):
      """Obtain density ratio from JS lower bound then output MI estimate from NWJ bound."""
      nwj = self.nwj_lower_bound(f)
      js = self.js_fgan_lower_bound(f)

      with torch.no_grad():
          nwj_js = nwj - js

      return js + nwj_js


  def dv_upper_lower_bound(self, f):
      """
      Donsker-Varadhan lower bound, but upper bounded by using log outside.
      Similar to MINE, but did not involve the term for moving averages.
      """
      first_term = f.diag().mean()
      second_term = self.logmeanexp_nodiag(f)

      return first_term - second_term


  def mine_lower_bound(self, f, buffer=None, momentum=0.9):
      """
      MINE lower bound based on DV inequality.
      """
      if buffer is None:
          buffer = torch.tensor(1.0).to(self.device)
      first_term = f.diag().mean()

      buffer_update = self.logmeanexp_nodiag(f).exp()
      with torch.no_grad():
          second_term = self.logmeanexp_nodiag(f)
          buffer_new = buffer * momentum + buffer_update * (1 - momentum)
          buffer_new = torch.clamp(buffer_new, min=1e-4)
          third_term_no_grad = buffer_update / buffer_new

      third_term_grad = buffer_update / buffer_new

      return first_term - second_term - third_term_grad + third_term_no_grad, buffer_update


  def smile_lower_bound(self,f, clip=None):
      if clip is not None:
          f_ = torch.clamp(f, -clip, clip)
      else:
          f_ = f
      z = self.logmeanexp_nodiag(f_, dim=(0, 1))
      dv = f.diag().mean() - z

      js = self.js_fgan_lower_bound(f)

      with torch.no_grad():
          dv_js = dv - js

      return js + dv_js


  def _ent_js_fgan_lower_bound(self, vec, ref_vec):
      """Lower bound on Jensen-Shannon divergence from Nowozin et al. (2016)."""
      first_term = -F.softplus(-vec).mean()
      second_term = torch.sum(F.softplus(ref_vec)) / ref_vec.size(0)
      return first_term - second_term

  def _ent_smile_lower_bound(self, vec, ref_vec, clip=None):
      if clip is not None:
          ref = torch.clamp(ref_vec, -clip, clip)
      else:
          ref = ref_vec

      batch_size = ref.size(0)
      z = log_mean_ef_ref = torch.logsumexp(ref, dim=(0, 1)) - torch.log(torch.tensor(batch_size)).to(self.device)
      dv = vec.mean() - z
      js = self._ent_js_fgan_lower_bound(vec, ref_vec)

      with torch.no_grad():
          dv_js = dv - js
      
      return js + dv_js

  def entropic_smile_lower_bound(self, f, clip=None):
      t_xy, t_xy_ref, t_x, t_x_ref, t_y, t_y_ref = f

      d_xy = self._ent_smile_lower_bound(t_xy, t_xy_ref, clip=clip)
      d_x = self._ent_smile_lower_bound(t_x, t_x_ref, clip=clip)
      d_y = self._ent_smile_lower_bound(t_y, t_y_ref, clip=clip)

      return d_xy, d_x, d_y

  def chisquare_pred(self, vec, ref_vec, beta=.9, with_smile=False, clip=None):
      b = torch.tensor(beta).to(self.device)
      if with_smile:
        if clip is not None:
            ref = torch.clamp(ref_vec, -clip, clip)
        else:
            ref = ref_vec
        
        batch_size = ref.size(0)
        z = log_mean_ef_ref = torch.logsumexp(ref, dim=(0, 1)) - torch.log(torch.tensor(batch_size)).to(self.device)
      else:
        z = log_mean_ef_ref = torch.mean(torch.exp(ref_vec))/b - torch.log(b) - 1
      
      dv = vec.mean() - z
      js = self._ent_js_fgan_lower_bound(vec, ref_vec)

      with torch.no_grad():
          dv_js = dv - js
      
      return js + dv_js

  def chi_square_lower_bound(self, f, beta=.9, with_smile=False, clip=None): 
      t_xy, t_xy_ref = f

      d_xy = self.chisquare_pred(t_xy, t_xy_ref, beta=beta, with_smile=with_smile, clip=clip)
    
      return d_xy

  def mi_est_loss(self, net_output, **kwargs):
      """Estimate variational lower bounds on mutual information.

    Args:
      net_output: output(s) of the neural network estimator

    Returns:
      scalar estimate of mutual information
      """
      if self.estimator == 'infonce':
          mi = self.infonce_lower_bound(net_output)
      elif self.estimator == 'nwj':
          mi = self.nwj_lower_bound(net_output)
      elif self.estimator == 'tuba':
          mi = self.tuba_lower_bound(net_output, **kwargs)
      elif self.estimator == 'js':
          mi = self.js_lower_bound(net_output)
      elif self.estimator == 'smile':
          mi = self.smile_lower_bound(net_output, **kwargs)
      elif self.estimator == 'dv':
          mi = self.dv_upper_lower_bound(net_output)
      elif self.estimator == 'ent_smile':
          mi = self.entropic_smile_lower_bound(net_output, **kwargs)
      elif self.estimator == 'chi_square':
          mi = self.chi_square_lower_bound(net_output, **kwargs)
      return mi



class Linear1_Net(nn.Module):
    def __init__(self,
                 In_1=256,
        stride_size=1,
        num_channels=1,
        depth_1=1,
        depth_2=5,
        kernel_size_2=2,
        num_hidden=2048,
        num_labels=5
                ):
        super(Linear1_Net, self).__init__()
        
        self.classifier=nn.Sequential(
            nn.Linear(2,num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden,num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden,num_hidden),
            nn.ReLU(),

            #nn.MaxPool1d(1),
            #nn.Dropout(0.1),
            #nn.Conv1d(in_channels=1, out_channels=1, kernel_size=5)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
          
        )

        self.fc2 = nn.Sequential(
           
            nn.Linear(num_hidden,1)
            
        )
        self.sigm=nn.Sigmoid()
    def forward(self,x):
        x=self.classifier(x)
        x=self.fc1(x)
        logit=self.fc2(x)
        prob=self.sigm(logit)
        return prob,logit

def mlp(dim, hidden_dim, output_dim, layers, activation):
    """Create a mlp from the configurations."""
    activation = {
        'relu': nn.ReLU,
        'lrelu': nn.LeakyReLU
    }[activation]

    seq = [nn.Linear(dim, hidden_dim), activation()]
    for _ in range(layers):
        seq += [nn.Linear(hidden_dim, hidden_dim), activation()]
    seq += [nn.Linear(hidden_dim, output_dim)]

    return nn.Sequential(*seq)


class SeparableCritic(nn.Module):
    """Separable critic. where the output value is g(x) h(y). """

    def __init__(self, dim, hidden_dim, embed_dim, layers, activation, **extra_kwargs):
        super(SeparableCritic, self).__init__()
        self._g = mlp(dim, hidden_dim, embed_dim, layers, activation)
        self._h = mlp(dim, hidden_dim, embed_dim, layers, activation)

    def forward(self, x, y):
        scores = torch.matmul(self._h(y), self._g(x).t())
        return scores


class ConcatCritic(nn.Module):
    """Concat critic, where we concat the inputs and use one MLP to output the value."""

    def __init__(self, dim, hidden_dim, layers, activation, **extra_kwargs):
        super(ConcatCritic, self).__init__()
        # output is scalar score
        self._f = mlp(dim * 2, hidden_dim, 1, layers, activation)
        
        #self._f=CNN1_Net()

    def forward(self, x, y):
        batch_size = x.size(0)
        # Tile all possible combinations of x and y
        x_tiled = torch.stack([x] * batch_size, dim=0)
        y_tiled = torch.stack([y] * batch_size, dim=1)
        # xy is [batch_size * batch_size, x_dim + y_dim]
        xy_pairs = torch.reshape(torch.cat((x_tiled, y_tiled), dim=2), [
                                 batch_size * batch_size, -1])
        # Compute scores for each x_i, y_j pair.
        scores = self._f(xy_pairs)
        return torch.reshape(scores, [batch_size, batch_size]).t()

class ConcatCritic1(nn.Module):
    """Concat critic, where we concat the inputs and use one MLP to output the value."""

    def __init__(self, dim, hidden_dim, layers, activation, **extra_kwargs):
        super(ConcatCritic1, self).__init__()
        # output is scalar score
       # self._f = mlp(dim * 2, hidden_dim, 1, layers, activation)
        self._f=Linear1_Net()
        #self._f=CNN1_Net()

    def forward(self, x, y):
        batch_size = x.size(0)
        # Tile all possible combinations of x and y
        x_tiled = torch.stack([x] * batch_size, dim=0)
        y_tiled = torch.stack([y] * batch_size, dim=1)
        # xy is [batch_size * batch_size, x_dim + y_dim]
        xy_pairs = torch.reshape(torch.cat((x_tiled, y_tiled), dim=2), [
                                 batch_size * batch_size, -1])
        # Compute scores for each x_i, y_j pair.
        prob,logit = self._f(xy_pairs)
        return torch.reshape(prob, [batch_size, batch_size]).t(),torch.reshape(logit, [batch_size, batch_size]).t()

class EntropicCritic(nn.Module):
  # Inner class that defines the neural network architecture
  def __init__(self, dim, hidden_dim, embed_dim, layers, activation):
      super(EntropicCritic, self).__init__()
      self.f = mlp(dim, hidden_dim, embed_dim, layers, activation)

  def forward(self, inp):
      output = self.f(inp)
      return output


class ConcatEntropicCritic():
    """Concat entropic critic, where we concat the inputs and use one MLP to output the value."""
    def __init__(self, dim, hidden_dim, layers, activation, 
                 device, ref_batch_factor=1,**extra_kwargs):
        # output is scalar score
        self.ref_batch_factor = ref_batch_factor
        self.fxy = EntropicCritic(dim * 2, hidden_dim, 1, layers, activation)
        self.fx = EntropicCritic(dim, hidden_dim, 1, layers, activation)
        self.fy = EntropicCritic(dim, hidden_dim, 1, layers, activation)
        self.device = device

    def _uniform_sample(self, data, batch_size):
      # Sample the reference uniform distribution
      data_min = data.min(dim=0)[0]
      data_max = data.max(dim=0)[0]
      return (data_max - data_min) * torch.rand((batch_size, data_min.shape[0])).to(self.device) + data_min

    def to(self, device):
      self.fxy.to(device)
      self.fx.to(device)
      self.fy.to(device)

    def forward(self, x, y):
        batch_size = x.size(0)
        XY = torch.cat((x, y), dim=1)
        X_ref = self._uniform_sample(x, batch_size=int(
            self.ref_batch_factor * batch_size))
        Y_ref = self._uniform_sample(y, batch_size=int(
            self.ref_batch_factor * batch_size))
        XY_ref = torch.cat((X_ref, Y_ref), dim=1)

        # Compute t function outputs approximated by NNs
        t_xy = self.fxy(XY)
        t_xy_ref = self.fxy(XY_ref)
        t_x = self.fx(x)
        t_x_ref = self.fx(X_ref)
        t_y = self.fy(y)
        t_y_ref = self.fy(Y_ref)
        return (t_xy, t_xy_ref, t_x, t_x_ref, t_y, t_y_ref)

class chiCritic():
    """Concat entropic critic, where we concat the inputs and use one MLP to output the value."""
    def __init__(self, dim, hidden_dim, layers, activation, 
                 device, ref_batch_factor=1,**extra_kwargs):
        # output is scalar score
        self.ref_batch_factor = ref_batch_factor
        self.fxy = EntropicCritic(dim * 2, hidden_dim, 1, layers, activation)
        
        
        self.device = device

    def _uniform_sample(self, data, batch_size):
      # Sample the reference uniform distribution
      data_min = data.min(dim=0)[0]
      data_max = data.max(dim=0)[0]
      return (data_max - data_min) * torch.rand((batch_size, data_min.shape[0])).to(self.device) + data_min

    def to(self, device):
      self.fxy.to(device)

    def forward(self, x, y):
        batch_size = x.size(0)
        XY = torch.cat((x, y), dim=1)
        X_ref = self._uniform_sample(x, batch_size=int(
            self.ref_batch_factor * batch_size))
        Y_ref = self._uniform_sample(y, batch_size=int(
            self.ref_batch_factor * batch_size))
        XY_ref = torch.cat((X_ref, Y_ref), dim=1)

        # Compute t function outputs approximated by NNs
        t_xy = self.fxy(XY)
        t_xy_ref = self.fxy(XY_ref)
        return (t_xy, t_xy_ref)





class CNN1_Net(nn.Module):
    def __init__(self,
                 kernel_size_1=5,
        stride_size=1,
        num_channels=1,
        depth_1=1,
        depth_2=5,
        kernel_size_2=2,
        num_hidden=1024,
        num_labels=5
                ):
        super(CNN1_Net, self).__init__()
        
        self.classifier=nn.Sequential(
            nn.Conv1d(1,1, kernel_size=kernel_size_1),
            nn.ReLU(),
            nn.Conv1d(1,1,kernel_size=5),
            nn.ReLU(),
            nn.Conv1d(1,1,kernel_size=5),
            nn.ReLU(),

            #nn.MaxPool1d(1),
            #nn.Dropout(0.1),
            #nn.Conv1d(in_channels=1, out_channels=1, kernel_size=5)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(244, 256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden,num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden,1)
            
        )
    
    def forward(self,x):
        x=self.classifier(x)
        x=self.fc1(x)
        x=self.fc2(x)
       # print(x)
        return(x)

class PeakConstraint(nn.Module):
  """Implements an activation for peak constraint """
  def __init__(self, peak, **extra_kwargs):
    super(PeakConstraint, self).__init__()
    self.peak_activation = nn.Threshold(-peak, -peak)
  def forward(self, x):
    x = self.peak_activation(x)
    neg1 = torch.tensor(-1.0)
    x = neg1 * x
    x = self.peak_activation(x)
    x = neg1 * x
    return x


class NIT(nn.Module):
  """NIT """
  def __init__(self, dim, hidden_dim, layers, activation, avg_P, peak=None, **extra_kwargs):
    super(NIT, self).__init__()
   # self._f = mlp(dim, hidden_dim, dim, layers, activation)
    self._f=CNN1_Net()
    self.avg_P = torch.tensor(avg_P)  # average power constraint
    self.peak = peak  # peak constraint  
    if self.peak is not None:
      print('here')
      self.peak_activation = PeakConstraint(peak)

  def forward(self, x):
    batch_size = x.size(0)
    unnorm_tx = self._f(x)

    norm_tx = unnorm_tx/torch.sqrt(torch.mean(torch.pow(unnorm_tx,2.0)))*torch.sqrt(self.avg_P)

    if self.peak is not None:
      norm_tx = self.peak_activation(norm_tx)

    return norm_tx

class AWGN_Chan(nn.Module):
  """AWGN Channel """
  def __init__(self, stdev=1.0):
    super(AWGN_Chan, self).__init__()
    self.stdev = torch.tensor(stdev)

  def forward(self, x):
    noise = torch.randn_like(x) * self.stdev
    return x + noise

critic_params = {
    'dim': 1,
    'layers': 4,
    'embed_dim': 32,
    'hidden_dim': 256,
    'activation': 'relu',
    'ref_batch_factor': 10,
    'learning_rate': 0.00001
}

nit_params = {'dim': 1,
              'layers': 4,
              'hidden_dim': 256,
              'activation': 'relu',
              'tx_power': 1.0,
              'learning_rate': 0.000001,
              'peak_amp': None }

estimator = 'smile'
clip = 100
init_epoch = 9000
max_epoch = 10
itr_every_nit = 2
itr_every_mi = 5
batch_size = 512
seed_rv_std = 1

if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"

device = torch.device(dev)
all_snr = [1, 2, 5, 10, 15, 20]
all_snr=[10,100,200,500]
all_snr=[10]
actual_cap = []
est_cap = []
for snr in all_snr:
 



  nit = NIT(nit_params['dim'], nit_params['hidden_dim'],
            nit_params['layers'], nit_params['activation'], 
            nit_params['tx_power'], peak=nit_params['peak_amp'])
 # nit.to(device)
  loss1=nn.BCEWithLogitsLoss()
  loss1.to(device)
  opt_nit = torch.optim.Adam(nit.parameters(), lr=nit_params['learning_rate'])

  awgn = AWGN_Chan()
  awgn.to(device)

  mi_est_loss = MI_Est_Losses('dv', device)
  mi_neuralnet = ConcatCritic1(critic_params['dim'], critic_params['hidden_dim'],
                              critic_params['layers'], critic_params['activation'])
  linlayer=Linear1_Net()
  mi_neuralnet.to(device)
  linlayer.to(device)

  opt_mi1 = torch.optim.Adam(linlayer.parameters(), lr=critic_params['learning_rate'])

  estimates = np.zeros(init_epoch)
  
  batch_x_ev = torch.sqrt(torch.tensor(snr))*torch.tensor(np.random.randn(batch_size,1),dtype=torch.float,requires_grad=True).to(device)
  batch_y_ev = awgn(batch_x_ev)
  #batch_y_ev=2*torch.tensor(np.random.randn(batch_size,1),dtype=torch.float,requires_grad=True).to(device)
  joint_ev=torch.cat((batch_x_ev,batch_y_ev),1)
  index0_ev=torch.randperm(batch_size)
  index1_ev=torch.randperm(batch_size)
  indep0_ev=batch_x_ev[index0_ev]
  indep1_ev=batch_y_ev[index1_ev]
      #  joint_pred0=joint[:,0]
      #  joint_pred1=joint[:,1]
      #  indep0=joint_pred0.view(-1)[index0].view(joint_pred0.size())
      #  indep1=joint_pred1.view(-1)[index1].view(joint_pred0.size())
  indep0_ev=indep0_ev.view(batch_size,1)
  indep1_ev=indep1_ev.view(batch_size,1)
  indep_ev=torch.cat((indep0_ev,indep1_ev),1)
  data1_ev=torch.vstack((joint_ev,indep_ev))
  label1_ev=torch.vstack((torch.ones(batch_size,1),torch.zeros(batch_size,1))).to(device)
  ######### train-set
  batch_x = torch.sqrt(torch.tensor(snr))*torch.tensor(np.random.randn(batch_size,1),dtype=torch.float,requires_grad=True).to(device)#batch_s#nit(batch_s)
  batch_y = awgn(batch_x)
 # batch_y=2*torch.tensor(np.random.randn(batch_size,1),dtype=torch.float,requires_grad=True).to(device)
  joint=torch.cat((batch_x,batch_y),1)
  index0=torch.randperm(batch_size)
  index1=torch.randperm(batch_size)
  indep0=batch_x[index0]
  indep1=batch_y[index1]
  #  joint_pred0=joint[:,0]
  #  joint_pred1=joint[:,1]
  #  indep0=joint_pred0.view(-1)[index0].view(joint_pred0.size())
  #  indep1=joint_pred1.view(-1)[index1].view(joint_pred0.size())
  indep0=indep0.view(batch_size,1)
  indep1=indep1.view(batch_size,1)
  indep=torch.cat((indep0,indep1),1)
  data1=torch.vstack((joint,indep))
  label1=torch.vstack((torch.ones(batch_size,1),torch.zeros(batch_size,1))).to(device)
    
  indx2=torch.randperm(2*batch_size)
  data_in=data1[indx2,0:2]##all indep and joint permuted
  data_in=torch.tensor(data_in,requires_grad=True)
    
  label_in=label1[indx2,0:2]### all indep and joint label permuted
    #label_in=label_in.view(2*batch_size,1)

  for e in tqdm(range(init_epoch)):
   ######### train-set
    batch_x = torch.sqrt(torch.tensor(snr))*torch.tensor(np.random.randn(batch_size,1),dtype=torch.float,requires_grad=True).to(device)#batch_s#nit(batch_s)
    batch_y = awgn(batch_x)
  # batch_y=2*torch.tensor(np.random.randn(batch_size,1),dtype=torch.float,requires_grad=True).to(device)
    joint=torch.cat((batch_x,batch_y),1)
    index0=torch.randperm(batch_size)
    index1=torch.randperm(batch_size)
    indep0=batch_x[index0]
    indep1=batch_y[index1]
    #  joint_pred0=joint[:,0]
    #  joint_pred1=joint[:,1]
    #  indep0=joint_pred0.view(-1)[index0].view(joint_pred0.size())
    #  indep1=joint_pred1.view(-1)[index1].view(joint_pred0.size())
    indep0=indep0.view(batch_size,1)
    indep1=indep1.view(batch_size,1)
    indep=torch.cat((indep0,indep1),1)
    data1=torch.vstack((joint,indep))
    label1=torch.vstack((torch.ones(batch_size,1),torch.zeros(batch_size,1))).to(device)
      
    indx2=torch.randperm(2*batch_size)
    data_in=data1[indx2,0:2]##all indep and joint permuted
    data_in=torch.tensor(data_in,requires_grad=True)
      
    label_in=label1[indx2,0:2]### all indep and joint label permuted
      #label_in=label_in.view(2*batch_size,1)
    indx2=torch.randperm(2*batch_size)
    data_in=data1[indx2,0:2]##all indep and joint permuted
    data_in=torch.tensor(data_in,requires_grad=True)
    
    label_in=label1[indx2,0:2]### all indep and joint label permuted
    #label_in=label_in.view(2*batch_size,1)
    prob,logit = linlayer(data_in)
    loss=torch.mean(loss1(logit,label_in))
    #mi_est = mi_est_loss.mi_est_loss(t_xy)
    #loss = -mi_est
  #  opt_mi1.zero_grad()
    loss.backward()
    #torch.nn.utils.clip_grad_norm_(mi_neuralnet.parameters(), 0.2)
    opt_mi1.step()
    
    mi_est = loss.detach().cpu().numpy()
    estimates[e]= mi_est
    if e%100==1:
       
       prob,logit=linlayer(data1_ev)
       prob1=prob.round()
       correct_pred=torch.eq(prob1,label1_ev)
       correct_pred=torch.tensor(correct_pred,dtype=torch.float)
       accuracy=torch.mean(correct_pred)
       print('accuracy for snr {} in iteration {} is {}'.format(snr,e,accuracy))
       
  prob_indep,logit1=linlayer(indep_ev)
  prob_joint,logit2=linlayer(joint_ev)
  lkh_indep=prob_indep/(1-prob_indep)
  lkh_joint=prob_joint/(1-prob_joint)
  #out_indep=torch.log(torch.abs(lkh_indep))
  out_indep=lkh_indep
  out_joint=torch.log(torch.abs(lkh_joint))
  



    


  plt.plot(estimates)
  plt.show()
  mi_est=torch.mean(out_joint)-torch.log(torch.mean(out_indep))
  cap=.5*np.log(1+snr)
  print('the estimated MI By CCMI is {}:'.format(mi_est))
  print('the actual MI is {} :'.format(cap))

  estimates = np.zeros(max_epoch)
  for e in tqdm(range(max_epoch)):
     for i in range(itr_every_nit):
       opt_nit.zero_grad()
       opt_mi.zero_grad()
  #     ##
       seed_rv = np.random.randn(batch_size,1,batch_size)*seed_rv_std
       batch_s = torch.tensor(seed_rv, dtype=torch.float).to(device) 
       batch_x = nit(batch_s)
       batch_y = awgn(batch_x)
  #     t_xy = mi_neuralnet(batch_x, batch_y)##
  #     mi_est = mi_est_loss.mi_est_loss(t_xy)
  #     loss = -mi_est
  #     loss.backward()
  #     #torch.nn.utils.clip_grad_norm_(nit.parameters(), 0.2)
  #     opt_nit.step()
  #     #mi_est = mi_est.detach().cpu().numpy()
    
  #   for i in range(itr_every_mi):
  #     opt_nit.zero_grad()
  #     opt_mi.zero_grad()
  #     seed_rv = np.random.randn(batch_size,1,batch_size)*seed_rv_std
  #     batch_s = torch.tensor(seed_rv, dtype=torch.float).to(device) 
  #     batch_x = nit(batch_s)
  #     batch_y = awgn(batch_x)
  #     t_xy = mi_neuralnet(batch_x, batch_y)
  #     mi_est = mi_est_loss.mi_est_loss(t_xy)
  #     loss = -mi_est
  #     loss.backward()
  #     ##
          
  #     #torch.nn.utils.clip_grad_norm_(mi_neuralnet.parameters(), 0.2)
  #     opt_mi.step()
  #     mi_est = mi_est.detach().cpu().numpy()

  #     estimates[e]= mi_est

  # plt.plot(estimates)
  # plt.show()

  # opt_nit.zero_grad()
  # opt_mi.zero_grad() 
  # seed_rv = np.random.randn(10000,1,batch_size)*seed_rv_std
  # batch_s = torch.tensor(seed_rv, dtype=torch.float).to(device) 
  # batch_x = nit(batch_s)
  # batch_x = batch_x.detach().cpu().numpy()
  # print("The variance of batch is {}".format(np.var(batch_x)))
  # print("-Estimated Capacity {:.4f}".format(np.mean(estimates[-50:])))
  # plt.hist(batch_x, 100)
  # plt.show()
  # est_cap.append(np.mean(estimates[-50:]))

# plt.plot(all_snr,actual_cap, '-k', label='Actual Capacity')
# plt.plot(all_snr,est_cap, '--r', label='SMILE EST Capacity')
# plt.legend()
# plt.xlabel('SNR (dB)')
# plt.ylabel('Capacity (nats)')
# plt.show()

prob_joint

batch_x.shape,batch_y.shape,mi_est

.5*np.log(1+4)

er=batch_x.reshape(10000,1)
plt.hist(er, 100)
#%matplotlib inline

plt.xlabel('alphabets')
plt.ylabel('density')
from google.colab import files
plt.savefig("smileawgnmid.pdf")
files.download("smileawgnmid.pdf")

batch_x.reshape(10000,1)

re=torch.cat((batch_x,batch_y),1)
re.shape
indx=torch.randperm(batch_size)
indx1=torch.randperm(batch_size)
dd=(re[:,1,0])
sc=dd.view(-1)[indx].view(dd.size())
ko=re[indx,0:2,0]
ko.shape

label1.shape,data_in.shape

indep1.view(256,1,1).shape

joint.shape,indep0.shape,indep1.shape

correct_pred,prob1.shape

torch.equal(prob1,label1)

indep_ev.shape,joint_ev.shape

out_indep.shape,out_joint.shape

torch.logsumexp(out_indep,0)#,out_joint)

torch.mean

torch.mean(out_indep)-torch.logsumexp(out_joint,1),torch.mean(out_joint)-torch.logsumexp(out_indep,0)



out_indep

w=str([-3.418482971191406250e+01,0.000000000000000000e+00
-3.381885147094726562e+01,1.000000000000000000e+00
-3.345286941528320312e+01,0.000000000000000000e+00
-3.308689117431640625e+01,0.000000000000000000e+00
-3.272091293334960938e+01,0.000000000000000000e+00
-3.235493087768554688e+01,0.000000000000000000e+00
-3.198895072937011719e+01,1.000000000000000000e+00
-3.162297248840332031e+01,2.000000000000000000e+00
-3.125699234008789062e+01,3.000000000000000000e+00
-3.089101219177246094e+01,1.000000000000000000e+00
-3.052503204345703125e+01,2.000000000000000000e+00
-3.015905189514160156e+01,1.000000000000000000e+00
-2.979307174682617188e+01,1.000000000000000000e+00
-2.942709159851074219e+01,0.000000000000000000e+00
-2.906111145019531250e+01,3.000000000000000000e+00
-2.869513320922851562e+01,0.000000000000000000e+00
-2.832915306091308594e+01,2.000000000000000000e+00
-2.796317291259765625e+01,2.000000000000000000e+00
-2.759719276428222656e+01,3.000000000000000000e+00
-2.723121261596679688e+01,4.000000000000000000e+00
-2.686523246765136719e+01,3.000000000000000000e+00
-2.649925231933593750e+01,1.000000000000000000e+00
-2.613327407836914062e+01,3.000000000000000000e+00
-2.576729393005371094e+01,5.000000000000000000e+00
-2.540131378173828125e+01,8.000000000000000000e+00
-2.503533363342285156e+01,4.000000000000000000e+00
-2.466935348510742188e+01,3.000000000000000000e+00
-2.430337333679199219e+01,6.000000000000000000e+00
-2.393739318847656250e+01,4.000000000000000000e+00
-2.357141494750976562e+01,4.000000000000000000e+00
-2.320543479919433594e+01,5.000000000000000000e+00
-2.283945465087890625e+01,1.200000000000000000e+01
-2.247347450256347656e+01,5.000000000000000000e+00
-2.210749435424804688e+01,1.500000000000000000e+01
-2.174151420593261719e+01,1.200000000000000000e+01
-2.137553405761718750e+01,1.700000000000000000e+01
-2.100955581665039062e+01,1.500000000000000000e+01
-2.064357566833496094e+01,1.900000000000000000e+01
-2.027759552001953125e+01,1.800000000000000000e+01
-1.991161537170410156e+01,1.800000000000000000e+01
-1.954563522338867188e+01,1.500000000000000000e+01
-1.917965507507324219e+01,1.300000000000000000e+01
-1.881367492675781250e+01,2.400000000000000000e+01
-1.844769668579101562e+01,1.500000000000000000e+01
-1.808171653747558594e+01,1.900000000000000000e+01
-1.771573638916015625e+01,2.200000000000000000e+01
-1.734975624084472656e+01,2.400000000000000000e+01
-1.698377609252929688e+01,2.800000000000000000e+01
-1.661779594421386719e+01,2.100000000000000000e+01
-1.625181579589843750e+01,2.800000000000000000e+01
-1.588583660125732422e+01,3.200000000000000000e+01
-1.551985645294189453e+01,3.200000000000000000e+01
-1.515387725830078125e+01,4.300000000000000000e+01
-1.478789710998535156e+01,3.400000000000000000e+01
-1.442191696166992188e+01,3.900000000000000000e+01
-1.405593681335449219e+01,4.300000000000000000e+01
-1.368995761871337891e+01,4.700000000000000000e+01
-1.332397747039794922e+01,4.600000000000000000e+01
-1.295799732208251953e+01,4.500000000000000000e+01
-1.259201812744140625e+01,5.900000000000000000e+01
-1.222603797912597656e+01,6.100000000000000000e+01
-1.186005783081054688e+01,5.800000000000000000e+01
-1.149407768249511719e+01,6.600000000000000000e+01
-1.112809848785400391e+01,6.400000000000000000e+01
-1.076211833953857422e+01,7.300000000000000000e+01
-1.039613819122314453e+01,8.300000000000000000e+01
-1.003015899658203125e+01,7.500000000000000000e+01
-9.664178848266601562e+00,7.300000000000000000e+01
-9.298198699951171875e+00,1.010000000000000000e+02
-8.932218551635742188e+00,8.100000000000000000e+01
-8.566239356994628906e+00,8.100000000000000000e+01
-8.200259208679199219e+00,9.400000000000000000e+01
-7.834279060363769531e+00,8.500000000000000000e+01
-7.468299388885498047e+00,9.500000000000000000e+01
-7.102319717407226562e+00,1.050000000000000000e+02
-6.736339569091796875e+00,1.060000000000000000e+02
-6.370359897613525391e+00,1.100000000000000000e+02
-6.004379749298095703e+00,1.080000000000000000e+02
-5.638400077819824219e+00,1.040000000000000000e+02
-5.272419929504394531e+00,1.310000000000000000e+02
-4.906440258026123047e+00,1.070000000000000000e+02
-4.540460586547851562e+00,1.420000000000000000e+02
-4.174480438232421875e+00,1.580000000000000000e+02
-3.808500528335571289e+00,1.330000000000000000e+02
-3.442520618438720703e+00,1.460000000000000000e+02
-3.076540946960449219e+00,1.490000000000000000e+02
-2.710561037063598633e+00,1.470000000000000000e+02
-2.344581127166748047e+00,1.620000000000000000e+02
-1.978601217269897461e+00,1.690000000000000000e+02
-1.612621307373046875e+00,1.360000000000000000e+02
-1.246641397476196289e+00,1.320000000000000000e+02
-8.806615471839904785e-01,1.370000000000000000e+02
-5.146816372871398926e-01,1.530000000000000000e+02
-1.487017869949340820e-01,1.350000000000000000e+02
+2.172780930995941162e-01,1.550000000000000000e+02
+5.832579731941223145e-01,1.480000000000000000e+02
+9.492378830909729004e-01,1.380000000000000000e+02
+1.315217733383178711e+00,1.400000000000000000e+02
+1.681197643280029297e+00,1.400000000000000000e+02
+2.047177553176879883e+00,1.450000000000000000e+02
+2.413157463073730469e+00,1.360000000000000000e+02
+2.779137372970581055e+00,1.550000000000000000e+02
+3.145117044448852539e+00,1.600000000000000000e+02
+3.511096954345703125e+00,1.440000000000000000e+02
+3.877076864242553711e+00,1.500000000000000000e+02
+4.243056774139404297e+00,1.260000000000000000e+02
+4.609036445617675781e+00,1.320000000000000000e+02
+4.975016593933105469e+00,1.230000000000000000e+02
+5.340996265411376953e+00,1.280000000000000000e+02
+5.706976413726806641e+00,1.230000000000000000e+02
+6.072956085205078125e+00,1.230000000000000000e+02
+6.438936233520507812e+00,1.240000000000000000e+02
+6.804915904998779297e+00,1.390000000000000000e+02
+7.170896053314208984e+00,1.390000000000000000e+02
+7.536875724792480469e+00,1.200000000000000000e+02
+7.902855396270751953e+00,1.190000000000000000e+02
+8.268835067749023438e+00,8.600000000000000000e+01
+8.634815216064453125e+00,1.170000000000000000e+02
+9.000795364379882812e+00,9.700000000000000000e+01
+9.366775512695312500e+00,1.100000000000000000e+02
+9.732754707336425781e+00,1.040000000000000000e+02
+1.009873485565185547e+01,1.090000000000000000e+02
+1.046471500396728516e+01,9.900000000000000000e+01
+1.083069419860839844e+01,8.300000000000000000e+01
+1.156265449523925781e+01,7.800000000000000000e+01
+1.192863464355468750e+01,8.900000000000000000e+01
+1.229461383819580078e+01,8.900000000000000000e+01
+1.266059398651123047e+01,7.500000000000000000e+01
+1.302657413482666016e+01,7.500000000000000000e+01
+1.339255428314208984e+01,7.700000000000000000e+01
+1.375853347778320312e+01,6.900000000000000000e+01
+1.412451362609863281e+01,6.500000000000000000e+01
+1.449049377441406250e+01,6.600000000000000000e+01
+1.485647296905517578e+01,6.400000000000000000e+01
+1.522245311737060547e+01,4.700000000000000000e+01
+1.558843326568603516e+01,4.000000000000000000e+01
+1.595441341400146484e+01,4.500000000000000000e+01
+1.632039260864257812e+01,4.000000000000000000e+01
+1.668637275695800781e+01,4.400000000000000000e+01
+1.705235290527343750e+01,3.400000000000000000e+01
+1.741833305358886719e+01,3.500000000000000000e+01
+1.778431320190429688e+01,3.400000000000000000e+01
+1.815029144287109375e+01,3.400000000000000000e+01
+1.851627159118652344e+01,2.700000000000000000e+01
+1.888225173950195312e+01,1.900000000000000000e+01
+1.924823188781738281e+01,2.200000000000000000e+01
+1.961421203613281250e+01,1.900000000000000000e+01
+1.998019218444824219e+01,2.300000000000000000e+01
+2.034617233276367188e+01,2.000000000000000000e+01
+2.071215057373046875e+01,1.300000000000000000e+01
+2.107813072204589844e+01,2.200000000000000000e+01
+2.144411087036132812e+01,1.500000000000000000e+01
+2.181009101867675781e+01,1.000000000000000000e+01
+2.217607116699218750e+01,1.400000000000000000e+01
+2.254205131530761719e+01,2.000000000000000000e+01
+2.290803146362304688e+01,9.000000000000000000e+00
+2.327400970458984375e+01,1.100000000000000000e+01
+2.363998985290527344e+01,9.000000000000000000e+00
+2.400597000122070312e+01,1.000000000000000000e+01
+2.437195014953613281e+01,1.000000000000000000e+01
+2.473793029785156250e+01,1.700000000000000000e+01
+2.510391044616699219e+01,8.000000000000000000e+00
+2.546989059448242188e+01,7.000000000000000000e+00
+2.583586883544921875e+01,9.000000000000000000e+00
+2.620184898376464844e+01,8.000000000000000000e+00
+2.656782913208007812e+01,4.000000000000000000e+00
+2.693380928039550781e+01,6.000000000000000000e+00
+2.729978942871093750e+01,1.000000000000000000e+00
+2.766576957702636719e+01,8.000000000000000000e+00
+2.803174972534179688e+01,3.000000000000000000e+00
+2.839772987365722656e+01,6.000000000000000000e+00
+2.876370811462402344e+01,6.000000000000000000e+00
+2.912968826293945312e+01,4.000000000000000000e+00
+2.949566841125488281e+01,4.000000000000000000e+00
+2.986164855957031250e+01,6.000000000000000000e+00
+3.022762870788574219e+01,7.000000000000000000e+00
+3.059360885620117188e+01,1.000000000000000000e+00
+3.095958900451660156e+01,6.000000000000000000e+00
+3.132556724548339844e+01,2.000000000000000000e+00
+3.169154739379882812e+01,1.000000000000000000e+00
+3.205752944946289062e+01,3.000000000000000000e+00
+3.242350769042968750e+01,2.000000000000000000e+00
+3.278948593139648438e+01,2.000000000000000000e+00
+3.315546798706054688e+01,0.000000000000000000e+00
+3.352144622802734375e+01,2.000000000000000000e+00
+3.388742828369140625e+01,1.000000000000000000e+00
+3.425340652465820312e+01,1.000000000000000000e+00
+3.461938858032226562e+01,0.000000000000000000e+00
+3.498536682128906250e+01,2.000000000000000000e+00])

import numpy as np

w1=np.array(w)
w1.shape

np.histogram(w1)

w[2]



df = pd.read_csv('/content/drive/MyDrive/Data_CSV/mine-awgn-20.csv')
ass=np.array(df)
#print(df.to_string()) 
print(ass[193][1])
