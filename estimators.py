# -*- coding: utf-8 -*-


import os
import h5py
import glob, os
import numpy as np
import matplotlib.pyplot as plt
from numpy import mean
from numpy import std
#from sklearn.model_selection import KFold
#from sklearn.model_selection import cross_val_score
import torch
import torch.nn as nn
from tqdm.auto import tqdm, trange
from numpy.random import default_rng
import torch.nn.functional as F


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

   def reverse_jensen(self,f):
      c=torch.tensor(5.0)
      thr=torch.tensor(1e-14)
      joint=f.diag().mean()
      marge1= f - torch.diag(np.inf * torch.ones(batch_size).to(self.device))                                 
      marge2=(2*marge1).exp().mean()                                                                           
      marge3=torch.maximum(marge1.exp().mean(),thr)
      tot= c/(1-torch.sqrt(1/c)) *(marge2/(torch.maximum(marge3**2,thr)))                                                                                                   
      marge4=tot*(torch.log(1+marge1.exp()).mean())                                                                                                          
      marge5=torch.log(torch.maximum(marge3/(1+c*marge2/marge3),thr))
      marge5=marge4+marge5
      return joint-marge5         
  def nwj_lower_bound(self, scores):
      return self.tuba_lower_bound(scores - 1.)


  def infonce_lower_bound(scores):
      nll = scores.diag().mean() - scores.logsumexp(dim=1)
    
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

      return first_term - second_term - third_term_grad + third_term_no_grad#, buffer_update


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
      
      ref = ref_vec
        
      batch_size = ref.size(0)
      z = log_mean_ef_ref = torch.logsumexp(ref, dim=(0, 1)) - torch.log(torch.tensor(batch_size)).to(self.device)
      
      z = log_mean_ef_ref = torch.mean(torch.exp(ref_vec))/b - torch.log(b) - 1
      
      dv = vec.mean() - z
      js = self._ent_js_fgan_lower_bound(vec, ref_vec)

      with torch.no_grad():
          dv_js = dv - js
      
      return js + dv_js

  def chi_square_lower_bound(self, f, beta=.9): 
      t_xy, t_xy_ref = f

      d_xy = self.chisquare_pred(t_xy, t_xy_ref, beta=beta)
    
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
      elif self.estimator=='mine':
          mi=self.mine_lower_bound(net_output, **kwargs)
      elif self.estimator == 'tuba':
    
          
          mi = self.tuba_lower_bound(net_output, **kwargs)
      elif self.estimator=="reverse-jensen":
          mi=self.reverse_jensen(net_output,**kwargs)
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
