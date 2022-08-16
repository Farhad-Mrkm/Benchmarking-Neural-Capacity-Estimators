# -*- coding: utf-8 -*-


#Author__Farhad_Mirkarimi -*- coding: utf-8 -*-
import os
import h5py
import glob, os
import numpy as np
import matplotlib.pyplot as plt

from numpy import mean
from numpy import std
import torch
import torch.nn as nn
from tqdm.auto import tqdm, trange
from numpy.random import default_rng
import torch.nn.functional as F
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
  def __init__(self, dim, hidden_dim, layers, activation, avg_P,chan_type, peak=None,positive=None,p1=3/7,p2=4/7, **extra_kwargs):
    super(NIT, self).__init__()
    self._f = mlp(dim, hidden_dim, dim, layers, activation)
    self.avg_P = torch.tensor(avg_P)  # average power constraint
    self.peak = peak  # peak constraint  
    self.positive=positive
    self.chan_type=chan_type
    self.p1=p1
    self.p2=p2
    if self.peak is not None:
      
      self.peak_activation = PeakConstraint(peak)

  def forward(self, x):
    if self.chan_type=='discrt_rician' :
      batch_size = x.size(0)
      unnorm_tx = self._f(x)

      norm_tx = unnorm_tx/torch.sqrt(torch.mean(torch.pow(unnorm_tx,2.0)))*torch.sqrt(self.avg_P)
      norm_tx=(norm_tx/torch.pow(torch.mean(torch.pow(norm_tx,4.0)),.25))*torch.sqrt(self.avg_P)*torch.tanh(norm_tx)

      if self.peak is not None:
        norm_tx = self.peak_activation(norm_tx)

      return norm_tx
    elif self.chan_type=='conts_awgn':
        unnorm_tx = self._f(x)

        norm_tx = unnorm_tx/torch.sqrt(torch.mean(torch.pow(unnorm_tx,2.0)))*torch.sqrt(self.avg_P)
        

        if self.peak is not None:
          norm_tx = self.peak_activation(norm_tx)
        if self.positive is not None:
          norm_tx=F.softplus(norm_tx)

        return norm_tx
    
    elif self.chan_type=='discrt_poisson':
      batch_size = x.size(0)
      unnorm_tx = self._f(x)
   # norm_tx = unnorm_tx/torch.sqrt(torch.mean(torch.pow(unnorm_tx,2.0)))*torch.sqrt(self.avg_P)
      norm_tx=(unnorm_tx/(torch.mean(unnorm_tx)))*(torch.sqrt(self.avg_P))
    #norm_tx=torch.tanh((unnorm_tx))*torch.sqrt(self.avg_P)
    #if self.channel == 'awgn':
      #norm_tx = unnorm_tx/torch.sqrt(torch.mean(torch.pow(unnorm_tx,2.0)))*torch.sqrt(self.avg_P)
    #elif self.channel == 'poisson':
      #unnorm_tx = torch.log10(torch.cosh(unnorm_tx))
      #unnorm_tx = torch.absolute(unnorm_tx)
      #norm_tx = unnorm_tx/torch.mean(unnorm_tx)*self.avg_P
      #norm_tx = torch.log10(torch.cosh(unnorm_tx))
    
      if self.positive is not None:
        norm_tx=(torch.cosh(norm_tx))-1.0
      #norm_tx=self.ps(norm_tx)
      if self.peak is not None:
        norm_tx=self.peak_activation(norm_tx)
      return norm_tx
  
      
    elif self.chan_type=='mac':
            
        batch_size = x.size(0)
        unnorm_tx = self._f(x)
        norm_tx = unnorm_tx/torch.sqrt(torch.mean(torch.pow(unnorm_tx,2.0)))*torch.sqrt(self.avg_P)

        if self.peak is not None:
          norm_tx = self.peak_activation(norm_tx)
        if self.positive is not None:
          norm_tx=F.softplus(norm_tx)
        norm_tx[:,1]=self.p1*norm_tx[:,1]
        norm_tx[:,0]=self.p2*norm_tx[:,0]
        

      
        return norm_tx
      

#class AWGN_Chan(nn.Module):
  #"""AWGN Channel """
  #def __init__(self, stdev=1.0):
   # super(AWGN_Chan, self).__init__()
    #self.stdev = torch.tensor(stdev)

  #def forward(self, x):
   # noise = torch.randn_like(x) * self.stdev
   # return x + noise
class _Channel(nn.Module):
  """AWGN Channel """
  def __init__(self,type1):
    super(_Channel, self).__init__()
    self.stdev = torch.tensor(1.0,dtype=torch.float)
    self.type1=type1
    ##
    self.dark_current =0#torch.tensor(params)
    self.delta0=10
    #self.stdev = torch.tensor(params/10000)
      #self.delta=torch.tensor(0.3839)
    self.delta=torch.tensor(1.0)
  def forward(self, x):
    if self.type1=='discrt_rician':
       noise = torch.randn_like(x) * self.stdev
      #fc=torch.randn_like(torch.transpose(x,0,1),dtype=torch.cfloat)
       fc=torch.randn_like(x)*(self.stdev)
       fc1=torch.ones_like(x)*.5
       fmid = torch.mul(fc,x)
       fmid1=torch.mul(fc1,x)
       y=fmid[:,0]-fmid[:,1]+noise[:,0]+fmid1[:,0]
       return y
    elif self.type1=='conts_awgn':
       noise = torch.randn_like(x) * self.stdev
       return x + noise
    elif self.type1=='discrt_poisson':
      y = torch.poisson(x*self.delta+self.dark_current*self.delta0)
      noise = torch.randn_like(x) * self.stdev
      return y#+noise
    elif self.type1=='mac':
      noise = torch.randn_like(x) * self.stdev
      return x[:,1] + x[:,0] + noise[:,0]
