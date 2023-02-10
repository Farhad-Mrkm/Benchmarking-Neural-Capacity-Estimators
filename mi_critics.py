# -*- coding: utf-8 -*-


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
