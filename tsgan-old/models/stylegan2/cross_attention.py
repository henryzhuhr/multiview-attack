# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import nn

import datetime



class CrossAttention(nn.Module):
    r"""
    A cross attention layer.

    Parameters:
        query_dim (`int`):                              The number of channels in the query.
        cross_attention_dim (`int`, *optional*):        The number of channels in the encoder_hidden_states. If not given, defaults to `query_dim`.
        heads (`int`,  *optional*, defaults to 8):      The number of heads to use for multi-head attention.
        dim_head (`int`,  *optional*, defaults to 64):  The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        bias (`bool`, *optional*, defaults to False):   Set to `True` for the query, key, and value linear layers to contain a bias parameter.
    """
    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias=False,
        upcast_softmax: bool = False,
        added_kv_proj_dim: Optional[int] = None,
        norm_num_groups: Optional[int] = None,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.upcast_softmax = upcast_softmax

        self.scale = dim_head**-0.5

        self.heads = heads

        if norm_num_groups is not None:
            self.group_norm = nn.GroupNorm(num_channels=inner_dim, num_groups=norm_num_groups, eps=1e-5, affine=True)
        else:
            self.group_norm = None

        self.to_q = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_k = nn.Linear(cross_attention_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(cross_attention_dim, inner_dim, bias=bias)

        if added_kv_proj_dim is not None:
            self.add_k_proj = nn.Linear(added_kv_proj_dim, cross_attention_dim)
            self.add_v_proj = nn.Linear(added_kv_proj_dim, cross_attention_dim)

        self.to_out = nn.Sequential(*[
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout),
        ])

    def prepare_attention_mask(self, attention_mask: torch.Tensor, target_length: int):
        head_size = self.heads
        if attention_mask is None:
            return attention_mask

        if attention_mask.shape[-1] != target_length:
            attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
            attention_mask = attention_mask.repeat_interleave(head_size, dim=0)
        return attention_mask

    def head_to_batch_dim(self, x: torch.Tensor):
        B, C = x.shape
        x = x.reshape(B, self.heads, C // self.heads)
        # x = x.permute(0, 2, 1).reshape(B * num_head, C // num_head)
        return x

    def batch_to_head_dim(self, x: torch.Tensor):
        B, H, C = x.shape
        x = x.reshape(B, H * C)
        return x

    def forward(
        self,
        latent: torch.Tensor,
        cond_latent: torch.Tensor ,
    ):
        B, C = latent.shape
        # print(logheader(), "Q", Q.size())

        # Q, K, V projexction
        Q = self.head_to_batch_dim(self.to_q(latent))
        K = self.head_to_batch_dim(self.to_k(cond_latent))
        V = self.head_to_batch_dim(self.to_v(cond_latent))

        attention_scores = torch.baddbmm(
            torch.empty(Q.size(0), Q.size(1), K.size(1), dtype=Q.dtype, device=Q.device),
            Q,
            K.transpose(-1, -2),
            beta=0,
            alpha=self.scale,
        )
        attention_probs = attention_scores.softmax(dim=-1).to(Q.dtype)
        multi_features = torch.bmm(attention_probs, V)
        features = self.batch_to_head_dim(multi_features)
        # torch.Size([32, 4, 64])
        # latent=torch.concat()
        features = self.to_out(features)
        return features

