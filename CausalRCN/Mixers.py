import torch
from torch import nn as nn
from transformers.activations import ACT2FN
import torch.nn.functional as F
from typing import Optional
# from .kan import KAN, KANLinear
from .fasterkan import FastKAN as KAN
from .faster_kan_conv import FastKANConv1DLayer as KCN1d
import os 
import math
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_CUDA_ARCH_LIST'] = "7.0"
from torch.utils.cpp_extension import load
from sru import SRUCell

class Causal_RCNN(nn.Module):
    # version 1
    def __init__(self, input_size, hidden_size, kernel_size, drop_rate):
        super().__init__()
        self.sru = SRUCell(input_size, hidden_size,
                           dropout=drop_rate,      
                           bidirectional=False,   
                           layer_norm=True,      
                           highway_bias=-2, 
                           rescale=True,
                           use_tanh=True)
        self.cnn = nn.Conv1d(hidden_size, hidden_size, kernel_size)
        self.pad = nn.ConstantPad1d((kernel_size - 1, 0), 0)
        # self.output = nn.Linear(hidden_size, hidden_size)
        self.norm = RMSNorm(hidden_size)
        # self.bn = nn.BatchNorm1d(hidden_size)
        self.lambda_weight = nn.Parameter(torch.randn(hidden_size))

        self.drop_out = nn.Dropout(drop_rate)

    def forward(self, x):
        pad_input = self.pad(x.transpose(1, 2))
        cnn_output = self.cnn(pad_input).permute(2, 0, 1)
        output_states, c_states = self.sru(cnn_output)
        output_states = self.lambda_weight * cnn_output + (1 - self.lambda_weight) * output_states
        output_states = output_states.transpose(0, 1)
        output_states = self.norm(output_states)
        output_states = self.drop_out(output_states)
        return output_states
    

class CausalRCNBlock(nn.Module):
    def __init__(self,
        hidden_dim: int, 
        window_size: int,
        activation: str,
        drop_rate: float = 0.5,
        use_rnn: bool = True,
        ffn_type: Optional[str] = None,
        *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.activate = ACT2FN[activation]
        self.use_rnn = use_rnn

        if ffn_type is None:
            self.tokenMixing = None
        elif ffn_type == "kan":
            self.tokenMixing = KAN([hidden_dim, hidden_dim*2, hidden_dim])
        elif ffn_type == "ffn":
            self.tokenMixing = CausalRCNTokenMixer(
                                    hidden_dim=hidden_dim, 
                                    internal_dim=hidden_dim * 2, 
                                    activation=activation,
                                    drop_rate=drop_rate)
        else:
            raise ValueError(f"{ffn_type} have not been supported in this version.")
        if self.use_rnn:
            self.memory_mixer = Causal_RCNN(
                hidden_dim, hidden_dim, kernel_size=window_size, drop_rate=drop_rate
            )
        else:
            self.memory_mixer = None
        self.drop_out = nn.Dropout(drop_rate)
    
    def forward(self, input_tensor: torch.tensor):
        if self.use_rnn:
            out = self.memory_mixer.forward(input_tensor) * input_tensor
        else:
            out = input_tensor
        if isinstance(self.tokenMixing, (KAN, CausalRCNTokenMixer)):
            out = self.tokenMixing.forward(out)+out
        return out


class CausalRCN(nn.Module):
    def __init__(self,
        n_layers: int,
        hidden_dim: int,
        window_size: int,
        activation: str,
        drop_rate: float = 0.5,
        use_rnn: bool = True,
        ffn_type: Optional[str] = None,
        *args, **kwargs) -> None:
        super(CausalRCN, self).__init__(*args, **kwargs)

        self.CausalRCNBlocks = nn.ModuleList(
            CausalRCNBlock(
                hidden_dim=hidden_dim, 
                window_size=window_size,
                activation=activation, 
                drop_rate=drop_rate,
                use_rnn=use_rnn,
                ffn_type=ffn_type
                ) for i in range(n_layers)
        )
        
    def forward(self,
            bottleneck: torch.Tensor): 
        aux = []
        # [length, batch_size, hidden]
        token_mixing = bottleneck
        for step, layer in enumerate(self.CausalRCNBlocks):
            token_mixing = layer(token_mixing)
            if step % 8 == 0:
                aux.append(token_mixing)
        rfft_tokens = token_mixing
        outputs = rfft_tokens
        return outputs, aux
    

class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
    
    def _norm(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        return hidden_states * torch.rsqrt(variance + self.eps)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.weight * self._norm(hidden_states.float()).type_as(hidden_states)


class CausalRCNTokenMixer(nn.Module):
    def __init__(self, 
        hidden_dim: torch.Tensor, 
        internal_dim: torch.Tensor, 
        activation: str,
        drop_rate: float,
        *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.activate = ACT2FN[activation]
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.drop_out = nn.Dropout(drop_rate)
        self.linear1 = nn.Linear(hidden_dim, internal_dim, bias=False)
        self.ln2 = nn.LayerNorm(internal_dim)
        self.linear2 = nn.Linear(internal_dim, hidden_dim)
        
    def forward(self, input_: torch.Tensor):
        token_mixing = self.linear1(self.activate(self.drop_out(self.ln1(input_))))
        token_mixing = self.linear2(self.activate(self.drop_out(self.ln2(token_mixing))))
        return token_mixing
    

class CNNblock(nn.Module):
    def __init__(self, kernel_size: int, hidden_size: int, drop_rate: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pad = nn.ConstantPad1d((kernel_size - 1, 0), 0)
        self.cnn = nn.Conv1d(hidden_size, hidden_size, kernel_size)
        self.output = nn.Linear(hidden_size, hidden_size, bias=False)
        self.drop_out = nn.Dropout(drop_rate)
        self.norm = nn.LayerNorm(hidden_size)
    
    def forward(self, inputs: torch.tensor):
        pad_input = self.pad(inputs.transpose(1, 2))
        cnn_output = self.cnn(pad_input).transpose(1, 2)
        output = self.drop_out(cnn_output)
        output = self.output(output)
        output = self.norm(output)
        return output * inputs



class Causal_CNN(nn.Module):
    def __init__(self,
                n_layers: int,
                hidden_dim: int,
                window_size: int,
                activation: str,
                drop_rate: float = 0.5,
          *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hidden_dim = hidden_dim
        self.window_size = window_size
        self.cnn = nn.ModuleList(
            [CNNblock(window_size, hidden_dim, drop_rate) for i in range(n_layers)]
        )
    
    def forward(self, inputs: torch.tensor):
        outputs = inputs
        for layer in self.cnn:
            outputs = layer(outputs)
        return outputs, None
