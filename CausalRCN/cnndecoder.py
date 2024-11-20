import torch
from torch import nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from typing import Optional, List, Dict
from fft_conv_pytorch import FFTConv1d
from dataclasses import dataclass, asdict, field


class CNNDecoder(nn.Module):
    def __init__(self, hidden_dim: int, internal_dim: int, kernel_size:List[int], padding: List[int], *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv_layers = nn.Sequential(
            nn.Conv1d(hidden_dim, internal_dim, kernel_size[0], padding=padding[0]),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size[1], 1, padding=padding[1]),
            nn.BatchNorm1d(internal_dim),
            # nn.ReLU(),
            nn.Conv1d(internal_dim, hidden_dim, kernel_size[1], padding=padding[1]),
            nn.ReLU(),
        )
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)


    def forward(self, tensor: torch.tensor):
        resdual = tensor
        outputs = self.conv_layers(tensor.transpose(1, 2)).transpose(1, 2)
        gate_outputs = F.sigmoid(self.linear(resdual))
        add_and_norm = self.ln(outputs + gate_outputs)
        return add_and_norm


class CNNMixer(nn.Module):
    def __init__(self, n_layers: int, hidden_dim: int, internal_dim: int, kernel_size: List[int], padding: List[int], *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        print(n_layers, hidden_dim, internal_dim, kernel_size, padding)
        self.cnn_decoders = nn.ModuleList(
            CNNDecoder(hidden_dim, internal_dim, kernel_size, padding) for i in range(n_layers)
        )
    
    def forward(self, tensors: torch.tensor):
        for layer in self.cnn_decoders:
            tensors = layer(tensors)
        return tensors


@dataclass
class CNNMixerConfig:
    n_layers: int = 12
    hidden_dim: int = 512
    internal_dim: int = 768
    kernel_size: list = field(default_factory=list)
    padding: list = field(default_factory=list)

import time

if __name__ == "__main__":
    start_time = time.time()
    cmconf = CNNMixerConfig()
    cmconf.kernel_size = [3, 5]
    cmconf.padding = [1, 2]
    kwargs = asdict(cmconf)
    model = CNNMixer(**kwargs).cuda()
    input_tensors = torch.randn((64, 256, 512)).cuda()
    outputs = model(input_tensors)
    end_time = time.time()
    print(outputs.shape)
    elapsed_time = (end_time - start_time) * 1000  # 转换为毫秒
    print(f"Elapsed time: {elapsed_time:.2f} ms")
