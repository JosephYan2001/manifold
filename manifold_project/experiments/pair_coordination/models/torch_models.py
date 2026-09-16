"""PyTorch 训练模型；权重按原有 JSON 布局保存，兼容旧 checkpoint。"""
import copy
import numpy as np
import torch
from torch import nn


def resolve_device(name="auto"):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("device 请选择 auto、cpu 或 cuda[:编号]")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("当前 Python 环境的 CUDA 不可用，请检查 PyTorch 或使用 --device cpu")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError("CUDA 设备编号超出可用范围")
    return device


class TorchModel(nn.Module):
    """统一封装 MLP/类型表 actor 和方向函数，训练使用 float64。"""

    def __init__(self, reference, device="auto"):
        super().__init__()
        self.template = copy.deepcopy(reference.state())
        self.kind = self.template["kind"]
        self.is_actor = self.kind in ("mlp_actor", "binary_type_table")
        self.n_types = (reference.n_types if hasattr(reference, "n_types")
                        else len(reference.parameters))
        self.beta = getattr(reference, "beta", None)
        self.q_max = getattr(reference, "q_max", None)
        self.shapes = reference.network.shapes if hasattr(reference, "network") else None
        # 沿用初始化和权重排列，使旧模型可以精确导入。
        self.flat = nn.Parameter(torch.tensor(reference.parameters.copy(), dtype=torch.float64,
                                              device=resolve_device(device)))
        self.register_buffer("inputs", torch.eye(self.n_types, dtype=self.flat.dtype,
                                                 device=self.flat.device))

    def tensor(self, values, dtype=None):
        return torch.tensor(np.asarray(values), dtype=dtype or self.flat.dtype, device=self.flat.device)

    def tensor_logits(self):
        if self.shapes is None:
            return self.flat
        arrays, offset = [], 0
        for shape in self.shapes:
            count = int(np.prod(shape))
            arrays.append(self.flat[offset:offset+count].view(shape))
            offset += count
        value = self.inputs
        for i in range(0, len(arrays), 2):
            value = value @ arrays[i] + arrays[i+1]
            if i < len(arrays)-2:
                value = torch.tanh(value)
        return value[:, 0]

    def forward(self):
        z = self.tensor_logits()
        if self.is_actor:
            p = self.beta/2 + (1-self.beta)*torch.sigmoid(z)
            return torch.stack((1-p, p), dim=-1)
        t = self.q_max*torch.tanh(z)
        return torch.stack((-t, t), dim=-1)

    def table(self):
        with torch.no_grad():
            return self().cpu().numpy().copy()

    def probabilities(self, local_types):
        return self.table()[local_types]

    def copy(self):
        return copy.deepcopy(self)

    def state(self):
        state = copy.deepcopy(self.template)
        values = self.flat.detach().cpu().tolist()
        if self.shapes is not None:
            state["network"]["parameters"] = values
        else:
            state["logits" if self.is_actor else "parameters"] = values
        if not self.is_actor:
            state["table"] = self.table().tolist()
        return state
