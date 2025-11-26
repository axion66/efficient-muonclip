# Efficient MuonClip: Flash-Muon + QKClip
This repo implements an efficient version of MuonClip using triton. (https://arxiv.org/pdf/2507.20534)


## Usage

### Option 1: Using MuonClip (Recommended)

MuonClip combines Muon optimizer + QK-Clip for attention stability.
Note that unlike the paper, I removed the if statement and add a fallback of gamma=1.0 to avoid conditional-branching on multi-gpu setup (DDP).

```python
from muonclip import MuonClip
import torch

# Find >=2D parameters in the body of the network -- these should be optimized by Muon
muon_params = [p for p in model.body.parameters() if p.ndim >= 2]
# Find everything else -- these should be optimized by AdamW
adamw_params = ([p for p in model.body.parameters() if p.ndim < 2]
              + [*model.head.parameters(), *model.embed.parameters()])

muonclip = MuonClip(
    params=muon_params,
    attention_params=[model.body.layer1.attn, model.body.layer2.attn],  # Note: attention_params should be your attention layer modules.
    mode='mha',  # or 'mla'
    metadata={'w_q': 'weight_q', 'w_k': 'weight_k'}, # Match self.weight_q = nn.Linear inside your attn module.
    n_head=8,
    lr=0.02, 
    momentum=0.95,
    rank=0,
    world_size=1
)
adamw = torch.optim.AdamW(adamw_params, lr=3e-4, betas=(0.90, 0.95), weight_decay=0.01)

for batch in dataloader:
    loss, s_max = model(batch) # Your model should output s_max too for each attention layers' heads.
    
    loss.backward()
    muonclip.step(s_max)  # Applies both Muon and QK-Clip
    adamw.step()
    muonclip.zero_grad()
    adamw.zero_grad()
```

### Option 2: Using Muon and QKClip Separately

If you prefer more control, you can use Muon and QKClip separately:
```python
from muon import Muon
from qk_clip import QKClip
import torch

muon_params = [p for p in model.body.parameters() if p.ndim >= 2]
adamw_params = ([p for p in model.body.parameters() if p.ndim < 2]
              + [*model.head.parameters(), *model.embed.parameters()])

muon = Muon(muon_params, lr=0.02, momentum=0.95, rank=0, world_size=1)
adamw = torch.optim.AdamW(adamw_params, lr=3e-4, betas=(0.90, 0.95), weight_decay=0.01)
qkclip = QKClip(
    attention_params=[model.body.layer1.attn, model.body.layer2.attn],
    mode='mha',
    metadata={'w_q': 'weight_q', 'w_k': 'weight_k'}, 
    n_head=8,
    rank=0,
    world_size=1
)

for batch in dataloader:
    loss, s_max = model(batch)
    loss.backward()
    
    # Instead of muonclip.step()
    muon.step()
    qkclip.step(s_max)  # Apply QK-Clip after Muon

    adamw.step()
    
    muon.zero_grad()
    adamw.zero_grad()
```

**Note:** Your model's forward pass should track and return the maximum logits per head (`s_max`) for each attention layer. The format should be:
```python
s_max = [
    [head1_max_logit, head2_max, ...],  # Layer 1
    [head1_max, head2_max, ...],  # Layer 2
    ...
]
```
where each value is $S_h^{max} = \frac{1}{\sqrt{d}} \max_{X \in B} \max_{i,j} Q_h^i K_h^{T,j}$

### Example Attention Module

Here's an example of how to implement an attention module that tracks `s_max`:

```python
import torch
import torch.nn as nn
import math

class MHA(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head
        
        self.weight_q = nn.Linear(d_model, d_model, bias=False)
        self.weight_k = nn.Linear(d_model, d_model, bias=False)
        self.weight_v = nn.Linear(d_model, d_model, bias=False)
        self.weight_o = nn.Linear(d_model, d_model, bias=False)
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        
        Q = self.weight_q(x)  
        K = self.weight_k(x)
        V = self.weight_v(x)
        
        Q = Q.view(batch_size, seq_len, self.n_head, self.d_head).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_head, self.d_head).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_head, self.d_head).transpose(1, 2)
        
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_head)  
        
        # s_max per head = (1/sqrt(d)) * max logit across batch and sequence
        s_max_per_head = []
        for h in range(self.n_head):
            max_logit = scores[:, h, :, :].max().item()
            s_max_per_head.append(torch.tensor(max_logit))
        
        a = torch.softmax(scores, dim=-1)
        a = a @ V  
        a = a.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        output = self.weight_o(a)
        
        return output, s_max_per_head


class TransformerModel(nn.Module):
    def __init__(self, d_model, n_head, n_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            MHA(d_model, n_head) for _ in range(n_layers)
        ])
        self.head = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        s_max_all_layers = []
        
        for layer in self.layers:
            x, s_max_layer = layer(x)
            s_max_all_layers.append(s_max_layer)
        
        logits = self.head(x)
        loss = compute_loss(logits)
        
        return loss, s_max_all_layers
```


**Disclaimer:** QK-Clip does **not support fused QKV linear layers**. You must use separate linear projections for Q, K, and V so that QK-Clip can independently scale the Q and K weight matrices per head. Same for MLA.


## Acknowledgement
The Muon optimizer implementation is based on [Flash-Muon](https://github.com/nil0x9/flash-muon) by Tianyang Lin, which provides an efficient implementation of Muon.

```
@misc{lin2025flash,
  author       = {Tianyang Lin},
  title        = {Flash-Muon: An Efficient Implementation of Muon Optimizer},
  year         = {2025},
  url          = {https://github.com/nil0x9/flash-muon}
}
```

The QK-Clip implementation is based on the approach described in the Kimi K2 paper for attention stability.


## Citation

If you use MuonClip in your research, please cite:

```
@misc{muonclip2025,
  author       = {axion66},
  title        = {Efficient MuonClip: Flash-Muon + QKClip},
  year         = {2025},
  url          = {https://github.com/axion66/efficient-muonclip}
}
```