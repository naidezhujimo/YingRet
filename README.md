# RetNet: Efficient Sequence Modeling with Dual-Paradigm Attention

RetNet is a novel neural architecture that combines **parallel computation for training** and **recurrent computation for inference**, featuring a retention mechanism with exponential decay. This implementation demonstrates efficient sequence modeling through hybrid computation modes and hardware-aware optimizations.


## Key Features
- 🌀 **Dual Computation Modes**:
  - **Parallel Mode**: Full-sequence attention for training
  - **Recurrent Mode**: O(1) memory inference
- 📉 **Exponential Decay Matrix**: Position-aware attention decay
- 🧠 **Enhanced Value Projection**: Optional double-dimensional V vectors
- ⚖️ **Group Normalization**: Head-wise normalization代替LayerNorm
- 🧩 **Modular Design**: Plug-and-play RetNet blocks

## Installation
```bash
git clone https://github.com/yourusername/retnet.git
cd retnet
pip install torch
```

## Usage

### Basic Model Initialization
```python
from model import RetNet

model = RetNet(
    n_layers=6,
    d_model=512,
    n_heads=8,
    vocab_size=32000
).cuda()

# Training mode (parallel computation)
output = model(x, mode='parallel')  # Input shape: [B, L]

# Inference mode (recurrent computation)
output = model(x, mode='recurrent')
```

### Retention Mechanism Configuration
```python
retention = Retention(
    d_model=512,
    n_heads=8,
    double_v_dim=True  # Enable enhanced value projection
)

# Generate decay matrix for sequence length 1024
D = retention.get_decay_matrix(1024, device='cuda')  # [8, 1024, 1024]
```

## Model Architecture
| Component               | Specification                          |
|-------------------------|----------------------------------------|
| Hidden Dimension        | 512                                   |
| Attention Heads         | 8                                     |
| FFN Intermediate Dim    | 2048                                  |
| Default Layers          | 6                                     |
| Value Dimension         | 1024 (when double_v_dim=True)         |

## Core Implementations

### Retention Mechanism
```python
def forward_parallel(self, Q, K, V):
    # Multi-head splitting
    Q = Q.view(B, L, H, D).transpose(1, 2)  # [B, H, L, D]
    # Compute decayed attention
    attn = (Q @ K.transpose(-2, -1)) * self.scale
    attn = attn * D.unsqueeze(0)  # Apply head-specific decay
```

### Recurrent Mode
```python
def forward_recurrent(self, Q, K, V):
    # State maintenance
    state = gamma * state + torch.einsum('bhd,bhe->bhde', Kt, Vt)
    # Output computation
    output = torch.einsum('bhd,bhde->bhe', Qt, state)
```

## Training Configuration
- **Normalization**: GroupNorm over LayerNorm
- **Initialization**:
  - Xavier for linear layers
  - Learned decay parameters (γ)
- **Value Projection**: 
  - Default 2x dimension expansion
  - Disable with `double_v_dim=False`

## Performance
| Mode       | Memory Complexity | Typical Use Case      |
|------------|--------------------|-----------------------|
| Parallel   | O(L²)              | Training              |
| Recurrent  | O(1)               | Inference/Deployment  |

## License
[MIT License](LICENSE) - Open for academic and commercial use.

---

**Note**: For production deployment:
1. Add mixed-precision training support
2. Implement gradient checkpointing
3. Add cross-sequence batch support
4. Consider CUDA kernel optimization for recurrent mode
```
