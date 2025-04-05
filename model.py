import torch
import torch.nn as nn
import torch.nn.functional as F

class Retention(nn.Module):
    """
    Args:
        d_model: 隐层维度
        n_heads: 头数
        double_v_dim: 是否将V的维度加倍(增加表达能力)
    """
    def __init__(self, d_model, n_heads, double_v_dim=True):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.double_v = double_v_dim

        # 初始化投影矩阵
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(
            d_model,
            d_model * (2 if double_v_dim else 1),
            bias=False
        )

        # 衰减因子参数化
        self.gamma_logit = nn.Parameter(torch.randn(n_heads))

        # 使用 GroupNorm 替代 LayerNorm
        self.group_norm = nn.GroupNorm(
            num_groups=n_heads,
            num_channels=d_model
        )

    def get_decay_matrix(self, seq_len, device):
        """生成衰减矩阵D(因果指数衰减)
        Args:
            seq_len: 序列长度
            device: 计算设备
        Returns:
            D: [H, L, L]衰减矩阵，下三角且每行指数衰减
        """
        position = torch.arange(seq_len, device=device).unsqueeze(0) - \
                    torch.arange(seq_len, device=device).unsqueeze(1) # 相对位置差 [L, L]
        gamma = torch.sigmoid(self.gamma_logit) # 每个头的衰减率 [H]
        decay = gamma.view(-1, 1, 1) ** position.abs().clamp(min=0) # γ^|n-m| [H, L, L]
        return decay.tril() # 下三角因果掩码

    def forward_parallel(self, Q, K, V):
        """并行计算模式(训练时使用)
        Args:
            Q: [B, L, d_model] 查询向量
            K: [B, L, d_model] 键向量
            V: [B, L, d_model*(1+int(double_v))] 值向量
        Returns:
            output: [B, L, d_model] 保留机制输出
        """
        B, L, _ = Q.size()
        D = self.get_decay_matrix(L, Q.device) # [H, L, L]

        # 多头拆分
        Q = Q.view(B, L, self.n_heads, self.head_dim).transpose(1, 2) # [B, H, L, D]
        K = K.view(B, L, self.n_heads, self.head_dim).transpose(1, 2) # [B, H, L, D]
        V = V.view(B, L, self.n_heads, self.head_dim * (2 if self.double_v else 1)).transpose(1, 2)

        # 保留得分计算
        attn = (Q @ K.transpose(-2, -1)) * self.scale # [B, H, L, L]
        attn = attn * D.unsqueeze(0) # 应用衰减矩阵
        output = attn @ V  # [B, H, L, D*(1+int(double_v))]

        # 合并多头输出
        output = output.transpose(1, 2).contiguous().view(B, L, -1) # [B, L, d_model]
        return self.group_norm(output)
    
    def forward_recurrent(self, Q, K, V):
        """递归计算模式(推理时使用)
        Args:
            Q: [B, L, d_model] 查询向量
            K: [B, L, d_model] 键向量
            V: [B, L, d_model*(1+int(double_v))] 值向量
        Returns:
            output: [B, L, d_model] 保留机制输出
        """
        B, L, _ = Q.size
        outputs = []

        # 初始化状态(存储KV的外积和)
        state = torch.zeros(
            B, self.n_heads,
            self.head_dim, # K的维度
            self.head_dim * (2 if self.double_v else 1),
            device=Q.device
        )

        gamma = torch.sigmoid(self.gamma_logit) # [H]
        gamma = gamma.view(1, -1, 1, 1)

        for t in range(L):
            # 当前时刻的Q/K/V向量
            Qt = Q[:, t].view(B, self.n_heads, self.head_dim) # [B, H, D]
            Kt = K[:, t].view(B, self.n_heads, self.head_dim) # [B, H, D]
            Vt = V[:, t].view(B, self.n_heads, self.head_dim * (2 if self.double_v else 1)) # [B, H, D*]

            # 状态更新: s_t = γ * s_{t-1} + K_t^T V_t
            # 使用einsum进行高效外积求和
            outer_product = torch.einsum('bhd,bhe->bhde', Kt, Vt) # [B, H, D, E]
            state = gamma * state + outer_product

            # 计算输出: y_t = Q_t s_t
            output = torch.einsum('bhd,bhde->bhe', Qt, state) # [B, H, E]
            outputs.append(output)

        # 合并所有时间步输出
        output = torch.stack(outputs, dim=1).view(B, L, -1)
        return self.group_norm(output)
    
    def forward(self, x, mode='parallel'):
        """前向传播
        Args:
            x: [B, L, d_model] 输入序列
            mode: 计算模式(parallel/recurrent)
        """
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        if mode == 'parallel':
            return self.forward_parallel(Q, K, V)
        elif mode == 'recurrent':
            return self.forward_recurrent(Q, K, V)
        else:
            raise ValueError(f"无效模式: {mode}, 必须为parallel或recurrent")
        
class RetNetBlock(nn.Module):
    """
    Args:
        d_model: 隐层维度
        n_heads: 头数
        ffn_dim: FFN中间层维度(默认2048)
    """
    def __init__(self, d_model, n_heads, ffn_dim=2048):
        super().__init__()
        self.retention = Retention(d_model, n_heads)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model)
        )

        # 层归一化
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mode='parallel'):
        retained = self.retention(self.norm(x), mode=mode)
        x = x + retained

        fed_forward = self.ffn(self.norm(x))
        x = x + fed_forward
        return x
    
class RetNet(nn.Module):
    """
    Args:
        n_layers: 层数
        d_model: 模型维度
        n_heads: 头数
        vocab_size: 词表大小
    """
    def __init__(self, n_layers=6, d_model=512, n_heads=8, vocab_size=32000):
        super().__init__()
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model)

        # 堆叠RetNet块
        self.layers = nn.ModuleList([
            RetNetBlock(d_model, n_heads)
            for _ in range(n_layers)
        ])

        # 输出层
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, x, mode='parallel'):
        # 输入形状: [B, L]
        x = self.embedding(x)  # [B, L, d_model]
        
        # 逐层处理
        for layer in self.layers:
            x = layer(x, mode=mode)
            
        return self.out(x)  # [B, L, vocab_size]