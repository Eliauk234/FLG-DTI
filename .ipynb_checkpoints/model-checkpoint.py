import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ---------------------- Focal Loss ----------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()

# ---------------------- ECA ----------------------
class ECA(nn.Module):
    def __init__(self, channels, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(channels, channels, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c, 1)
        y = self.conv(y)
        y = self.sigmoid(y).view(b, c, 1)
        return x * y.expand_as(x)

# ---------------------- EnhancedResidualBlock ----------------------
class EnhancedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=(kernel_size - 1) // 2),
            nn.BatchNorm1d(out_channels),
            nn.ELU(),
            ECA(out_channels),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=(kernel_size - 1) // 2),
            nn.BatchNorm1d(out_channels)
        )
        self.shortcut = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv_layers(x)
        if out.size(2) != residual.size(2):
            diff = out.size(2) - residual.size(2)
            residual = F.pad(residual, (0, diff))
        out += residual
        return F.elu(out)

# ---------------------- FeatureFusionModule ----------------------
class FeatureFusionModule(nn.Module):
    def __init__(self, conv):
        super().__init__()
        self.conv = conv
        self.gate = nn.Sequential(
            nn.Linear(self.conv * 2, self.conv * 4),
            nn.GLU(dim=-1)
        )

    def forward(self, drug_feat, protein_feat):
        concated = torch.cat([drug_feat, protein_feat], dim=1)
        return self.gate(concated)

# ---------------------- Cless ----------------------
class EfficientHead(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.layers(x)

# ---------------------- Multi-Head Attention ----------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, n_heads=4, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        assert embed_dim % n_heads == 0

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, L, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, L, self.n_heads, 3 * self.head_dim).permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, L, self.embed_dim)
        x = self.out_proj(x)
        return x
# ---------------------- FLGDTI ----------------------
class FLGDTI(nn.Module):
    def __init__(self, hp, args):
        super(FLGDTI, self).__init__()
        self.dim = hp.char_dim
        self.conv = hp.conv
        self.drug_kernel = hp.drug_kernel
        self.protein_kernel = hp.protein_kernel
        self.drug_vocab_size = args['input_d_dim']
        self.protein_vocab_size = args['input_p_dim']
        self.dropout_rate = 0.2

        # Embedding
        self.drug_embed = nn.Embedding(self.drug_vocab_size, self.dim, padding_idx=0)
        self.protein_embed = nn.Embedding(self.protein_vocab_size, self.dim, padding_idx=0)

        # CNN
        self.Drug_CNNs = nn.Sequential(
            EnhancedResidualBlock(self.dim, self.conv // 2, self.drug_kernel[0]),
            ECA(self.conv // 2),
            EnhancedResidualBlock(self.conv // 2, self.conv, self.drug_kernel[1]),
            nn.AdaptiveMaxPool1d(1)
        )
        self.Protein_CNNs = nn.Sequential(
            EnhancedResidualBlock(self.dim, self.conv // 2, self.protein_kernel[0]),
            ECA(self.conv // 2),
            EnhancedResidualBlock(self.conv // 2, self.conv, self.protein_kernel[1]),
            nn.AdaptiveMaxPool1d(1)
        )

        # Attention
        self.drug_flow_attn = MultiHeadAttention(self.conv)
        self.protein_flow_attn = MultiHeadAttention(self.conv)

        # Feature
        self.feature_fusion = FeatureFusionModule(self.conv)

        # Classifier
        self.cls_head = EfficientHead(self.conv * 2)

    def forward(self, drug, protein):
        # Embedding
        drug_embed = self.drug_embed(drug).permute(0, 2, 1)
        protein_embed = self.protein_embed(protein).permute(0, 2, 1)

        # CNN
        drug_cnn_out = self.Drug_CNNs(drug_embed).squeeze(-1)
        protein_cnn_out = self.Protein_CNNs(protein_embed).squeeze(-1)

        drug_cnn_out = drug_cnn_out.unsqueeze(1)
        protein_cnn_out = protein_cnn_out.unsqueeze(1)

        # Attention
        drug_attn_out = self.drug_flow_attn(drug_cnn_out).squeeze(1)
        protein_attn_out = self.protein_flow_attn(protein_cnn_out).squeeze(1)

        # Feature
        fused_features = self.feature_fusion(drug_attn_out, protein_attn_out)

        # Classifier
        predict = self.cls_head(fused_features)

        return predict
