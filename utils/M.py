# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

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

class FeatureFusionModule(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(FeatureFusionModule, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=4, dim_feedforward=hidden_dim, dropout=0.1)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)

    def forward(self, x):
        return self.encoder(x)
class CustomCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, max_seq_length=1672):
        super(CustomCrossAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.positional_encoding = nn.Parameter(torch.randn(1, max_seq_length, embed_dim))

    def forward(self, query, key, value):
        seq_length = key.size(0)  # 获取 key 的序列长度
        batch_size = key.size(1)  # 获取 batch_size
        embed_dim = key.size(2)   # 获取嵌入维度

        # 调整 positional_encoding 的形状
        if seq_length > self.positional_encoding.size(1):
            # 插值调整 positional_encoding 的序列长度
            pos_enc = F.interpolate(self.positional_encoding, size=(seq_length,), mode="linear", align_corners=False)
        else:
            # 截取 positional_encoding 的序列长度
            pos_enc = self.positional_encoding[:, :seq_length, :embed_dim]

        # 调整 pos_enc 的维度顺序以匹配 key
        pos_enc = pos_enc.expand(batch_size, -1, -1).permute(1, 0, 2)

        # 确保 pos_enc 和 key 的形状一致
        assert pos_enc.shape == key.shape, f"pos_enc shape {pos_enc.shape} does not match key shape {key.shape}"

        # 加法操作
        key = key + pos_enc
        value = value + pos_enc

        # 注意力机制
        attn_output, _ = self.attention(query, key, value)
        return attn_output

class FMCADTI(nn.Module):
    def __init__(self, hp, args):
        super(FMCADTI, self).__init__()
        self.dim = 256
        self.conv = hp.conv
        self.drug_kernel = hp.drug_kernel
        self.protein_kernel = hp.protein_kernel
        self.drug_vocab_size = args['input_d_dim']
        self.protein_vocab_size = args['input_p_dim']
        self.lstm_hidden = 64
        self.dropout_rate = 0.5

        self.drug_embed = nn.Embedding(self.drug_vocab_size, self.dim, padding_idx=0)
        self.protein_embed = nn.Embedding(self.protein_vocab_size, self.dim, padding_idx=0)

        self.Drug_CNNs = nn.Sequential(
            nn.Conv1d(self.dim, self.conv // 2, kernel_size=self.drug_kernel[0]),
            nn.BatchNorm1d(self.conv // 2),
            nn.ELU(),
            ECA(self.conv // 2),
            nn.Conv1d(self.conv // 2, self.conv, kernel_size=self.drug_kernel[1]),
            nn.BatchNorm1d(self.conv),
            nn.ELU(),
        )
        self.Protein_CNNs = nn.Sequential(
            nn.Conv1d(self.dim, self.conv // 2, kernel_size=self.protein_kernel[0]),
            nn.BatchNorm1d(self.conv // 2),
            nn.ELU(),
            ECA(self.conv // 2),
            nn.Conv1d(self.conv // 2, self.conv, kernel_size=self.protein_kernel[1]),
            nn.BatchNorm1d(self.conv),
            nn.ELU(),
        )

        self.drug_lstm = nn.LSTM(
            input_size=self.conv,
            hidden_size=self.lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=self.dropout_rate
        )
        self.protein_lstm = nn.LSTM(
            input_size=self.conv,
            hidden_size=self.lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=self.dropout_rate
        )

        self.attention = CustomCrossAttention(embed_dim=self.lstm_hidden * 2, num_heads=8, dropout=self.dropout_rate)

        self.feature_fusion = FeatureFusionModule(input_dim=self.lstm_hidden * 4, hidden_dim=256)

        self.fc1 = nn.Linear(self.lstm_hidden * 4, 128)
        self.dropout1 = nn.Dropout(self.dropout_rate)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 1)

    def forward(self, drug, protein):
        drug_embed = self.drug_embed(drug).permute(0, 2, 1)
        protein_embed = self.protein_embed(protein).permute(0, 2, 1)

        drug_cnn_out = self.Drug_CNNs(drug_embed).permute(0, 2, 1)
        protein_cnn_out = self.Protein_CNNs(protein_embed).permute(0, 2, 1)

        drug_lstm_out, _ = self.drug_lstm(drug_cnn_out)
        protein_lstm_out, _ = self.protein_lstm(protein_cnn_out)

        drug_lstm_out = drug_lstm_out.permute(1, 0, 2)
        protein_lstm_out = protein_lstm_out.permute(1, 0, 2)

        attention_output = self.attention(drug_lstm_out, protein_lstm_out, protein_lstm_out)

        drug_vec = torch.max(attention_output, dim=0).values
        protein_vec = torch.max(protein_lstm_out, dim=0).values

        pair = torch.cat([drug_vec, protein_vec], dim=1)

        fused_features = self.feature_fusion(pair.unsqueeze(1)).squeeze(1)

        fully1 = F.relu(self.fc1(fused_features))
        fully1 = self.dropout1(fully1)
        fully2 = F.relu(self.fc2(fully1))
        predict = self.out(fully2)

        return predict