import torch
import torch.nn as nn
import torch.nn.functional as F

class ReportGuidanceHead(nn.Module):
    def __init__(self, bottleneck_channels: int, text_dim: int = 768, hidden_dim: int = 1024, out_dim: int = 512):
        super().__init__()
        self.img_proj = nn.Conv2d(bottleneck_channels, out_dim, kernel_size=1)
        # text side: 768 -> hidden -> out
        self.txt_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        )
        self.img_pool = nn.AdaptiveAvgPool2d((1, 1))


    def forward(self, bottleneck_feat, text_emb_768):
        v = self.img_proj(bottleneck_feat)
        z_img = self.img_pool(v).squeeze(-1).squeeze(-1)
        z_txt = self.txt_proj(text_emb_768)
        return z_img, z_txt
