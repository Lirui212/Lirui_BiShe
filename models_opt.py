# models_opt.py
import torch
import torch.nn as nn

class MultiScaleConv1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernels=(3,7,15), stride=2):
        super().__init__()
        self.kernels = kernels
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=stride, padding=k//2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU()
            )
            for k in kernels
        ])
        self.fuse = nn.Sequential(
            nn.Conv1d(out_ch * len(kernels), out_ch, kernel_size=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU()
        )

    def forward(self, x, return_embed=False):
        ys = [b(x) for b in self.branches]   # 每个分支输出 (B, out_ch, T')
        y = torch.cat(ys, dim=1)             # (B, out_ch*len(kernels), T')
        return self.fuse(y)                  # (B, out_ch, T')

class CNN1D_MS(nn.Module):
    def __init__(self, n_classes=16, use_meta=True, meta_dim=33, ms_kernels = (3, 7 , 15)):
        super().__init__()
        self.use_meta = use_meta

        self.backbone = nn.Sequential(
            MultiScaleConv1D(1, 32, kernels=ms_kernels, stride=2),
            MultiScaleConv1D(32, 64, kernels=ms_kernels, stride=2),

            nn.Conv1d(64, 128, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, 256, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        feat_dim = 256
        in_dim = feat_dim + meta_dim if use_meta else feat_dim
        self.head = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, n_classes)
        )

    def forward(self, x_wave, x_meta=None, return_embed=False):
        z = self.backbone(x_wave)
        z = self.pool(z).squeeze(-1)  # (B,256)

        embed = z

        if self.use_meta:
            z = torch.cat([z, x_meta], dim=1)

        logits = self.head(z)
        if return_embed:
            return logits, embed

        return logits
    
class CNN1D_MS_1Stage(nn.Module):
    def __init__(self, n_classes=16, use_meta=True, meta_dim=33, ms_kernels = (3, 7 , 15)):
        super().__init__()
        self.use_meta = use_meta

        self.backbone = nn.Sequential(
            MultiScaleConv1D(1, 32, kernels=ms_kernels, stride=2),
            
            nn.Conv1d(32, 64, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64, 128, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, 256, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        feat_dim = 256
        in_dim = feat_dim + meta_dim if use_meta else feat_dim
        self.head = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, n_classes)
        )

    def forward(self, x_wave, x_meta=None, return_embed=False):
        z = self.backbone(x_wave)
        z = self.pool(z).squeeze(-1)  # (B,256)
        embed = z

        if self.use_meta:
            z = torch.cat([z, x_meta], dim=1)

        logits = self.head(z)
        if return_embed:
            return logits, embed

        return logits