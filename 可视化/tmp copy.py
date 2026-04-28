import torch
import torch.nn as nn

# ====================== 直接复制你代码里的两个模型 ======================
class CNN1D(nn.Module):
    def __init__(self, n_classes=6, use_meta=True, meta_dim=33):
        super().__init__()
        self.use_meta = use_meta

        self.backbone = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),

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
        if use_meta:
            self.head = nn.Sequential(
                nn.Linear(feat_dim + meta_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, n_classes)
            )
        else:
            self.head = nn.Sequential(
                nn.Linear(feat_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, n_classes)
            )

    def forward(self, x_wave, x_meta=None, return_embed=False):
        z = self.backbone(x_wave)
        z = self.pool(z).squeeze(-1)
        embed = z

        if self.use_meta:
            z = torch.cat([z, x_meta], dim=1)

        logits = self.head(z)
        if return_embed:
            return logits, embed
        return logits

class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=5, dilation=1, dropout=0.15):
        super().__init__()
        pad = (k - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, k, padding=pad, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, k, padding=pad, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.down = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        y = self.drop(torch.relu(self.bn1(self.conv1(x))))
        y = self.drop(torch.relu(self.bn2(self.conv2(y))))
        res = x if self.down is None else self.down(x)
        return torch.relu(y + res)

class TCN(nn.Module):
    def __init__(self, n_classes=6, use_meta=True, meta_dim=33, channels=(16,32,64,128)):
        super().__init__()
        self.use_meta = use_meta
        layers = []
        in_ch = 1
        for i, out_ch in enumerate(channels):
            layers.append(TCNBlock(in_ch, out_ch, k=5, dilation=2**i, dropout=0.15))
            in_ch = out_ch
        self.tcn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

        feat_dim = channels[-1]
        if use_meta:
            self.fc = nn.Sequential(
                nn.Linear(feat_dim + meta_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, n_classes)
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(feat_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, n_classes)
            )

    def forward(self, x_wave, x_meta=None):
        z = self.tcn(x_wave)
        z = self.pool(z).squeeze(-1)
        if self.use_meta:
            z = torch.cat([z, x_meta], dim=1)
        return self.fc(z)

# ====================== 导出模型 ======================
if __name__ == "__main__":
    # 你训练时的配置（必须和训练时一致！）
    n_classes = 6       # 你的故障分类数 C0~C5
    use_meta = True     # 是否使用meta信息
    meta_dim = 33

    # 1. 初始化模型
    cnn_model = CNN1D(n_classes=n_classes, use_meta=use_meta, meta_dim=meta_dim)
    tcn_model = TCN(n_classes=n_classes, use_meta=use_meta, meta_dim=meta_dim)

    # 2. 导出 初始化权重.pth（可用于后续训练/可视化）
    torch.save(cnn_model.state_dict(), "CNN1D_init.pth")
    torch.save(tcn_model.state_dict(), "TCN_init.pth")

    # 3. 导出 完整模型结构+权重（Netron直接打开可视化）
    torch.save(cnn_model, "CNN1D_full_model.pth")
    torch.save(tcn_model, "TCN_full_model.pth")

    print("✅ 模型导出完成！")
    print("📁 CNN1D_init.pth       (仅权重)")
    print("📁 TCN_init.pth         (仅权重)")
    print("📁 CNN1D_full_model.pth (结构+权重，Netron专用)")
    print("📁 TCN_full_model.pth   (结构+权重，Netron专用)")