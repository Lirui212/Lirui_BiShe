import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # logits: (B, C), targets: (B,)
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)

        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)   # (B,)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)           # (B,)

        # focal part
        focal_factor = (1.0 - pt) ** self.gamma
        loss = - focal_factor * log_pt

        # alpha / class weights
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
        
def compute_cb_weights_from_counts(counts, beta=0.9999):
    """
    counts: numpy array, shape (C,)
    return: numpy array, shape (C,)
    """
    counts = np.asarray(counts, dtype=np.float64)
    effective_num = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / (effective_num + 1e-12)
    weights = weights / weights.mean()
    return weights

class CenterLoss(nn.Module):
#     """
#     Center Loss
#     features: (B, feat_dim)
#     targets:  (B,)
#     """
#     def __init__(self, num_classes, feat_dim, device):
#         super().__init__()
#         self.num_classes = num_classes
#         self.feat_dim = feat_dim
#         self.device = device

#         self.centers = nn.Parameter(
#             torch.randn(num_classes, feat_dim, device=device)
#         )

#     def forward(self, features, targets):
#         # 取出每个样本所属类别的中心
#         centers_batch = self.centers[targets]   # (B, feat_dim)
#         loss = 0.5 * ((features - centers_batch) ** 2).sum(dim=1).mean()
#         return loss
    
# class BalancedCenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, device, normalize=True):
        super().__init__()
        self.normalize = normalize
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim, device=device))

    def forward(self, features, targets):
        centers = self.centers

        if self.normalize:
            features = F.normalize(features, dim=1)
            centers = F.normalize(centers, dim=1)

        centers_batch = centers[targets]
        dist = 0.5 * ((features - centers_batch) ** 2).sum(dim=1)

        present_classes = targets.unique()
        loss = 0.0
        for c in present_classes:
            mask = (targets == c)
            loss = loss + dist[mask].mean()

        loss = loss / len(present_classes)
        return loss

class BalancedSupConLoss(nn.Module):
    """
    监督对比损失（支持类别均衡聚合），相比单中心约束更适合“同类多簇”分布。

    核心思路：
    1) 只要求同类样本相对更近、异类样本更远，不强行把整类压成一个中心；
    2) 支持 class-balanced 聚合，减少头部类对嵌入损失的主导；
    3) 使用 batch 内全部同类样本作为正样本，天然允许多模态类内结构存在。
    """
    def __init__(
        self,
        temperature=0.07,
        base_temperature=0.07,
        normalize=True,
        reduction="mean",
        balance_mode="class",   # class / sample
    ):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.normalize = normalize
        self.reduction = reduction
        self.balance_mode = balance_mode

    def forward(self, features, targets):
        """
        features: (B, D)
        targets:  (B,)
        """
        if features.ndim != 2:
            raise ValueError(f"BalancedSupConLoss expects features shape (B, D), got {features.shape}")

        if self.normalize:
            features = F.normalize(features, dim=1)

        device = features.device
        batch_size = features.shape[0]

        if batch_size <= 1:
            return features.new_tensor(0.0)

        targets = targets.contiguous().view(-1, 1)
        if targets.shape[0] != batch_size:
            raise ValueError("Num of labels does not match num of features")

        mask = torch.eq(targets, targets.T).float().to(device)  # (B, B)

        logits = torch.div(torch.matmul(features, features.T), self.temperature)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        pos_per_anchor = mask.sum(dim=1)
        valid_anchor = pos_per_anchor > 0
        if not torch.any(valid_anchor):
            return features.new_tensor(0.0)

        loss_per_anchor = - (self.temperature / self.base_temperature) * (
            (mask * log_prob).sum(dim=1) / (pos_per_anchor + 1e-12)
        )
        loss_per_anchor = loss_per_anchor[valid_anchor]
        valid_targets = targets.squeeze(1)[valid_anchor]

        if self.balance_mode == "sample" or loss_per_anchor.numel() == 0:
            if self.reduction == "sum":
                return loss_per_anchor.sum()
            return loss_per_anchor.mean() if loss_per_anchor.numel() > 0 else features.new_tensor(0.0)

        # class-balanced: 先对每个类别内部取平均，再对类别取平均
        unique_cls = valid_targets.unique(sorted=True)
        cls_losses = []
        for c in unique_cls:
            cls_mask = (valid_targets == c)
            if cls_mask.any():
                cls_losses.append(loss_per_anchor[cls_mask].mean())

        if not cls_losses:
            return features.new_tensor(0.0)

        loss = torch.stack(cls_losses).mean()
        return loss

class MultiCenterLoss(nn.Module):
    """
    多中心嵌入约束：
    - 每个类别有 K 个子中心
    - 每个样本只拉向其所属类别的最近子中心
    - 可选类内子中心分离项，避免多个子中心塌缩到一起
    """
    def __init__(
        self,
        num_classes,
        feat_dim,
        device,
        num_centers=2,
        normalize=True,
        balance_mode="class",          # class / sample
        separation_weight=0.1,
        separation_margin=0.4,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.num_centers = num_centers
        self.normalize = normalize
        self.balance_mode = balance_mode
        self.separation_weight = separation_weight
        self.separation_margin = separation_margin

        self.centers = nn.Parameter(
            torch.randn(num_classes, num_centers, feat_dim, device=device)
        )

    def forward(self, features, targets):
        """
        features: (B, D)
        targets:  (B,)
        """
        if self.normalize:
            features = F.normalize(features, dim=1)
            centers = F.normalize(self.centers, dim=2)
        else:
            centers = self.centers

        B, D = features.shape
        device = features.device

        # 取出每个样本所属类别的 K 个中心: (B, K, D)
        class_centers = centers[targets]

        # 计算到所属类各子中心距离: (B, K)
        dist = 0.5 * ((features.unsqueeze(1) - class_centers) ** 2).sum(dim=2)

        # 最近子中心距离
        min_dist, assign_idx = dist.min(dim=1)

        # pull loss
        if self.balance_mode == "sample":
            pull_loss = min_dist.mean()
        else:
            pull_loss = features.new_tensor(0.0)
            present_classes = targets.unique()
            for c in present_classes:
                mask = (targets == c)
                pull_loss = pull_loss + min_dist[mask].mean()
            pull_loss = pull_loss / max(len(present_classes), 1)

        # separation loss: 同一类不同子中心不要太相似
        sep_loss = features.new_tensor(0.0)
        if self.num_centers > 1 and self.separation_weight > 0:
            cls_terms = []
            present_classes = targets.unique()
            for c in present_classes:
                cc = centers[c]                          # (K, D)
                sim = torch.matmul(cc, cc.t())          # (K, K)
                eye = torch.eye(self.num_centers, device=device, dtype=torch.bool)
                sim = sim.masked_fill(eye, -1.0)        # 去掉对角线
                off_diag = sim[~eye]
                if off_diag.numel() > 0:
                    cls_terms.append(F.relu(off_diag - self.separation_margin).mean())

            if len(cls_terms) > 0:
                sep_loss = torch.stack(cls_terms).mean()

        total_loss = pull_loss + self.separation_weight * sep_loss
        return total_loss

class StageAdaptiveCompositeLoss(nn.Module):
    """
    阶段自适应复合损失：
    L = lambda_ce * CE + lambda_cb * CBCE + lambda_fl * Focal
    """
    def __init__(self, class_counts, total_epochs, beta=0.999, gamma_max=2.5):
        super().__init__()
        self.total_epochs = total_epochs
        self.beta = beta
        self.gamma_max = gamma_max

        class_counts = torch.tensor(class_counts, dtype=torch.float32)

        # 1) CB-CE 用的权重
        effective_num = 1.0 - torch.pow(torch.tensor(beta, dtype=torch.float32), class_counts)
        cb_weights = (1.0 - beta) / (effective_num + 1e-12)
        cb_weights = cb_weights / cb_weights.mean()
        self.register_buffer("cb_weights", cb_weights)

        # 2) 原始 FocalLoss 用的 alpha（逆频率）
        focal_alpha = class_counts.sum() / (class_counts + 1e-12)
        focal_alpha = focal_alpha / focal_alpha.mean()
        self.register_buffer("focal_alpha", focal_alpha)

    def get_stage_weights(self, epoch):
        r = epoch / max(self.total_epochs - 1, 1)

        if r < 0.3:
            u = r / 0.3
            lambda_ce = 1.0 - 0.5 * u
            lambda_cb = 0.5 * u
            lambda_fl = 0.0
            gamma = 1.0

        elif r < 0.7:
            u = (r - 0.3) / 0.4
            lambda_ce = 0.5 - 0.2 * u
            lambda_cb = 0.5 - 0.1 * u
            lambda_fl = 0.3 * u
            gamma = 1.0 + 1.0 * u

        else:
            u = (r - 0.7) / 0.3
            lambda_ce = 0.3 - 0.1 * u
            lambda_cb = 0.4 - 0.1 * u
            lambda_fl = 0.3 + 0.2 * u
            gamma = 2.0 + (self.gamma_max - 2.0) * u

        return lambda_ce, lambda_cb, lambda_fl, gamma

    def forward(self, logits, target, epoch):
        lambda_ce, lambda_cb, lambda_fl, gamma = self.get_stage_weights(epoch)

        loss_ce = F.cross_entropy(logits, target)
        loss_cb = F.cross_entropy(logits, target, weight=self.cb_weights)

        focal_loss_fn = FocalLoss(alpha=self.focal_alpha, gamma=gamma, reduction="mean")
        loss_fl = focal_loss_fn(logits, target)

        total_loss = (
            lambda_ce * loss_ce
            + lambda_cb * loss_cb
            + lambda_fl * loss_fl
        )

        loss_dict = {
            "loss_total": float(total_loss.detach().item()),
            "loss_ce": float(loss_ce.detach().item()),
            "loss_cb": float(loss_cb.detach().item()),
            "loss_fl": float(loss_fl.detach().item()),
            "lambda_ce": float(lambda_ce),
            "lambda_cb": float(lambda_cb),
            "lambda_fl": float(lambda_fl),
            "gamma": float(gamma),
        }
        return total_loss, loss_dict