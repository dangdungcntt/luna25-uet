import torch

class TverskyLoss(torch.nn.Module):
    """
    Tversky Loss for imbalanced segmentation.
    
    - alpha controls FP penalty
    - beta controls FN penalty
    - Higher beta = penalize false negatives more (good when missing nodules is costly)
    """
    def __init__(self, alpha=0.3, beta=0.7, smooth=1.0):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def forward(self, logits, targets):
        # Handle size mismatch
        if logits.shape[2:] != targets.shape[2:]:
            logits = torch.nn.functional.interpolate(
                logits,
                size=targets.shape[2:],
                mode='trilinear',
                align_corners=False
            )
        
        probs = torch.sigmoid(logits)
        
        # Flatten
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        
        # True Positives, False Positives, False Negatives
        TP = (probs_flat * targets_flat).sum()
        FP = (probs_flat * (1 - targets_flat)).sum()
        FN = ((1 - probs_flat) * targets_flat).sum()
        
        tversky_index = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        
        return 1 - tversky_index
    
class TverskyBCELoss(TverskyLoss):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1.0, weight=None, pos_weight=None):
        super().__init__(alpha, beta, smooth)
        self.bce_loss = torch.nn.BCEWithLogitsLoss(weight=weight, pos_weight=pos_weight)

    def forward(self, logits, labels):
        tversky = super().forward(logits, labels)
        bce = self.bce_loss(logits, labels.float())
        return (tversky + bce) * 0.5
    
    def update_weight(self, weight: torch.FloatTensor = None):
        if weight is not None:
            self.bce_loss.weight = weight
    
    def update_pos_weight(self, pos_weight: torch.FloatTensor = None):
        if pos_weight is not None:
            self.bce_loss.pos_weight = pos_weight