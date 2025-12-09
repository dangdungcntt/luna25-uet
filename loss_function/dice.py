import torch

class DiceLoss(torch.nn.Module):
    # Dice Loss for segmentation tasks
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum()
        dice = (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        return 1 - dice
    
class DiceBCELoss(DiceLoss):
    def __init__(self, smooth=1.0, weight=None, pos_weight=None):
        super().__init__(smooth)
        self.bce_loss = torch.nn.BCEWithLogitsLoss(weight=weight, pos_weight=pos_weight)

    def forward(self, logits, targets):
        dice = super().forward(logits, targets)
        bce = self.bce_loss(logits, targets.float())
        return (dice + bce) * 0.5
    
    def update_weight(self, weight: torch.FloatTensor = None):
        if weight is not None:
            self.bce_loss.weight = weight
    
    def update_pos_weight(self, pos_weight: torch.FloatTensor = None):
        if pos_weight is not None:
            self.bce_loss.pos_weight = pos_weight
    
if __name__ == "__main__":
    x = torch.randn(4, 1, 64, 64, 64)
    y = torch.randint(0, 2, size=x.shape)
    loss = DiceBCELoss()
    print(loss(x, y))