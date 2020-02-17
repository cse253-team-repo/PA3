import torch
import torch.nn as nn
import torch.nn.functional as F

class Diceloss(nn.Module):
    def init(self):
        super(Diceloss, self).__init__()

    def forward(self,pred, target):

        smooth = 1.
        pred = nn.Softmax()(pred)
        iflat = pred.contiguous().view(-1).cuda()
        tflat = target.contiguous().view(-1).cuda()

        intersection = (iflat * tflat).sum()
        A_sum = torch.sum(iflat * iflat)
        B_sum = torch.sum(tflat * tflat)
        return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth) )

class Loss:
    def __init__(self, method="cross-entropy"):
        """
            Implement various loss function inside this class inclusing naive cross-entropy
            and a loss weighting scheme.
        """
        if method == "cross-entropy":
            self.loss = self.cross_entropy
    def cross_entropy(self, y, target):
        pass


class WCELoss(nn.modules.loss._WeightedLoss):
    def __init__(self, gamma=2, weight=None, size_average=None, ignore_index=-1,
                 reduce=None, reduction='mean'):
        super(WCELoss, self).__init__(weight, size_average, reduce, reduction)
        if weight:
            self.weight = torch.tensor(weight)
        else:
            self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        n,c,h,w = pred.shape
        iflat = pred.contiguous().view(-1).cuda()
        tflat = target.contiguous().view(-1).cuda()
        iflat = F.log_softmax(iflat, dim=1)

        if  self.weight:
            loss = -(torch.mul(self.weight, iflat * tflat)).sum(dim=-1)
        else:
            loss = -(iflat * tflat).sum()
        return loss

class OhemCELoss(nn.Module):
    def __init__(self, thresh, n_min, ignore_lb=255, *args, **kwargs):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).cuda()
        self.n_min = n_min
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels):
        N, C, H, W = logits.size()
        loss = self.criteria(logits, labels).view(-1)
        loss, _ = torch.sort(loss, descending=True)
        if loss[self.n_min] > self.thresh:
            loss = loss[loss>self.thresh]
        else:
            loss = loss[:self.n_min]
        return torch.mean(loss)


class SoftmaxFocalLoss(nn.Module):
    def __init__(self, gamma, ignore_lb=255, *args, **kwargs):
        super(SoftmaxFocalLoss, self).__init__()
        self.gamma = gamma
        self.nll = nn.NLLLoss(ignore_index=ignore_lb)

    def forward(self, logits, labels):
        scores = F.softmax(logits, dim=1)
        factor = torch.pow(1.-scores, self.gamma)
        log_score = F.log_softmax(logits, dim=1)
        log_score = factor * log_score
        loss = self.nll(log_score, labels)
        return loss

