import torch
def iou(pred, target):
    """
        Calculate IOU in a for each class
        Args:
            pred: prediction label with one hot encoding
            target: target label with one hot encoding
        Returns:
            ious: list of iou for each class
    """
    ious = []
    n_class = target.shape[-1]

    for cls in range(n_class):
        # Complete this function
        one_hot_coding = torch.eye(n_class)[cls,:]
        intersection = torch.sum((pred*target)[torch.all(target==one_hot_coding,axis=-1)])# intersection calculation
        union = torch.sum(pred[torch.all(pred==one_hot_coding, axis=-1)]) + \
                torch.sum(target[torch.all(target==one_hot_coding, axis=-1)]) - intersection#Union calculation
        if union == 0:
            ious.append(float('nan'))  # if there is no ground truth, do not include in evaluation
        else:
            ious.append(intersection/union)# Append the calculated IoU to the list ious
    
    return ious


def pixel_acc(pred, target):
    pass
    #Complete this function


if __name__ == "__main__":
    # test IOU
    # create pred
    pred = torch.tensor([[[ 0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
         [ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
         [ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]],
        [[ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
         [ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
         [ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]]])
    target = torch.tensor([[[ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
         [ 0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
         [ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]],
        [[ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
         [ 0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.],
         [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.]]])
    print(iou(pred, target))