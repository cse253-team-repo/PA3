import torch
def iou(pred, target):
    """
        Calculate IOU in a for each class; Assume data is one hot encoding
        Args:
            pred: prediction label with one hot encoding; shape -- [n_batch, rows, cols, n_class]
            target: target label with one hot encoding; shape -- [n_batch, rows, cols, n_class]
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
        #print("class: {} intersection: {} union: {}".format(cls, intersection, union))
        if union == 0:
            ious.append(float('nan'))  # if there is no ground truth, do not include in evaluation
        else:
            ious.append(intersection/union)# Append the calculated IoU to the list ious
    
    return ious


def pixel_acc(pred, target):
    """
        Calculate the accuracy for all the pixels
        Args:
            pred: prediction label with one hot encoding;
            target: target label with one hot encoding;
        Returns:
            accuracy: percentage of correct prediction for all the pixels of all the images
    """
    y_hat = torch.argmax(pred, axis=-1)
    y = torch.argmax(target, axis=-1)
    correct = torch.sum(y_hat==y)
    # Number of pixels
    N = (y.reshape(-1,1).shape[0]) 
    return correct / N


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
    print(pixel_acc(pred, target))