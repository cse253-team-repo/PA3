import torch
import yaml
import time
import numpy as np

from PIL import Image
from dataloader import labels_classes



color_array = []
for i in labels_classes:
        if i.ignoreInEval == False:
            color_array.append(i.color)
        else:
            color_array.append(i.color)
color_array = np.array(color_array)

def iou(pred, target):
    """
        Calculate IOU in a for each class; Assume data is one hot encoding
        Args:
            pred: prediction label with one hot encoding; shape -- [n_batch, n_class, rows, cols]
            target: target label with one hot encoding; shape -- [n_batch, n_class, rows, cols]
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

def iou2(pred, target):
    """
        Calculate IOU in a for each class; Assume data is one hot encoding
        Args:
            pred: prediction label with one hot encoding; shape -- [n_batch, n_class, rows, cols]
            target: target label with one hot encoding; shape -- [n_batch, n_class, rows, cols]
        Returns:
            ious: list of iou for each class
    """

    intersection = torch.sum(pred * target, dim=[0,2,3]) # intersection every class
    union = torch.sum(pred, dim=[0,2,3]) + torch.sum(target, dim=[0,2,3]) - intersection
    ious = (intersection / union).tolist()
    return ious

def load_config(path):
    """
    Load the configuration from config.yaml.
    """
    return yaml.load(open(path, 'r'), Loader=yaml.SafeLoader)


def pixel_acc(y_hat, y):
    """
        Calculate the accuracy for all the pixels
        Args:
            pred: prediction label
            y: target label;
        Returns:
            accuracy: percentage of correct prediction for all the pixels of all the images
    """
    correct = torch.sum(y_hat==y).item()
    # Number of pixels
    N = y[y>=0].view(-1).shape[0]
    #print(N, correct)
    return correct / N

def visualize(output, label):
    batch_size, h,w = output.shape[0], output.shape[-2], output.shape[-1] 

    pred = torch.argmax(output, dim=1) 

    pred_img = color_array[pred.detach().cpu().numpy()] # batch_size, h, w, 3
    label_img = color_array[label.detach().cpu().numpy()]

    print("pred img shape:", pred_img.shape)

    pred_img = Image.fromarray(np.uint8(pred_img[0]))
    label_img = Image.fromarray(np.uint8(label_img[0]))




if __name__ == "__main__":
    # test IOU
    # create pred
    n_class = 8
    h = w = 513
    pred = torch.eye(n_class)[torch.randint(0, n_class, (h*w,))].view(h,w,8)
    target = torch.eye(n_class)[torch.randint(0, n_class, (h*w,))].view(h,w,8)
    """
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
    """
    pred = pred.unsqueeze(0)
    target = target.unsqueeze(0)
    print(iou(pred, target))
    start =time.time()
    out = iou2(pred.permute(0, 3, 1, 2), target.permute(0, 3, 1, 2))
    end = time.time()
    print(out)
    print("time: {}".format(end - start))
    #print(pixel_acc(pred, target))


def to_one_hot(label,num_class):
    b, h, w = label.shape
    label_onehot = torch.zeros(b,num_class, h, w).long()
    for c in range(num_class):
        label_onehot[:,c,:,:][label == c] = 1
    return label_onehot
# if __name__ == "__main__":
#     pred_img = np.zeros((3,512,512))
#     color_array = []
#     for i in labels_classes:
#         if i.ignoreInEval == False:
#             color_array.append(i.color)
#         else:
#             color_array.append(i.color)
#     color_array = np.array(color_array)
#     print("color array shape: ", color_array.shape)
#     preds = np.arange(0,24).reshape(2,3,4)
#     a = color_array[preds]
#     print(a.shape)
#
#     for i in a:
#         print(a[0].shape)