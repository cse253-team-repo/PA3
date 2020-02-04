def iou(pred, target):
    ious = []
    for cls in range(n_class):
        # Complete this function
        intersection = # intersection calculation
        union = #Union calculation
        if union == 0:
            ious.append(float('nan'))  # if there is no ground truth, do not include in evaluation
        else:
            # Append the calculated IoU to the list ious
    return ious


def pixel_acc(pred, target):
    #Complete this function