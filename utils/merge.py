import numpy as np


class merge_iou:
	def __init__(self):
		self.class_list = [2, 7, 11, 13, 18]
		
	def calculate_5_class(self, iou):
		mean = np.mean(iou)
		class_iou = iou[self.class_list]
		mean_class = np.mean(class_iou)

		return mean, list(class_iou), mean_class



if __name__ == "__main__":
	merge = merge_iou()
	basic_fcn = [7.30241179e-01, 3.32855314e-01, 6.31398380e-01, 4.79245995e-04,
				 4.89278836e-03, 1.26147553e-01, 0.00000000e+00, 2.01303989e-01,
				 7.97311604e-01, 3.06767225e-01, 7.65498996e-01, 3.01871002e-01,
				 0.00000000e+00, 5.07823408e-01, 0.00000000e+00, 0.00000000e+00,
				 0.00000000e+00, 0.00000000e+00, 1.22203693e-01]
	unet_bn = [0.76558155, 0.57691383, 0.77113068, 0.12833847, 0.22157449, 0.45102286,
				 0.36401013, 0.55917227, 0.87275022, 0.43038559, 0.81529766, 0.46183452,
				 0.15253368, 0.77989101, 0.04093395, 0.11353317, 0.01616228, 0.09720594,
				 0.48719034]
	resnet50_fcn = [7.57946491e-01, 5.16903460e-01, 7.31663048e-01, 3.01460642e-02,
				 6.08708570e-03, 4.26573008e-02, 0.00000000e+00, 2.02575713e-01,
				 8.29552948e-01, 3.62801403e-01, 8.31954777e-01, 3.24804872e-01,
				 0.00000000e+00, 7.18937099e-01, 5.57212683e-04, 1.87122394e-04,
				 5.30782563e-04, 0.00000000e+00, 1.71973988e-01]
	fcn_weighted_loss = [7.51631737e-01, 3.85198027e-01, 6.65327966e-01, 1.13397270e-01,
						 1.62945315e-01, 2.85945386e-01, 1.56231195e-01, 3.51170719e-01,
						 8.15237820e-01, 3.47734809e-01, 7.43639052e-01, 3.37053925e-01,
						 5.80587909e-02, 5.26960194e-01, 1.52075618e-05, 7.42391124e-02,
						 1.38068618e-02, 6.51845485e-02, 3.01597744e-01]
	fcn_dice_loss = [8.23633015e-01, 4.06612396e-01, 7.10122526e-01, 6.01044670e-02,
					 5.85489832e-02, 2.16648847e-01, 1.19540622e-04, 2.80434281e-01,
					 8.41342866e-01, 3.18277240e-01, 8.07833970e-01, 3.42988700e-01,
					 5.51283756e-06, 6.25492811e-01, 0.00000000e+00, 0.00000000e+00,
					 0.00000000e+00, 0.00000000e+00, 1.26702085e-01]

	models ={"basic_fcn":basic_fcn, 
			"unet_bn":unet_bn,
			"resnet50_fcn":resnet50_fcn,
			"fcn_weighted_loss":fcn_weighted_loss,
			"fcn_dice_loss": fcn_dice_loss
			}
	accuracy = {"basic_fcn":0.8313693035693839, 
			"unet_bn":0.9106153805324106,
			"resnet50_fcn":0.8803274661511491,
			"fcn_weighted_loss": 0.8365317358563858,
			"fcn_dice_loss": 0.8709270761440688
			}

	for model_name in models.keys():
		iou = models[model_name]
		mean, class_iou, mean_class = merge.calculate_5_class(np.array(iou))
		print("IOU for model {}:".format(model_name))
		print("Mean: %.3f"%(mean))
		temp = [float(format(i, '.3f')) for i in class_iou]
		print("IOU for 5 classes: {}".format(temp))
		print("Mean for 5 classes: %.3f"%(mean_class))
		print("Accuracy: %.3f"%(accuracy[model_name]))

