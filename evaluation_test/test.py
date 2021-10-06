import os
import cv2
import numpy as np

def evaluate_on_a_set(DNN, image_shape, eval_set):
	preds = []
	errors = []
	if 'Dataset' in os.listdir('..'):
		if 'Test' + str(eval_set) in os.listdir('../Dataset'):
			is_labelled = False
			if 'test' + str(eval_set) + '_labels.npy' in os.listdir('../Dataset'):
				labels = np.load('../Dataset/' + 'test' + str(eval_set) + '_labels.npy')
				is_labelled = True
			images = ['../Dataset/Test' + str(eval_set) + '/' + e for e in os.listdir('../Dataset/Test' + str(eval_set)) if '.png' in e]
			images.sort()
			for cpt, img in enumerate(images):
				print('\rtesting on set %i/4 : %i/%i'%(eval_set, cpt+1, len(images)), end='')
				img_ = cv2.imread(img, cv2.IMREAD_UNCHANGED)
				img = cv2.resize(cv2.imread(img, cv2.IMREAD_UNCHANGED), (image_shape, image_shape)) / 255
				pred = DNN.predict(np.expand_dims(img,axis = 0)) / image_shape
				if is_labelled:
					KP=labels[cpt]
					KP_x = np.copy(KP[::2]) / img_.shape[0]
					KP_y = np.copy(KP[1::2]) / img_.shape[1]
					KP[::2] = KP_x
					KP[1::2] = KP_y
					errors.append(np.sum(np.abs(KP - pred)[0][KP > 0]))
				else:
					preds.append(pred)
			print()
		if is_labelled:
			return np.mean(errors)
		return None

def eval_DNN(DNN, image_shape):
	return (evaluate_on_a_set(DNN=DNN, image_shape=image_shape, eval_set=1),
			evaluate_on_a_set(DNN=DNN, image_shape=image_shape, eval_set=2),
			evaluate_on_a_set(DNN=DNN, image_shape=image_shape, eval_set=3),
			evaluate_on_a_set(DNN=DNN, image_shape=image_shape, eval_set=4))


	import os
import cv2
import numpy as np

def evaluate_on_a_set(DNN, image_shape, eval_set):
	preds = []
	errors = []
	if 'Dataset' in os.listdir('..'):
		if 'Test' + str(eval_set) in os.listdir('../Dataset'):
			is_labelled = False
			if 'test' + str(eval_set) + '_labels.npy' in os.listdir('../Dataset'):
				labels = np.load('../Dataset/' + 'test' + str(eval_set) + '_labels.npy')
				is_labelled = True
			images = ['../Dataset/Test' + str(eval_set) + '/' + e for e in os.listdir('../Dataset/Test' + str(eval_set)) if '.png' in e]
			images.sort()
			for cpt, img in enumerate(images):
				print('\rtesting on set %i/4 : %i/%i'%(eval_set, cpt+1, len(images)), end='')
				img_ = cv2.imread(img, cv2.IMREAD_UNCHANGED)
				img = cv2.resize(cv2.imread(img, cv2.IMREAD_UNCHANGED), (image_shape, image_shape)) / 255
				pred = DNN.predict(np.expand_dims(img,axis = 0)) / image_shape
				if is_labelled:
					KP=labels[cpt]
					KP_x = np.copy(KP[::2]) / img_.shape[0]
					KP_y = np.copy(KP[1::2]) / img_.shape[1]
					KP[::2] = KP_x
					KP[1::2] = KP_y
					errors.append(np.sum(np.abs(KP - pred)[0][KP > 0]))
				else:
					preds.append(pred)
			print()
		if is_labelled:
			return np.mean(errors)
		return None

def eval_DNN(DNN, image_shape):
	return (evaluate_on_a_set(DNN=DNN, image_shape=image_shape, eval_set=1),
			evaluate_on_a_set(DNN=DNN, image_shape=image_shape, eval_set=2),
			evaluate_on_a_set(DNN=DNN, image_shape=image_shape, eval_set=3),
			evaluate_on_a_set(DNN=DNN, image_shape=image_shape, eval_set=4))


	
