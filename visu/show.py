import numpy as np
import cv2
import sys
import time
import os

from train.metrics import max_coords

def draw_vt(vt, shape):
	thickness = max(int(np.round(5*shape[0]/224)), 1)
	output = np.zeros(shape)
	for i in range(int(len(vt)/2)):
		cv2.circle(output,(int(np.round(vt[2*i])), int(np.round(vt[2*i +1]))), thickness, (255,255,0), -1)
	return output

def create_visualization(pred, image, vt):
	W,H,C = image.shape
	output = np.zeros((W*3, H , C))
	if len(pred.shape) > 1:
		tmp = []
		for elem in pred:
			x,y = max_coords(elem)
			tmp.append(x)
			tmp.append(y)
		pred = tmp
	output[:W,:,:] = 255 * image
	if vt is not None:
		output[W:2*W,:,:] = draw_vt(vt=vt, shape=image.shape)
	output[2*W:,:,:] = draw_vt(vt=pred, shape=image.shape)
	return output

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
			images.sort(key=lambda f: int(filter(str.isdigit, f))
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
					preds.append(pred)
					image = create_visualization(pred=pred[0] * image_shape, image=img, vt=KP * image_shape)
					cv2.imshow('',np.uint8(image))
					k = cv2.waitKey(33)
					if k==27:	# Esc key to stop
						sys.exit()
					elif k==32: # Esc space to pause
						k = cv2.waitKey(33)
						while(k != 32):
							k = cv2.waitKey(33)
							continue
				else:
					preds.append(pred)
					image = create_visualization(pred=pred[0] * image_shape, image=img, vt=None)
					cv2.imshow('',np.uint8(image))
					k = cv2.waitKey(33)
					if k==27:	# Esc key to stop
						sys.exit()
					elif k==32: # Esc space to pause
						k = cv2.waitKey(33)
						while(k != 32):
							k = cv2.waitKey(33)
							continue
			print()
		if is_labelled:
			return np.mean(errors)
		return preds

def visu_eval_DNN(DNN, image_shape):
	return (evaluate_on_a_set(DNN=DNN, image_shape=image_shape, eval_set=1),
			evaluate_on_a_set(DNN=DNN, image_shape=image_shape, eval_set=2),
			evaluate_on_a_set(DNN=DNN, image_shape=image_shape, eval_set=3),
			evaluate_on_a_set(DNN=DNN, image_shape=image_shape, eval_set=4))

# def visualization(DNN, val_set, show_image):
# 	errors = []
# 	for step in range(val_set.__len__()):
# 		images, KP = val_set.__getitem__(index=step)
# 		preds = DNN.predict(images)
# 		for i in range(len(images)):
# 			errors.append(np.sqrt(np.sum((preds[i]-KP[i])**2)))
# 			if show_image:
# 				image = create_visualization(pred=preds[i], image=images[i], vt=KP[i])
# 				cv2.imshow('',np.uint8(image))
# 				k = cv2.waitKey(33)
# 				if k==27:	# Esc key to stop
# 					sys.exit()
# 				elif k==32: # Esc space to pause
# 					print('\rvisualisation : %i/%i'%(step, val_set.__len__()), end='') 
# 					k = cv2.waitKey(33)
# 					while(k != 32):
# 						k = cv2.waitKey(33)
# 						continue
# 			else:
# 				print('results :')
# 				print('\t err', errors[-1])
# 				print('\t',np.round(preds[i]).astype(int))
# 				print('\t',KP[i])
# 				# time.sleep(1)
# 	return errors[-1]
