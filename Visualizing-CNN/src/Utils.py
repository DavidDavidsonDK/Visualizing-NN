from PIL import Image
import numpy as np
from matplotlib import pylab as plt
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

VGG_INPUT_SIZE = 224
def load_image(img_path) :
	'''Load image'''
	image = load_img(img_path, target_size=(VGG_INPUT_SIZE, VGG_INPUT_SIZE))
	image = img_to_array(image)

	return image

def save_image( npdata, path):
	'''Save image'''
	img = Image.fromarray(np.asarray( np.clip(npdata,0,255), dtype="uint8"), "L" )
	img.save(path)
	

def show_image(image, grayscale = True, ax=None, title=''):
	if ax is None:
		plt.figure()
	plt.axis('off')
	
	if len(image.shape) == 2 or grayscale == True:
		if len(image.shape) == 3:
			image = np.sum(np.abs(image), axis=2)
			
		vmax = np.percentile(image, 99)
		vmin = np.min(image)

		plt.imshow(image, cmap=plt.cm.gray, vmin=vmin, vmax=vmax)
		plt.title(title)
	else:
		image = image #+ 127.5
		image = image.astype('uint8')
		
		plt.imshow(image)
		plt.title(title)