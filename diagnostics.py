import cv2
import numpy as np
import peakFinder as pf
import matplotlib.pyplot as plt
from multiprocessing import Process


################################################################################
#                                    opencv                                    #
################################################################################

def get_moments(image, rsz=1.0, kernel=(19,19), threshold=3):
	image = image.astype(np.uint8)
	image_prep = pf.preprocess(image, resize=rsz, kernel=kernel)
	image_contour = pf.get_contour(image_prep, threshold)
	M = cv2.moments(image_contour)
	return np.array(M['m00']), np.array((M['m10'], M['m01']))

################################################################################
#                                   Plotting                                   #
################################################################################

def plot_histogram(data, n_bins=50, title=None, xlabel=None, ylabel=None):
	numBins = 50
	n, bins, patches = plt.hist(data, n_bins, normed=1, facecolor='green', alpha=0.75)
	fill_plot_info(title, xlabel, ylabel)
	plt.grid()
	plt.show(block=True)


def plot_x_vs_y(x, y, title=None, xlabel=None, ylabel=None):
	plt.clf()
	fill_plot_info(title, xlabel, ylabel)
	scatter_plot = plt.scatter(x, y)
	plt.grid()
	plt.show(block=True)

def fill_plot_info(title, xlabel, ylabel):
	if title:
		plt.title(title)
	if xlabel:
		plt.xlabel(xlabel)
	if ylabel:
		plt.ylabel(ylabel)
			
################################################################################
#                                     Main                                     #
################################################################################


if __name__ == "__main__":
	src_folder = "/reg/neh/home/apra/work/python/peakfinding/"
	image_dir = src_folder + "images/"
	bad_images_dir = src_folder + "bad_images/"

	n_images = 100
	m_thresh = 6e5
	zero_order_m = np.zeros((n_images, 1))
	first_order_m = np.zeros((n_images, 2))

	
	images = pf.get_images_from_dir(image_dir,n_images=n_images, shuffle=True)
	for i, image in enumerate(images):
		# pf.plot_image(image)
		zero_m, first_m = get_moments(image)
		print zero_m

		if zero_m == 0.0:
			pf.plot_image(image)

		elif zero_m < m_thresh:
			zero_order_m[i,:] = zero_m
			first_order_m[i,:] = first_m
			
	zero_title = "Zero Order Image Moment Histogram"
	zero_xlabel = "Values"
	zero_ylabel = "Number of Images"
	histogram = Process(target=plot_histogram, args=(zero_order_m, 50, zero_title, 
													 zero_xlabel, zero_ylabel))
	histogram.start()

	first_title = "First Order Image Moment Histogram"
	first_xlabel = "X First Order Moment"
	first_ylabel = "Y First Order Moment"	
	first_order = Process(target=plot_x_vs_y, args=(first_order_m[:,0],
													first_order_m[:,1], 
													first_title, first_xlabel,
													first_ylabel))
	first_order.start()
