import IPython
import time
import cv2
import os 
import random
import psana
import numpy as np
import matplotlib.pyplot as plt
from  matplotlib.patches import Rectangle
from joblib import Memory
from tqdm import tqdm
from ast import literal_eval
from multiprocessing import Process

cachedir = "cache"
mem = Memory(cachedir=cachedir, verbose = 0)

################################################################################
#                                Default Globals                               #
################################################################################

rsz_default=1.0
kernel_size = (17,17)
max_m0 = 10e5
min_m0 = 10e1
total_images = 25425
src_folder = "/reg/neh/home/apra/work/python/peakfinding/"
DEFAULT = object()

################################################################################
#                                    OpenCV                                    #
################################################################################

def preprocess(image, resize=rsz_default, kernel=(15,15), sigma=0):
	"""Preprocess the image for resizing,noise reduction, etc"""
	image = to_uint8(image)
	image_small = cv2.resize(image, (0,0), fx=resize, fy=resize)
	image_gblur = cv2.GaussianBlur(image_small, kernel, sigma)
	return image_gblur

def get_contour(image, threshold=3):
	"""Returns the first element in the contours list."""
	max_val = image.max()
	std_val = image.std()
	_, image_thresh = cv2.threshold(image, threshold*std_val, max_val, 
	                                cv2.THRESH_TOZERO)	
	_, contours, _ = cv2.findContours(image_thresh, 1, 2)
	return contours[0]

def get_image_moments(image=None, contour=None, threshold=3):
	"""Returns the moments of an image."""
	if contour is None and image is not None:
		contour = get_contour(image, threshold)
	return cv2.moments(contour)

def get_centroid(M):
	"""Returns the centroid using the inputted image moments."""	
	return int(M['m10']/M['m00']), int(M['m01']/M['m00'])
		
def get_bounding_box(image, contour=None, threshold=3):
	"""
	Draws a box around the contour in the inputted image.
	"""
	if contour is None:
		contour = get_contour(image, threshold)
	x,y,w,h = cv2.boundingRect(contour)
	return x,y,w,h

def check_for_beam(M=None, image=None, contour=None, resize=rsz_default, 
                   kernel=kernel_size, m_thresh_max = max_m0, 
                   m_thresh_min=min_m0):
	"""
	Checks if there is a beam in the image by checking the value of the zeroth
	moment. If it is very large, then the image has no beam.
	"""	
	try:

		if M['m00'] < m_thresh_max and M['m00'] > m_thresh_min:
			return True
		else:
			return False
	except TypeError:
		if image is not None and contour is None:
			image = to_uint8(image)
			image_prep = preprocess(image, resize=resize, kernel=kernel)
			contour = get_contour(image_prep)
		M = get_image_moments(contour=contour)
	return check_for_beam(M)

def find_beam(image, resize=rsz_default, kernel=kernel_size):
	"""
	Returns the centroid and bounding box of the beam. Returns None if no beam 
	is present.
	"""
	image_prep = preprocess(image, resize=resize, kernel=kernel)
	contour = get_contour(image_prep)
	M = get_image_moments(contour=contour)
	
	centroid, bounding_box = None, None
	if check_for_beam(M):
		centroid     = [pos//resize for pos in get_centroid(M)]
		bounding_box = [val//resize for val in get_bounding_box(
			image_prep, contour)]
	return centroid, bounding_box

################################################################################
#                                  Plotting                                    #
################################################################################

def plot_image(image, centroid=DEFAULT, bounding_box=DEFAULT, msg = ""):
	"""
	Plots an image on its own or with a centroid, bounding box, message or any 
	combination of the three.
	"""
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1)
	ax.imshow(image)
	if centroid and centroid is not DEFAULT:
		circ = plt.Circle(centroid, radius=5, color='g')
		ax.add_patch(circ)
		msg = "Centroid: {0}".format(centroid)
	elif centroid is None:
		msg = "No Beam Found"
	if bounding_box is not None:
		x,y,w,h = bounding_box
		box = Rectangle((x,y),w,h,linewidth=2,edgecolor='r',facecolor='none')
		ax.add_patch(box)
	if msg:
		plt.text(0.95, 0.05, msg, ha='right', va='center', color='w',
		         transform=ax.transAxes)
	plt.grid()
	plt.show()

def get_present_and_absent_plot(images, save=False, plot=True):
	"""
	Plots and/or saves the first image that has the beam present and absent in
	one figure.
	"""
	fig = plt.figure()
	fig.suptitle('OpenCV on MFX1161 Yag Images', fontsize=20)
	beam_present_image = False
	beam_absent_image = False
	for image in images:		
		centroid, bounding_box = track_beam(image)
		if centroid and bounding_box and not beam_present_image:
			ax = fig.add_subplot(121)
			ax.imshow(image)
			circ = plt.Circle(centroid, radius=5, color='g')
			ax.add_patch(circ)
			x,y,w,h = bounding_box
			box = Rectangle((x,y), w, h, linewidth=1, edgecolor='r',
			                facecolor='none')
			ax.add_patch(box)
			beam_present_image = True
			msg = "Centroid: {0}".format(centroid)
			plt.text(0.95, 0.05, msg, ha='right', va='center', color='w',
			         transform=ax.transAxes)
		if not centroid and not bounding_box and not beam_absent_image:
			bx = fig.add_subplot(122)
			bx.imshow(image)
			msg = "Beam Not Present"
			plt.text(0.95, 0.05, msg, ha='right', va='center', color='w',
			         transform=bx.transAxes)
			beam_absent_image = True
		if beam_present_image and beam_absent_image:
			break
	if save:
		savefig(save)
	if plot:
		plt.show()

#This is broken
def plot_x_y_drift(x, y):
	"""Plots the drift over event number of the beam centroid."""
	x = [np.nan if i is None else i for i in x]
	y = [np.nan if i is None else i for i in y]
	mask_x = np.where(x == np.nan)
	mask_y = np.where(y == np.nan)
	mx = np.ma.array(x)
	mx[mask_x] = np.ma.masked
	my = np.ma.array(y)
	my[mask_y] = np.ma.masked
	fig = plt.figure()
	fig.suptitle('Beam Centroid Drift in MFX1161 Yag', fontsize=20)
	ax = fig.add_subplot(121)
	plt.xlabel('Event Number')
	plt.ylabel('Pixel Value')
	plt.title('Drift in X')
	plt.axis([0, len(x), 0, 1000])
	plt.plot(range(len(x)), mx)
	plt.grid()
	bx = fig.add_subplot(122)
	plt.xlabel('Event Number')
	plt.ylabel('Pixel Value')
	plt.title('Drift in Y')
	plt.axis([0, len(y), 0, 1000])
	plt.plot(range(len(y)), my)
	plt.grid()
	plt.show()

################################################################################
#                                    psana                                     #
################################################################################

@mem.cache
def classify_and_plot_events(data_source_str, detector_str, resize=rsz_default):
	print "Initializing DataSource using '{0}', and Detector '{1}'".format(
		data_source_str, detector_str)
	ds = psana.DataSource(data_source_str)
	det = psana.Detector(detector_str)
	all_x = all_y = []
	beam, no_beam, none_type = [], [], []
	for nevent, event in enumerate(tqdm(ds.events())):
		image = det.image(event)
		try:
			image = to_uint8(image)
			image_prep = preprocess(image, resize=resize)
			contour = get_contour(image_prep)
			M = get_image_moments(contour=contour)
			if check_for_beam(M):
				centroid = [pos//resize for pos in get_centroid(M)]
				beam.append(nevent)
				all_x.append(centroid[0])
				all_y.append(centroid[1])
			else:
				no_beam.append(nevent)
				all_x.append(None)
				all_y.append(None)
		except (TypeError, AttributeError):
			none_type.append(nevent)
			all_x.append(None)
			all_y.append(None)
	return beam, no_beam, none_type, all_x, all_y


@mem.cache
def get_all_centroids(data_source_str, detector_str):
	print "Initializing DataSource using '{0}', and Detector '{1}'".format(
		data_source_str, detector_str)
	ds = psana.DataSource(data_source_str)
	det = psana.Detector(detector_str)
	all_x, all_y = [], []
	for nevent, event in enumerate(tqdm(ds.events())):
		image = det.image(event)
		try:
			centroid, _ = find_beam(image)
			all_x.append(centroid[0])
			all_y.append(centroid[1])
		except (TypeError, AttributeError):
			all_x.append(None)
			all_y.append(None)
		if nevent > 1300: break
	return all_x, all_y

@mem.cache
def get_beam_nobeam_nonetype_all(data_source_str, detector_str):
	print "Initializing DataSource using '{0}', and Detector '{1}'".format(
		data_source_str, detector_str)
	ds = psana.DataSource(data_source_str)
	det = psana.Detector(detector_str)
	beam, no_beam, none_type = [], [], []
	for nevent, event in enumerate(tqdm(ds.events())):
		image = det.image(event)
		try:
			image = to_uint8(image)
			image_prep = preprocess(image, resize=rsz_default, kernel=kernel_size)
			contour = get_contour(image_prep)
			M = get_image_moments(contour=contour)
			if check_for_beam(M):
				beam.append(nevent)
			else:
				no_beam.append(nevent)
		except (TypeError, AttributeError):
			none_type.append(nevent)
	return beam, no_beam, none_type

@mem.cache
def get_all_timestamps(data_source_str, detector_str):
	"""
	Returns a list containing a tuple of the seconds, nanoseconds, and fiducials
	for every event.
	"""
	print "Initializing DataSource using '{0}', and Detector '{1}'".format(
		data_source_str, detector_str)
	ds = psana.DataSource(data_source_str)
	det = psana.Detector(detector_str)
	timestamps = []
	for event in tqdm(ds.events()):
		evtId = event.get(psana.EventId)
		timestamps.append((evtId.time()[0], evtId.time()[1], evtId.fiducials()))
	return timestamps

def get_event_from_nevent(ds, run, nevent, timestamps):
	"""Returns the event object associated for the nevent."""
	sec, nsec, fid = timestamps[nevent]
	event_time = psana.EventTime(int((sec<<32)|nsec),fid)
	return run.event(event_time)
		            
################################################################################
#                               Helper Functions                               #
################################################################################

def to_uint8(image):
	"""*Correctly* converts an image to uint8 type."""
	np.clip(image, 0, 255, out=image)
	return image.astype(np.uint8)

def get_n_nevents(n_events, n_start=0, n_end=total_images):
	return random.sample(range(n_start, n_end), n_events)

def print_image_info(image, resize=rsz_default, kernel=kernel_size):
	"""Prints general image information."""
	print "Image Size: {0}".format(image.shape)
	print "Image Max: {0}".format(image.max())
	print "Image Min: {0}".format(image.min())
	print "Image Mean: {0}".format(image.mean())
	print "Image dtype: {0}\n".format(image.dtype)
	image = to_uint8(image)
	image_prep = preprocess(image, resize=resize, kernel=kernel)
	contour = get_contour(image_prep)
	M = get_image_moments(contour=contour)
	second_m = ['m20', 'm11', 'm02', 'm30', 'm21', 'm12', 'm03']
	print "Zero Order Moment: {0}".format(M['m00'])
	print "First Order Moments: {0}, {1}".format(M['m10'], M['m01'])
	print "Second Order Moments:"
	second_m_str = ''
	for m2 in second_m:
		second_m_str += "{0},".format(M[m2])
	print second_m_str[:-1]

def get_images_from_dir(src_folder, n_images=None, shuffle=False):
	"""
	Crawls through the contents of a directory and saves files with image 
	extensions as images.
	"""
	valid_extensions = set(["bmp", "jpeg", "jpg", "png", "tif", "tiff"])
	src_contents = os.walk(src_folder)
	dirpath, _, fnames = src_contents.next()
	img_dir = os.path.split(dirpath)[-1]
	img_files = [os.path.join(dirpath, name) for name in fnames]
	if shuffle:
		random.shuffle(img_files)
	if n_images:
		img_files = img_files[:n_images]
	images = [cv2.imread(name, cv2.IMREAD_GRAYSCALE) for name in 
			  img_files[:n_images] if os.path.splitext(name)[-1][1:].lower() 
			  in valid_extensions]
	if shuffle:
		random.shuffle(images)
	return images


################################################################################
#                                    Main                                      #
################################################################################

if __name__ == "__main__":
	image_dir = src_folder + "images/"
	bad_images_dir = src_folder + "bad_images/"
	data_source_str = 'exp=mfx11616:run=69:smd'
	data_source_str_idx = 'exp=mfx11616:run=69:idx'
	detector_str = 'BeamSpec'
	run = ds.runs().next()
	# timestamps = get_all_timestamps(data_source_str, detector_str)
	n_images = 5
	nevent_start = 23700
	nevent_end = 24350

	x, y = get_all_centroids(data_source_str, detector_str)
	plot_x_y_drift(x,y)


	# all_x, all_y = [], []
	# images = get_images_from_dir(image_dir,n_images=n_images, shuffle=True)
	# for image in images:

	# 	try:
	# 		centroid, bounding_box = find_beam(image)
	# 		all_x.append(centroid[0])
	# 		all_y.append(centroid[1])
	# 	except TypeError:
	# 		all_x.append(None)
	# 		all_y.append(None)

	# print all_x
	# print all_y

		# plot_image(image, centroid=centroid, bounding_box=bounding_box)
	
	# ds = psana.DataSource(data_source_str_idx)
	# det = psana.Detector(detector_str)
	# run = ds.runs().next()
	# beam, no_beam, none_type = [], [], []
	# nevents = get_n_nevents(n_images)

	# for nevent in tqdm(nevents):
	# 	event = get_event_from_nevent(ds, run, nevent, timestamps)
	# 	try:
	# 		image = det.image(event)
	# 		image = to_uint8(image)
	# 		image_prep = preprocess(image, resize=rsz_default, kernel=kernel_size)
	# 		contour = get_contour(image_prep)
	# 		M = get_image_moments(contour=contour)
	# 		if check_for_beam(M):
	# 			beam.append(nevent)
	# 		else:
	# 			no_beam.append(nevent)
	# 	except (TypeError, AttributeError):
	# 		none_type.append(nevent)

	# beam, no_beam, none_type, all_x, all_y = classify_and_plot_events(
	# 	data_source_str, detector_str)

	# plot = Process(target=plot_x_y_drift, args=(all_x, all_y))
	# plot.start()

			
	# # print "Total number of events:               {0}".format(n_images)
	# print "Number of events with beam:           {0}".format(len(beam))
	# print "Number of events with no beam:        {0}".format(len(no_beam))
	# print "Number of events with NoneType image: {0}".format(len(none_type))

	# ds = psana.DataSource(data_source_str_idx)	
	# det = psana.Detector(detector_str)
	# run = ds.runs().next()
	# for nevent in tqdm(no_beam):
	# 	event = get_event_from_nevent(ds, run, nevent, timestamps)
	# 	image = det.image(event)
	# 	print_image_info(image)
	# 	_, _ = find_beam(image, plot=True)



	# beam_events, no_beam_events, none_type_events = get_beam_nobeam_nonetype(
	# 	data_source_str, detector_str)
	# print "Number of events with beam:           {0}".format(len(beam_events))
	# print "Number of events with no beam:        {0}".format(len(no_beam_events))
	# print "Number of events with NoneType image: {0}".format(len(none_type_events))
	# ds = psana.DataSource(data_source_str_idx)	
	# det = psana.Detector(detector_str)
	# run = ds.runs().next()
	# for nevent in tqdm(no_beam_events):
	# 	event = get_event_from_nevent(ds, run, nevent, timestamps)
	# 	image = det.image(event)
	# 	print_image_info(image)
	# 	_, _ = find_beam(image, plot=True)


	
	# ds = psana.DataSource(data_source_str_idx)	
	# det = psana.Detector(detector_str)
	# run = ds.runs().next()
	# for nevent in tqdm( range(nevent_start, nevent_end)):
	# 	print timestamps[nevent]
	# 	event = get_event_from_nevent(ds, run, nevent, timestamps)
	# 	image = det.image(event)
	# 	diag(image)


	# images = get_images_from_dir(image_dir,n_images=n_images, shuffle=True)
	# for image in images:
	# 	find_beam(image, plot=True)

