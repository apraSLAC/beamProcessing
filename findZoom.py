import cv2
import numpy as np
from matplotlib import pyplot as plt
from beamDetector import detect, to_uint8

################################################################################
#                                Default Globals                               #
################################################################################

DEFAULT_FD = 
kernel_size  = (9,9)

################################################################################
#                                    OpenCV                                    #
################################################################################

def preprocess(image, resize=rsz_default, kernel=kernel_size, sigma=0):
	"""Preprocess the image for resizing,noise reduction, etc"""
	image = to_uint8(image)
	image_small = cv2.resize(image, (0,0), fx=resize, fy=resize)
	image_gblur = cv2.GaussianBlur(image_small, kernel, sigma)
	return image_gblur

def get_kp_desc(image, feature_detector=cv2.ORB_create, mask=None):
    """
    Returns key points and descriptors of the inputted image using the inputted
    feature detector. 
    
    If no feature detector is inputted, the function is rerun using a default 
    feature dectector.
    """
    return kp_detector.detectAndCompute(image, mask)

def get_sample_kp(sample=None, load=None):
	"""
	Returns the key points and descriptors for the sample image using a preloaded
	file or an inputted image.
	
	If an image is inputted, then its key points and descriptors are saved for 
	later use.
	"""
	pass

def find_matches(kp1, kp2, matcher=cv2.FlannBasedMatcher, algorithm=0, trees=5, 
                 checks=50, min_n_matches = 10):
	"""
	Tries to find MIN_MATCH_COUNT matches between the two inputted keypoints. 
	
	If that number of matches are found, the source points and destination points
	are returned. Otherwise, None is returned for both.
	"""
	index_params = dict(algorithm = algorithm, trees = trees)
	search_params = dict(checks = checks)
	matcher = matcher(index_params, search_params)
	matches = matcher.knnMatch(des1,des2,k=2)
	good_matches = [m if m.distance < 0.7*n.distance for m,n in matches]
	if len(good_matches) > min_n_matches:
		src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])
		dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])
		return src_pts.reshape(-1,1,2), dst_pts.reshape(-1,1,2)
	else:
		print "Not enough matches were found: {0}/{1}".format(
			len(good_matches), min_n_matches)
		return None, None
		                                                      
def get_homography(src_pts, dst_pts, method=cv2.RANSAC, ransacRepojThresh=5.0):
	"""
	Computes the homography between the source points and the destination points
	and returns M and the flattened mask.
	"""
	M, mask = cv2.findHomography(src_pts, dst_pts, method, ransacRepojThresh)
	matches_mask = mask.ravel().tolist()
	return M, matches_mask

def warp_points(points, M):
	"""
	Warps the inputted points using the homography. This can be applied to an 
	entire image or individual points.
	"""
    return cv2.perspectiveTransform(points, M)

def get_zoom()
	pass


################################################################################
#                               Helper Functions                               #
################################################################################


################################################################################
#                                    Main                                      #
################################################################################

if __name__ == "__main__":
