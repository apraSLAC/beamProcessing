import cv2
import numpy as np
from matplotlib import pyplot as plt

################################################################################
#                                Default Globals                               #
################################################################################

MIN_MATCH_COUNT = 10
DEFAULT_FD = cv2.ORB_create

################################################################################
#                                    OpenCV                                    #
################################################################################

def get_kp_desc(image, kp_detector=DEFAULT_FD, mask=None):
    return kp_detector(image, mask)

def get_sample_kp(sample=None, load=None):
	pass

def find_matches(kp1, kp2):
	pass

def get_homography():
	pass

def get_beam_region():
	pass

def warp_image():
	pass

################################################################################
#                               Helper Functions                               #
################################################################################


################################################################################
#                                    Main                                      #
################################################################################

if __name__ == "__main__":
