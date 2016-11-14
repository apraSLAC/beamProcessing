# ASSIGNMENT 7
# Abdullah P Rashed Ahmed

import numpy as np
import scipy as sp
import cv2

from cv2 import ORB

""" Assignment 7 - Feature Detection and Matching

This file has a number of functions that you need to fill out in order to
complete the assignment. Please write the appropriate code, following the
instructions on which functions you may or may not use.

GENERAL RULES:
    1. DO NOT INCLUDE code that saves, shows, displays, writes the image that
    you are being passed in. Do that on your own if you need to save the images
    but the functions should NOT save the image to file. Thanks.

    2. DO NOT import any other libraries aside from those that we provide. You
    may not import anything else, you should be able to complete the assignment
    with the given libraries (and in most cases without them).

    3. DO NOT change the format of this file. Do not put functions into
    classes, or your own infrastructure. This makes grading very difficult for
    us. Please only write code in the allotted region.
"""

def findMatchesBetweenImages(image_1, image_2):
    """ Return the top 10 list of matches between two input images.

    This function detects and computes ORB (or SIFT) features from the
    input images, and returns the best matches using the normalized Hamming
    Distance.

    Follow these steps:
    1. Compute ORB keypoints and descriptors for both images
    2. Create a Brute Force Matcher, using the hamming distance (and set
       crossCheck to true).
    3. Compute the matches between both images.
    4. Sort the matches based on distance so you get the best matches.
    5. Return the image_1 keypoints, image_2 keypoints, and the top 10 matches
       in a list.

    Note: We encourage you use OpenCV functionality (also shown in lecture) to
    complete this function.

    Args:
    ----------
        image_1 : numpy.ndarray
            The first image (grayscale).

        image_2 : numpy.ndarray
            The second image. (grayscale).

    Returns:
    ----------
        image_1_kp : list
            The image_1 keypoints, the elements are of type cv2.KeyPoint.

        image_2_kp : list
            The image_2 keypoints, the elements are of type cv2.KeyPoint.

        matches : list
            A list of matches, length 10. Each item in the list is of type
            cv2.DMatch.
    """
    matches = None       # type: list of cv2.DMath
    image_1_kp = None    # type: list of cv2.KeyPoint items
    image_1_desc = None  # type: numpy.ndarray of numpy.uint8 values.
    image_2_kp = None    # type: list of cv2.KeyPoint items.
    image_2_desc = None  # type: numpy.ndarray of numpy.uint8 values.

    # WRITE YOUR CODE HERE.
    orb = ORB()
    image_1_kp, image_1_desc = orb.detectAndCompute(image_1, None)
    image_2_kp, image_2_desc = orb.detectAndCompute(image_2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(image_1_desc, image_2_desc)
    matches = sorted(matches, key = lambda x:x.distance)
    matches = matches[:10]

    # We coded the return statement for you. You are free to modify it -- just
    # make sure the tests pass.
    return image_1_kp, image_2_kp, matches


def drawMatches(image_1, image_1_keypoints, image_2, image_2_keypoints, matches):
    """ Draw the matches between the image_1 and image_2.

    NOTE: See docs for cv2 drawing functions:
        http://docs.opencv.org/2.4/modules/core/doc/drawing_functions.html

    Follow these steps:
        1. Mark the location of each matched keypoint on both image_1 and
           image_2 (e.g., cv2.circle, cv2.square)

        2. Fit the annotated image_1 and image_2 side-by-side in a single image

        3. Draw a line from each keypoint in image_1 to the corresponding
           endpoint in image_2. Given a cv2.DMatch object `match` and a list
           of cv2.KeyPoint objects `keypoints`, the pixel position in the
           first image where the matched keypoint was found is the queryIdx,
           and the corresponding position in the second image is the trainIdx:

               keypoints[match.queryIdx].pt  (location in first image)
               keypoints[match.trainIdx].pt  (location in second image)

    NOTE: OpenCV has functions to draw matches and keypoints, but they do
          not work on all OpenCV versions, and they are not available on
          the VM, so you MAY NOT use those functions. The list of known
          disabled functions includes:

          ['cv2.drawKeypoints', 'cv2.drawMatches', 'cv2.drawMatchesKnn']

    NOTE: This function is not graded by the autograder, but it is tested
          by the autograder to check for calls to disabled functions from
          openCV. It will be scored manually by the TAs.

    Args:
    ----------
        image_1 : numpy.ndarray
            The first image (can be color or grayscale).

        image_1_keypoints : list
            The image_1 keypoints, the elements are of type cv2.KeyPoint.

        image_2 : numpy.ndarray
            The image to search in (can be color or grayscale).

        image_2_keypoints : list
            The image_2 keypoints, the elements are of type cv2.KeyPoint.

    Returns:
    ----------
        output : numpy.ndarray
            An array containing both input images on a single canvas, with
            marks indicating matching features with lines connecting the
            matches.
    """

    if len(image_1.shape) == 3:
        nChannels = image_1.shape[2]
    else: nChannels = 1

    output = np.zeros((max(image_1.shape[0], image_2.shape[0]), image_1.shape[1] +
                       image_2.shape[1], 3))
    if nChannels == 1:
        for i in range(3):
            output[:image_1.shape[0], :image_1.shape[1], i] = image_1
            output[:image_2.shape[0], image_1.shape[1]:, i] = image_2
    else:
        output[:image_1.shape[0], :image_1.shape[1]] = image_1
        output[:image_2.shape[0], image_1.shape[1]:] = image_2

    for match in matches:
        image_1_point = (int(image_1_keypoints[match.queryIdx].pt[0]),
                         int(image_1_keypoints[match.queryIdx].pt[1]))
        image_2_point = (int(image_2_keypoints[match.trainIdx].pt[0] + image_1.shape[1]),
                         int(image_2_keypoints[match.trainIdx].pt[1]))

        cv2.circle(output, image_1_point, 5, (0, 0, 255))
        cv2.circle(output, image_2_point, 5, (0, 255, 0))
        cv2.line(output, image_1_point, image_2_point, (255, 0, 0), thickness = 3)
      
    return output


