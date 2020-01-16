# import the necessary packages
import numpy as np
import argparse
import cv2


if __name__ == "__main__":

    # load the image
    image = cv2.imread("/home/lennart/dorna/camera/images/calibrated.bmp")

    # define the list of boundaries
    boundaries = [([0, 69, 205], [0, 140, 255])]

    # loop over the boundaries
    for (lower, upper) in boundaries:
        # create NumPy arrays from the boundaries
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")

        # find the colors within the specified boundaries and apply
        # the mask
        mask = cv2.inRange(image, lower, upper)
        output = cv2.bitwise_and(image, image, mask=mask)

        # show the images
        cv2.imshow("images", np.hstack([image, output]))
        cv2.waitKey(0)

