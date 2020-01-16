from pyueye import ueye
import cv2
import numpy as np
import json
import math


class Camera:
    def __init__(self):
        self.h_cam = ueye.HIDS(0)
        self.pitch = ueye.INT()
        self.ColorMode = ueye.IS_CM_BGRA8_PACKED
        self.nBitsPerPixel = ueye.INT(32)
        self.bytes_per_pixel = int(self.nBitsPerPixel / 8)
        self.pitch = ueye.INT()
        self.pcImageMemory = ueye.c_mem_p()
        self.rectAOI = ueye.IS_RECT()
        self.MemID = ueye.int()
        self.width = None
        self.height = None

        self.load_calibration_config()

        orange_lower = [0, 25, 51]
        orange_upper = [25, 140, 255]
        green_lower = [0, 51, 0]
        green_upper = [102, 255, 102]
        blue_lower = [51, 0, 0]
        blue_upper = [255, 102, 102]
        self.boundaries = [(blue_lower, blue_upper)]


    def show_image(self):
        nRet = ueye.is_InitCamera(self.h_cam, None)
        nRet = ueye.is_SetDisplayMode(self.h_cam, ueye.IS_SET_DM_DIB)
        nRet = ueye.is_AOI(self.h_cam, ueye.IS_AOI_IMAGE_GET_AOI, self.rectAOI, ueye.sizeof(self.rectAOI))

        self.width = self.rectAOI.s32Width
        self.height = self.rectAOI.s32Height

        nRet = ueye.is_AllocImageMem(self.h_cam, self.width, self.height, self.nBitsPerPixel, self.pcImageMemory,
                                     self.MemID)
        nRet = ueye.is_SetImageMem(self.h_cam, self.pcImageMemory, self.MemID)
        nRet = ueye.is_SetColorMode(self.h_cam, self.ColorMode)
        nRet = ueye.is_CaptureVideo(self.h_cam, ueye.IS_DONT_WAIT)
        nRet = ueye.is_InquireImageMem(self.h_cam, self.pcImageMemory, self.MemID, self.width, self.height,
                                       self.nBitsPerPixel, self.pitch)

        while nRet == ueye.IS_SUCCESS:
            array = ueye.get_data(self.pcImageMemory, self.width, self.height, self.nBitsPerPixel, self.pitch, copy=False)
            frame = np.reshape(array, (self.height.value, self.width.value, self.bytes_per_pixel))
            frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            size = (self.height, self.width)
            new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.dist_coeff, size, 1, size)
            dst = cv2.undistort(frame, self.camera_matrix, self.dist_coeff, None, new_camera_matrix)
            x, y, w, h = roi
            self.dst = dst[y:y + h, x:x + w]

            self.detect_colors()
            self.draw_contours()

            cv2.imshow("camera", self.dst)
            cv2.imshow("only_orange", self.only_certain_color)

            # Kamera schließen
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            elif cv2.waitKey(1) & 0xFF == ord('t'):
                cv2.imwrite("/home/lennart/dorna/camera/images/contours.bmp", self.dst)

        ueye.is_FreeImageMem(self.h_cam, self.pcImageMemory, self.MemID)
        ueye.is_ExitCamera(self.h_cam)
        cv2.destroyAllWindows()

    def load_calibration_config(self):
        with open("/home/lennart/dorna/camera/camera_calibration_config.json", "r") as file:
            data = json.load(file)
            self.camera_matrix = np.array(data["camera_matrix"])
            self.dist_coeff = np.array(data["dist_coeff"])
            self.mean_error = data["mean_error"]
            # print(self.camera_matrix)
            # print(self.dist_coeff)

    def detect_colors(self):
        # create NumPy arrays from the boundaries
        hsv = cv2.cvtColor(self.dst, cv2.COLOR_BGR2HSV)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        for (lower, upper) in self.boundaries:
            lower = np.array(lower)
            upper = np.array(upper)

            self.mask = cv2.inRange(bgr, lower, upper)

            self.only_certain_color = cv2.bitwise_and(bgr, bgr, mask=self.mask)

    def draw_contours(self):
        contours_area = []
        contours_circles = []
        contours, hierarchy = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2:]

        # calculate area and filter into new array
        for con in contours:
            area = cv2.contourArea(con)
            if 500 < area < 10000:
                contours_area.append(con)

        # check if contour is of circular shape
        for con in contours_area:
            perimeter = cv2.arcLength(con, True)
            area = cv2.contourArea(con)
            if perimeter == 0:
                break
            circularity = 4 * math.pi * (area / (perimeter * perimeter))
            # print(circularity)
            if 0.7 < circularity < 1.5:
                contours_circles.append(con)

        for cnt in contours_circles:
            M = cv2.moments(cnt)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.drawContours(self.dst, [cnt], 0, (0, 255, 0), 1)
            cv2.circle(self.dst, (cX, cY + 5), 2, (0, 255, 0), -1)


if __name__ == "__main__":
    camera = Camera()
    camera.show_image()