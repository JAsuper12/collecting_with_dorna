from pyueye import ueye
import cv2
import numpy as np
import json
import math as m
from pyzbar import pyzbar


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

        blue_lower = [51, 0, 0]
        blue_upper = [255, 51, 51]
        self.boundaries = [(blue_lower, blue_upper)]
        self.cX = None
        self.cY = None


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

            self.qr_decoder()

            cv2.imshow("camera", self.dst)
            cv2.imshow("blue_only", self.only_certain_color)

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
        self.draw_contours()

    def draw_contours(self):
        self.ball_position = []
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
            circularity = 4 * m.pi * (area / (perimeter * perimeter))
            # print(circularity)
            if 0.7 < circularity < 1.5:
                contours_circles.append(con)

        for cnt in contours_circles:
            M = cv2.moments(cnt)
            self.cX = int(M["m10"] / M["m00"])
            self.cY = int(M["m01"] / M["m00"]) + 5
            self.ball_position.append((self.cX, self.cY))

            cv2.drawContours(self.dst, [cnt], 0, (0, 255, 0), 1)
            cv2.circle(self.dst, (self.cX, self.cY), 2, (0, 255, 0), -1)
        #print(self.ball_position)

    def qr_decoder(self):
        centres = []
        object_points_array = []
        self.localization_qr = []
        qr_codes = pyzbar.decode(self.dst)
        # loop over the detected barcodes
        for qr in qr_codes:
            # extract the bounding box location of the barcode and draw
            # the bounding box surrounding the barcode on the image
            (x, y, w, h) = qr.rect

            centre = (x + int((w / 2)), y + int((h / 2)))
            centres.append(centre)

            cv2.rectangle(self.dst, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.circle(self.dst, centre, 2, (0, 255, 0), -1)

            # the barcode data is a bytes object so if we want to draw it
            # on our output image we need to convert it to a string first
            data = qr.data.decode("utf-8")

            if data == "(40, 0, 0)":
                object_point = (40, 0, 0)
                #if len(self.localization_qr) < 3:
                    #self.localization_qr.append(centre)
            elif data == "(-40, 0, 0)":
                object_point = (-40, 0, 0)
                #if len(self.localization_qr) < 3:
                    #self.localization_qr.append(centre)
            elif data == "(30, 30, 0)":
                object_point = (30, 30, 0)
                if len(self.localization_qr) < 3:
                    self.localization_qr.append(centre)
            elif data == "(-30, 30, 0)":
                object_point = (-30, 30, 0)
                if len(self.localization_qr) < 3:
                    self.localization_qr.append(centre)
            elif data == "(0, 40, 0)":
                object_point = (0, 40, 0)
                if len(self.localization_qr) < 3:
                    self.localization_qr.append(centre)

            object_points_array.append(object_point)
            # draw the barcode data and barcode type on the image
            text = "{}".format(data)
            cv2.putText(self.dst, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        #print(self.localization_qr)

        if len(centres) >= 4:
             # print(object_points_array)
            self.image_points = np.array(centres, dtype="float")
            object_points = np.array(object_points_array, dtype="float")
            # print(object_points)
            # print(image_points)

            _, rvecs, tvecs, = cv2.solvePnP(object_points, self.image_points, self.camera_matrix, self.dist_coeff)

            origin, jacobian = cv2.projectPoints(np.array([(0.0, 0.0, 0.0)]), rvecs, tvecs, self.camera_matrix, self.dist_coeff)
            z_axis, jacobian = cv2.projectPoints(np.array([(0.0, 0.0, 10.0)]), rvecs, tvecs, self.camera_matrix, self.dist_coeff)
            x_axis, jacobian = cv2.projectPoints(np.array([(10.0, 0.0, 0.0)]), rvecs, tvecs, self.camera_matrix, self.dist_coeff)
            y_axis, jacobian = cv2.projectPoints(np.array([(0.0, 10.0, 0.0)]), rvecs, tvecs, self.camera_matrix, self.dist_coeff)

            axis = [x_axis, y_axis, z_axis]

            i = 0
            for x in axis:
                p1 = (int(origin[0][0][0]), int(origin[0][0][1]))
                p2 = (int(x[0][0][0]), int(x[0][0][1]))
                if i == 0:
                    self.dst = cv2.line(self.dst, p1, p2, (255, 0, 0), 5)
                elif i == 1:
                    self.dst = cv2.line(self.dst, p1, p2, (0, 255, 0), 5)
                elif i == 2:
                    self.dst = cv2.line(self.dst, p1, p2, (0, 0, 255), 5)
                i = i + 1

            self.get_ball_position()

    def get_ball_position(self):
        if len(self.localization_qr) == 3:
            i = 0
            for (ball_x, ball_y) in self.ball_position:
                for (qr_x, qr_y) in self.localization_qr:
                    distance = (m.sqrt(m.pow((ball_x - qr_x), 2) + m.pow((ball_y - qr_y), 2)))
                    if i == 0:
                        cv2.circle(self.dst, (qr_x, qr_y), int(distance), (255, 0, 0), 2)
                    elif i == 1:
                        cv2.circle(self.dst, (qr_x, qr_y), int(distance), (0, 255, 0), 2)
                    elif i == 2:
                        cv2.circle(self.dst, (qr_x, qr_y), int(distance), (0, 0, 255), 2)
                    print(distance)
                i = i + 1
            self.localization_qr = np.array(self.localization_qr)
            real_distance = m.sqrt(m.pow(self.localization_qr[1][0] - self.localization_qr[2][0], 2) + m.pow(self.localization_qr[1][1] - self.localization_qr[2][1], 2))
            print(real_distance)


if __name__ == "__main__":
    camera = Camera()
    camera.show_image()
