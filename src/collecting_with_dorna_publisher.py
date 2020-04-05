#! /usr/bin/env python

import rospy
from collecting_with_dorna.msg import Coordinates
from pyueye import ueye
import cv2
import numpy as np
import json
import math as m
from pyzbar import pyzbar
import os
import sys

class PublishCoordinates:
    def __init__(self):
        rospy.init_node("coordinates_publisher")
        self.coordinates_pub = rospy.Publisher("/coordinates", Coordinates, queue_size = 1)
        self.coordinates_msg = Coordinates()

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
        self.dst = None

        self.load_calibration_config()

        self.found_container = False
        self.contours_rectangle = []
        blue_lower = [51, 0, 0]
        blue_upper = [255, 62, 62]
        self.boundaries = [(blue_lower, blue_upper)]
        self.cX = None
        self.cY = None
        self.cX_container = None
        self.cY_container = None

        self.container_world_position = []
        self.ball_position = []
        self.container_position = []
        self.qr_centres = []
        self.world_points = []
        self.qr_codes = None
        self.image_points_of_qr_codes = None
        self.localization_qr_codes = []
        self.world_localization_qr_codes = []
        self.increment = 0.25

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

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            elif cv2.waitKey(1) & 0xFF == ord('t'):
                cv2.imwrite("/home/lennart/dorna/camera/images/gps.bmp", self.dst)

        ueye.is_FreeImageMem(self.h_cam, self.pcImageMemory, self.MemID)
        ueye.is_ExitCamera(self.h_cam)
        cv2.destroyAllWindows()

    def load_calibration_config(self):
        path = os.path.dirname(os.path.abspath(sys.argv[0]))
        path = path + "/config/camera_calibration_config.json"
        with open(path, "r") as file:
            data = json.load(file)
            self.camera_matrix = np.array(data["camera_matrix"])
            self.dist_coeff = np.array(data["dist_coeff"])
            self.mean_error = data["mean_error"]

    def detect_colors(self):
        hsv = cv2.cvtColor(self.dst, cv2.COLOR_BGR2HSV)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        for (lower, upper) in self.boundaries:
            lower = np.array(lower)
            upper = np.array(upper)

            self.mask = cv2.inRange(bgr, lower, upper)

            self.show_blue_color = cv2.bitwise_and(bgr, bgr, mask=self.mask)
        self.draw_contours()

    def draw_contours(self):
        self.ball_position.clear()
        contours_area = []
        contours_circles = []
        contours, hierarchy = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2:]

        if not self.found_container:
            self.contours_rectangle.clear()

        for con in contours:
            area = cv2.contourArea(con)
            if 200 < area < 10000:
                contours_area.append(con)

        for con in contours_area:
            perimeter = cv2.arcLength(con, True)
            area = cv2.contourArea(con)
            approx = cv2.approxPolyDP(con, 0.04 * perimeter, True)

            circularity = 4 * m.pi * (area / (perimeter * perimeter))

            if len(approx) == 4 and not self.found_container:
                # compute the bounding box of the contour
                self.contours_rectangle.append(con)

            elif 0.8 < circularity:
                contours_circles.append(con)

        for cnt in contours_circles:
            M = cv2.moments(cnt)
            self.cX = int(M["m10"] / M["m00"])
            self.cY = int(M["m01"] / M["m00"]) + 5
            self.ball_position.append((self.cX, self.cY))

            cv2.drawContours(self.dst, [cnt], 0, (0, 255, 0), 1)
            cv2.circle(self.dst, (self.cX, self.cY), 2, (0, 255, 0), -1)

        for cnt in self.contours_rectangle:
            M = cv2.moments(cnt)
            self.cX_container = int(M["m10"] / M["m00"])
            self.cY_container = int(M["m01"] / M["m00"])
            self.container_position.append((self.cX_container, self.cY_container))
            cv2.drawContours(self.dst, [cnt], 0, (0, 128, 255), 1)
            cv2.circle(self.dst, (self.cX_container, self.cY_container), 1, (0, 128, 255), -1)

    def qr_decoder(self):
        self.qr_centres.clear()
        self.world_points.clear()
        self.localization_qr_codes.clear()
        self.world_localization_qr_codes.clear()
        self.qr_codes = pyzbar.decode(self.dst)

        for qr in self.qr_codes:
            (x, y, w, h) = qr.rect

            centre = (x + int((w / 2)), y + int((h / 2)))
            self.qr_centres.append(centre)

            cv2.rectangle(self.dst, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.circle(self.dst, centre, 2, (0, 255, 0), -1)

            data = qr.data.decode("utf-8")

            if data == "(70, 10, 0)":
                world_point = (70, 10, 0)
                if len(self.localization_qr_codes) < 3:
                    self.localization_qr_codes.append(centre)
                    self.world_localization_qr_codes.append([70, 10, 0])
            elif data == "(60, 40, 0)":
                world_point = (60, 40, 0)
                if len(self.localization_qr_codes) < 3:
                    self.localization_qr_codes.append(centre)
                    self.world_localization_qr_codes.append([60, 40, 0])
            elif data == "(40, 60, 0)":
                world_point = (40, 60, 0)
                if len(self.localization_qr_codes) < 3:
                    self.localization_qr_codes.append(centre)
                    self.world_localization_qr_codes.append([40, 60, 0])
            elif data == "(-40, 60, 0)":
                world_point = (-40, 60, 0)
                if len(self.localization_qr_codes) < 3:
                    self.localization_qr_codes.append(centre)
                    self.world_localization_qr_codes.append([-40, 60, 0])
            elif data == "(-60, 40, 0)":
                world_point = (-60, 40, 0)
                if len(self.localization_qr_codes) < 3:
                    self.localization_qr_codes.append(centre)
                    self.world_localization_qr_codes.append([-60, 40, 0])
            elif data == "(-70, 10, 0)":
                world_point = (-70, 10, 0)
                if len(self.localization_qr_codes) < 3:
                    self.localization_qr_codes.append(centre)
                    self.world_localization_qr_codes.append([-70, 10, 0])

            self.world_points.append(world_point)

            text = "{}".format(data)
            cv2.putText(self.dst, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        world_points = np.array(self.world_points, dtype="float")

        positive_qr_codes = 0
        negative_qr_codes = 0
        for coordinates in world_points:
            if coordinates[0] > 0:
                positive_qr_codes = positive_qr_codes + 1
            else:
                negative_qr_codes = negative_qr_codes + 1

        if positive_qr_codes >= 2 and negative_qr_codes >= 2:
            valid_qr_spread = True
        else:
            valid_qr_spread = False

        if len(self.qr_centres) >= 4 and valid_qr_spread:
            self.image_points_of_qr_codes = np.array(self.qr_centres, dtype="float")

            _, self.rvecs, self.tvecs = cv2.solvePnP(world_points, self.image_points_of_qr_codes, self.camera_matrix, self.dist_coeff)

            self.origin, jacobian = cv2.projectPoints(np.array([(0.0, 0.0, 0.0)]), self.rvecs, self.tvecs, self.camera_matrix, self.dist_coeff)
            z_axis, jacobian = cv2.projectPoints(np.array([(0.0, 0.0, 10.0)]), self.rvecs, self.tvecs, self.camera_matrix, self.dist_coeff)
            x_axis, jacobian = cv2.projectPoints(np.array([(10.0, 0.0, 0.0)]), self.rvecs, self.tvecs, self.camera_matrix, self.dist_coeff)
            y_axis, jacobian = cv2.projectPoints(np.array([(0.0, 10.0, 0.0)]), self.rvecs, self.tvecs, self.camera_matrix, self.dist_coeff)
            axis = [x_axis, y_axis, z_axis]

            i = 0
            for x in axis:
                p1 = (int(self.origin[0][0][0]), int(self.origin[0][0][1]))
                p2 = (int(x[0][0][0]), int(x[0][0][1]))
                if i == 0:
                    self.dst = cv2.line(self.dst, p1, p2, (255, 0, 0), 5)
                elif i == 1:
                    self.dst = cv2.line(self.dst, p1, p2, (0, 255, 0), 5)
                elif i == 2:
                    self.dst = cv2.line(self.dst, p1, p2, (0, 0, 255), 5)
                i = i + 1
            self.get_ball_position_with_grid()

    def get_ball_position_with_grid(self):
        self.coordinates_msg.ball_coordinates = []
        ball_world_positions = []
        for ball in self.ball_position:
            y_coordinate_of_smallest_difference = []
            x_b, y_b = ball
            x = -60
            x_difference = []
            while x <= 60:
                y = -10
                y_difference = []
                while y <= 60:
                    image_coordinate, jacobian = cv2.projectPoints(np.array([(x, y, 0.0)]), self.rvecs, self.tvecs, self.camera_matrix, self.dist_coeff)
                    y_difference.append([y, y_b - image_coordinate[0][0][1]])

                    if len(y_difference) == 2:
                        if 0 < y_difference[0][1] < y_difference[1][1] or y_difference[1][1] < y_difference[0][1] < 0:
                            y_difference.remove(y_difference[1])
                        else:
                            y_difference.remove(y_difference[0])
                    y = y + self.increment
                y_coordinate_of_smallest_difference.append(y_difference[0][0])
                x = x + self.increment

            for y in range(len(y_coordinate_of_smallest_difference)):
                image_coordinate, jacobian = cv2.projectPoints(np.array([(y * self.increment - 60, y_coordinate_of_smallest_difference[y], 0.0)]), self.rvecs, self.tvecs, self.camera_matrix, self.dist_coeff)
                x_difference.append([y * self.increment - 60, y_coordinate_of_smallest_difference[y], x_b - image_coordinate[0][0][0]])

                if len(x_difference) == 2:
                    if 0 < x_difference[0][2] < x_difference[1][2] or x_difference[1][2] < x_difference[0][2] < 0:
                        x_difference.remove(x_difference[1])
                    else:
                        x_difference.remove(x_difference[0])

            ball_world_positions.append((x_difference[0][0], x_difference[0][1]))

        for balls in ball_world_positions:
            image_coordinate, jacobian = cv2.projectPoints(
                np.array([(x_difference[0][0], x_difference[0][1], 0.0)]), self.rvecs, self.tvecs,
                self.camera_matrix, self.dist_coeff)
            estimated_point = (int(image_coordinate[0][0][0]), int(image_coordinate[0][0][1]))
            cv2.circle(self.dst, estimated_point, 2, (0, 0, 255), -1)

        if not self.found_container and len(self.container_position) > 0:
            y_coordinate_of_smallest_difference = []
            x_b, y_b = self.container_position[0]
            x = -60
            x_difference = []
            while x <= 60:
                y = -10
                y_difference = []
                while y <= 60:
                    image_coordinate, jacobian = cv2.projectPoints(np.array([(x, y, 0.0)]), self.rvecs, self.tvecs,
                                                                   self.camera_matrix, self.dist_coeff)
                    y_difference.append([y, y_b - image_coordinate[0][0][1]])

                    if len(y_difference) == 2:
                        if 0 < y_difference[0][1] < y_difference[1][1] or y_difference[1][1] < y_difference[0][1] < 0:
                            y_difference.remove(y_difference[1])
                        else:
                            y_difference.remove(y_difference[0])
                    y = y + self.increment
                y_coordinate_of_smallest_difference.append(y_difference[0][0])
                x = x + self.increment

            for y in range(len(y_coordinate_of_smallest_difference)):
                image_coordinate, jacobian = cv2.projectPoints(
                    np.array([(y * self.increment - 60, y_coordinate_of_smallest_difference[y], 0.0)]), self.rvecs, self.tvecs,
                    self.camera_matrix, self.dist_coeff)
                x_difference.append(
                    [y * self.increment - 60, y_coordinate_of_smallest_difference[y], x_b - image_coordinate[0][0][0]])

                if len(x_difference) == 2:
                    if 0 < x_difference[0][2] < x_difference[1][2] or x_difference[1][2] < x_difference[0][2] < 0:
                        x_difference.remove(x_difference[1])
                    else:
                        x_difference.remove(x_difference[0])

            self.container_world_position.append((x_difference[0][0], x_difference[0][1]))
            self.found_container = True

        balls_array = []
        for ball in ball_world_positions:
            x_b, y_b = ball
            balls_array.append(x_b)
            balls_array.append(y_b)
        self.coordinates_msg.ball_coordinates = balls_array
        for container in self.container_world_position:
            self.coordinates_msg.container_coordinates = container

        self.coordinates_pub.publish(self.coordinates_msg)

if __name__ == '__main__':
    publisher = PublishCoordinates()
    publisher.show_image()