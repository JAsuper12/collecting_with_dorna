from pyueye import ueye
import cv2
import numpy as np
import json
import math as m
from pyzbar import pyzbar


class Camera:
    def __init__(self):
        # Variabeln für Kameraverarbeitung
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

        # laden der kalibrierten Matrix
        self.load_calibration_config()

        # Variabeln zur Farberkennung
        self.found_blue_container = False
        self.found_red_container = False
        self.blue_contours_rectangle = []
        self.red_contours_rectangle = []
        blue_lower = [51, 0, 0]
        blue_upper = [255, 62, 62]
        red_lower = [0, 0, 51]
        red_upper = [62, 62, 255]
        self.blue_boundaries = [(blue_lower, blue_upper)]
        self.red_boundaries = [(red_lower, red_upper)]
        self.cX = None
        self.cY = None
        self.cX_container = None
        self.cY_container = None

        # Variablen zur Positionierung
        self.blue_container_world_position = []
        self.red_container_world_position = []
        self.blue_ball_position = []
        self.red_ball_position = []
        self.blue_container_position = []
        self.red_container_position = []
        self.qr_centres = []
        self.world_points = []
        self.qr_codes = None
        self.image_points_of_qr_codes = None
        self.localization_qr_codes = []
        self.world_localization_qr_codes = []
        self.increment = 0.25
        self.results = 0

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
            self.new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.dist_coeff, size, 1, size)
            dst = cv2.undistort(frame, self.camera_matrix, self.dist_coeff, None, self.new_camera_matrix)
            x, y, w, h = roi
            self.dst = dst[y:y + h, x:x + w]

            self.detect_colors()
            self.extrinsic_calibration()
            cv2.imshow("camera", self.dst)
            cv2.imshow("blue_only", self.show_blue_color)
            cv2.imshow("red_only", self.show_red_color)

            # Kamera schließen
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            elif cv2.waitKey(100) & 0xFF == ord('t'):
                cv2.imwrite("C:\dorna\camera\images\image.bmp", self.dst)
                cv2.imwrite("C:\dorna\camera\images\blue_mask.bmp", self.show_blue_color)

            elif cv2.waitKey(100) & 0xFF == ord('l'):
                self.found_blue_container = False
                self.found_red_container = False
                self.blue_container_world_position.clear()
                self.red_container_world_position.clear()
                print("Behälterposition zurückgesetzt")

        ueye.is_FreeImageMem(self.h_cam, self.pcImageMemory, self.MemID)
        ueye.is_ExitCamera(self.h_cam)
        cv2.destroyAllWindows()

    def load_calibration_config(self):
        with open("C:\dorna\camera\camera_calibration_config.json", "r") as file:
            data = json.load(file)
            self.camera_matrix = np.array(data["camera_matrix"])
            self.dist_coeff = np.array(data["dist_coeff"])
            self.mean_error = data["mean_error"]

    def detect_colors(self):
        # create NumPy arrays from the blue_boundaries
        hsv = cv2.cvtColor(self.dst, cv2.COLOR_BGR2HSV)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        for (lower, upper) in self.blue_boundaries:
            lower = np.array(lower)
            upper = np.array(upper)

            self.blue_mask = cv2.inRange(bgr, lower, upper)

            self.show_blue_color = cv2.bitwise_and(bgr, bgr, mask=self.blue_mask)
        for (lower, upper) in self.red_boundaries:
            lower = np.array(lower)
            upper = np.array(upper)

            self.red_mask = cv2.inRange(bgr, lower, upper)

            self.show_red_color = cv2.bitwise_and(bgr, bgr, mask=self.red_mask)
        self.draw_blue_contours()
        self.draw_red_contours()

    def draw_blue_contours(self):
        self.blue_ball_position.clear()
        blue_contours_area = []
        blue_contours_circles = []
        blue_contours, hierarchy = cv2.findContours(self.blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2:]

        if not self.found_blue_container:
            self.blue_contours_rectangle.clear()

        # check area
        for con in blue_contours:
            area = cv2.contourArea(con)
            if 200 < area < 10000:
                blue_contours_area.append(con)

        # check if contour is of circular shape
        for con in blue_contours_area:
            perimeter = cv2.arcLength(con, True)
            area = cv2.contourArea(con)
            approx = cv2.approxPolyDP(con, 0.01 * perimeter, True)

            circularity = 4 * m.pi * (area / (perimeter ** 2))

            if len(approx) == 4 and not self.found_blue_container:
                # compute the bounding box of the contour
                self.blue_contours_rectangle.append(con)

            elif 0.8 < circularity:
                blue_contours_circles.append(con)

        for cnt in blue_contours_circles:
            M = cv2.moments(cnt)
            self.cX = int(M["m10"] / M["m00"])
            self.cY = int(M["m01"] / M["m00"]) + 5
            self.blue_ball_position.append((self.cX, self.cY))

            cv2.drawContours(self.dst, [cnt], 0, (0, 0, 255), 1)
            cv2.circle(self.dst, (self.cX, self.cY), 2, (0, 0, 255), -1)

        for cnt in self.blue_contours_rectangle:
            M = cv2.moments(cnt)
            self.cX_container = int(M["m10"] / M["m00"])
            self.cY_container = int(M["m01"] / M["m00"])
            self.blue_container_position.append((self.cX_container, self.cY_container))
            cv2.drawContours(self.dst, [cnt], 0, (0, 128, 255), 1)
            cv2.circle(self.dst, (self.cX_container, self.cY_container), 1, (0, 128, 255), -1)

    def draw_red_contours(self):
        self.red_ball_position.clear()
        red_contours_area = []
        red_contours_circles = []
        red_contours, hierarchy = cv2.findContours(self.red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2:]

        if not self.found_red_container:
            self.red_contours_rectangle.clear()

        # check area
        for con in red_contours:
            area = cv2.contourArea(con)
            if 200 < area < 10000:
                red_contours_area.append(con)

        # check if contour is of circular shape
        for con in red_contours_area:
            perimeter = cv2.arcLength(con, True)
            area = cv2.contourArea(con)
            approx = cv2.approxPolyDP(con, 0.01 * perimeter, True)

            circularity = 4 * m.pi * (area / (perimeter ** 2))

            if len(approx) == 4 and not self.found_red_container:
                # compute the bounding box of the contour
                self.red_contours_rectangle.append(con)

            elif 0.8 < circularity:
                red_contours_circles.append(con)

        for cnt in red_contours_circles:
            M = cv2.moments(cnt)
            self.cX = int(M["m10"] / M["m00"])
            self.cY = int(M["m01"] / M["m00"]) + 5
            self.red_ball_position.append((self.cX, self.cY))

            cv2.drawContours(self.dst, [cnt], 0, (255, 0, 0), 1)
            cv2.circle(self.dst, (self.cX, self.cY), 2, (255, 0, 0), -1)

        for cnt in self.red_contours_rectangle:
            M = cv2.moments(cnt)
            self.cX_container = int(M["m10"] / M["m00"])
            self.cY_container = int(M["m01"] / M["m00"])
            self.red_container_position.append((self.cX_container, self.cY_container))
            cv2.drawContours(self.dst, [cnt], 0, (0, 255, 0), 1)
            cv2.circle(self.dst, (self.cX_container, self.cY_container), 1, (0, 255, 0), -1)

    def extrinsic_calibration(self):
        self.qr_centres.clear()
        self.world_points.clear()
        self.localization_qr_codes.clear()
        self.world_localization_qr_codes.clear()
        self.qr_codes = pyzbar.decode(self.dst)
        # loop over the detected barcodes

        for qr in self.qr_codes:
            # extract the bounding box location of the barcode and draw
            # the bounding box surrounding the barcode on the image
            (x, y, w, h) = qr.rect

            centre = (x + int((w / 2)), y + int((h / 2)))
            self.qr_centres.append(centre)

            cv2.rectangle(self.dst, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.circle(self.dst, centre, 2, (0, 255, 0), -1)

            # the barcode data is a bytes world so if we want to draw it
            # on our output image we need to convert it to a string first
            data = qr.data.decode("utf-8")

            self.qr_decoder(data, centre)

            # draw the barcode data and barcode type on the image
            text = "{}".format(data)
            cv2.putText(self.dst, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        world_points = np.array(self.world_points, dtype="float")
        # prüfen, ob die qr Codes gut verteilt sind
        positive_qr_codes = 0
        negative_qr_codes = 0
        for coordinates in world_points:
            if coordinates[0] > 0:
                positive_qr_codes = positive_qr_codes + 1
            else:
                negative_qr_codes = negative_qr_codes + 1

        if positive_qr_codes >= 3 and negative_qr_codes >= 3:
            valid_qr_spread = True
        else:
            valid_qr_spread = False

        if len(self.qr_centres) >= 6 and valid_qr_spread:
            self.image_points_of_qr_codes = np.array(self.qr_centres, dtype="float")

            _, self.rvecs, self.tvecs = cv2.solvePnP(world_points, self.image_points_of_qr_codes, self.camera_matrix, self.dist_coeff)

            self.rmatrix = cv2.Rodrigues(self.rvecs)[0]

            self.origin, jacobian = cv2.projectPoints(np.array([(0.0, 0.0, 0.0)]), self.rvecs, self.tvecs, self.camera_matrix, self.dist_coeff)
            z_axis, jacobian = cv2.projectPoints(np.array([(0.0, 0.0, 55.0)]), self.rvecs, self.tvecs, self.camera_matrix, self.dist_coeff)
            x_axis, jacobian = cv2.projectPoints(np.array([(55.0, 0.0, 0.0)]), self.rvecs, self.tvecs, self.camera_matrix, self.dist_coeff)
            y_axis, jacobian = cv2.projectPoints(np.array([(0.0, 55.0, 0.0)]), self.rvecs, self.tvecs, self.camera_matrix, self.dist_coeff)
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
            cv2.imwrite("C:\dorna\camera\images\im.bmp", self.dst)
            self.get_blue_ball_position_with_grid()
            self.get_red_ball_position_with_grid()

    def qr_decoder(self, data, centre):
        if data == "(70, 10, 0)":
            self.world_points.append((70, 10, 0))
            if len(self.localization_qr_codes) < 3:
                self.localization_qr_codes.append(centre)
                self.world_localization_qr_codes.append([70, 10, 0])
        elif data == "(60, 40, 0)":
            self.world_points.append((60, 40, 0))
            if len(self.localization_qr_codes) < 3:
                self.localization_qr_codes.append(centre)
                self.world_localization_qr_codes.append([60, 40, 0])
        elif data == "(40, 60, 0)":
            self.world_points.append((40, 60, 0))
            if len(self.localization_qr_codes) < 3:
                self.localization_qr_codes.append(centre)
                self.world_localization_qr_codes.append([40, 60, 0])
        elif data == "(-40, 60, 0)":
            self.world_points.append((-40, 60, 0))
            if len(self.localization_qr_codes) < 3:
                self.localization_qr_codes.append(centre)
                self.world_localization_qr_codes.append([-40, 60, 0])
        elif data == "(-60, 40, 0)":
            self.world_points.append((-60, 40, 0))
            if len(self.localization_qr_codes) < 3:
                self.localization_qr_codes.append(centre)
                self.world_localization_qr_codes.append([-60, 40, 0])
        elif data == "(-70, 10, 0)":
            self.world_points.append((-70, 10, 0))
            if len(self.localization_qr_codes) < 3:
                self.localization_qr_codes.append(centre)
                self.world_localization_qr_codes.append([-70, 10, 0])
        elif data == "q":
            self.world_points.append((65, 25, 0))
            if len(self.localization_qr_codes) < 3:
                self.localization_qr_codes.append(centre)
                self.world_localization_qr_codes.append([65, 25, 0])
        elif data == "a":
            self.world_points.append((57.5, 55, 0))
            if len(self.localization_qr_codes) < 3:
                self.localization_qr_codes.append(centre)
                self.world_localization_qr_codes.append([57.5, 55, 0])
        elif data == "b":
            self.world_points.append((20, 65, 0))
            if len(self.localization_qr_codes) < 3:
                self.localization_qr_codes.append(centre)
                self.world_localization_qr_codes.append([20, 65, 0])
        elif data == "c":
            self.world_points.append((-20, 65, 0))
            if len(self.localization_qr_codes) < 3:
                self.localization_qr_codes.append(centre)
                self.world_localization_qr_codes.append([-20, 65, 0])
        elif data == "d":
            self.world_points.append((-55, 55, 0))
            if len(self.localization_qr_codes) < 3:
                self.localization_qr_codes.append(centre)
                self.world_localization_qr_codes.append([-55, 55, 0])
        elif data == "p":
            self.world_points.append((-65, 25, 0))
            if len(self.localization_qr_codes) < 3:
                self.localization_qr_codes.append(centre)
                self.world_localization_qr_codes.append([-65, 25, 0])
        elif data == "e":
            self.world_points.append((55, -10, 0))
            if len(self.localization_qr_codes) < 3:
                self.localization_qr_codes.append(centre)
                self.world_localization_qr_codes.append([55, -10, 0])
        elif data == "f":
            self.world_points.append((35, -10, 0))
            if len(self.localization_qr_codes) < 3:
                self.localization_qr_codes.append(centre)
                self.world_localization_qr_codes.append([35, -10, 0])
        elif data == "g":
            self.world_points.append((15, -10, 0))
            if len(self.localization_qr_codes) < 3:
                self.localization_qr_codes.append(centre)
                self.world_localization_qr_codes.append([15, -10, 0])
        elif data == "h":
            self.world_points.append((-15, -10, 0))
            if len(self.localization_qr_codes) < 3:
                self.localization_qr_codes.append(centre)
                self.world_localization_qr_codes.append([-15, -10, 0])
        elif data == "i":
            self.world_points.append((-35, -10, 0))
            if len(self.localization_qr_codes) < 3:
                self.localization_qr_codes.append(centre)
                self.world_localization_qr_codes.append([-35, -10, 0])
        elif data == "j":
            self.world_points.append((-55, -10, 0))
            if len(self.localization_qr_codes) < 3:
                self.localization_qr_codes.append(centre)
                self.world_localization_qr_codes.append([-55, -10, 0])
        elif data == "k":
            self.world_points.append((47.5, 10, 0))
            if len(self.localization_qr_codes) < 3:
                self.localization_qr_codes.append(centre)
                self.world_localization_qr_codes.append([47.5, 10, 0])
        elif data == "l":
            self.world_points.append((0, 35, 0))
            if len(self.localization_qr_codes) < 3:
                self.localization_qr_codes.append(centre)
                self.world_localization_qr_codes.append([0, 35, 0])

    def get_ball_position_with_circles(self):
        if len(self.localization_qr_codes) == 3:
            distance_balls_to_qr_code = []

            i = 0
            for (ball_x, ball_y) in self.blue_ball_position:
                distance = []
                n_distance = 0
                for (qr_x, qr_y) in self.localization_qr_codes:
                    distance.append(m.sqrt(m.pow((ball_x - qr_x), 2) + m.pow((ball_y - qr_y), 2)))
                    if i == 0:
                        cv2.circle(self.dst, (qr_x, qr_y), int(distance[n_distance]), (255, 0, 0), 2)
                    elif i == 1:
                        cv2.circle(self.dst, (qr_x, qr_y), int(distance[n_distance]), (0, 255, 0), 2)
                    elif i == 2:
                        cv2.circle(self.dst, (qr_x, qr_y), int(distance[n_distance]), (0, 0, 255), 2)
                    n_distance = n_distance + 1
                distance_balls_to_qr_code.append(distance)
                i = i + 1

            if len(self.blue_ball_position) > 0:
                math_error = False
                world_distance_origin_to_qr_code = []
                for [x, y, z] in self.world_localization_qr_codes:
                    world_distance_origin_to_qr_code.append(m.sqrt(x ** 2 + y ** 2))

                image_distance_origin_to_qr_code = []
                for [u, v] in self.localization_qr_codes:
                    image_distance_origin_to_qr_code.append(m.sqrt((u - self.origin[0][0][0]) ** 2 + (v - self.origin[0][0][1]) ** 2))

                cm_per_pixel = []
                for distance in range(3):
                    cm_per_pixel.append(world_distance_origin_to_qr_code[distance] / image_distance_origin_to_qr_code[distance])
                # print(cm_per_pixel)

                equations = []

                for j in range(len(self.world_localization_qr_codes)):
                    x = self.world_localization_qr_codes[j][0]
                    y = self.world_localization_qr_codes[j][1]
                    r = (distance_balls_to_qr_code[0][j] * cm_per_pixel[j])
                    equations.append([x, y, r])

                x_y_coordinates = []
                for i in range(3):
                    x_coordinates = []
                    y_coordinates = []
                    if i < 2:
                        a_1 = equations[i][0]
                        b_1 = equations[i][1]
                        r_1 = equations[i][2] + 1
                        a_2 = equations[i + 1][0]
                        b_2 = equations[i + 1][1]
                        r_2 = equations[i + 1][2] + 1
                    else:
                        a_1 = equations[i][0]
                        b_1 = equations[i][1]
                        r_1 = equations[i][2] + 1
                        a_2 = equations[0][0]
                        b_2 = equations[0][1]
                        r_2 = equations[0][2] + 1

                    a = a_1 - a_2

                    c = (r_2 ** 2 - r_1 ** 2 + b_1 ** 2 - b_2 ** 2 - a ** 2) / (2 * a)
                    d = -1 - (b_2 - b_1) ** 2 / a ** 2
                    e = 2 * b_1 - 2 * (b_2 - b_1) * c / a
                    f = r_1 ** 2 - b_1 ** 2 - c ** 2

                    p = e / d
                    q = f / d

                    try:
                        y_coordinates.append(-p / 2 + m.sqrt((p / 2) ** 2 - q))
                        y_coordinates.append(-p / 2 - m.sqrt((p / 2) ** 2 - q))
                    except ValueError:
                        math_error = True
                        break

                    for y in y_coordinates:
                        x_coordinates.append(m.sqrt(r_1 ** 2 - y ** 2 + 2 * b_1 * y - b_1 ** 2) + a_1)
                        x_coordinates.append(-m.sqrt(r_1 ** 2 - y ** 2 + 2 * b_1 * y - b_1 ** 2) + a_1)

                    for j in range(4):
                        if j < 2:
                            x_y_coordinates.append([x_coordinates[j], y_coordinates[0]])
                        else:
                            x_y_coordinates.append([x_coordinates[j], y_coordinates[1]])

                if not math_error:
                    index_error = False
                    coordinates_found = False
                    found_similar_x_coordinates = False
                    found_similar_y_coordinates = False
                    similar_y_coordinates = []
                    similar_coordinates = []
                    while not coordinates_found:
                        place_of_similar_x_coordinates = [0]
                        place_of_similar_y_coordinates = [0]
                        n = 1

                        if index_error:
                            break
                        while not found_similar_y_coordinates:
                            try:
                                difference = x_y_coordinates[0][1] - x_y_coordinates[n][1]
                            except IndexError:
                                index_error = True
                                break
                            if -5 < difference < 5:
                                place_of_similar_y_coordinates.append(n)
                                # print(place_of_similar_y_coordinates)
                            if len(place_of_similar_y_coordinates) < 6:
                                if n < len(x_y_coordinates) - 1:
                                    n = n + 1
                                else:
                                    removed = 0
                                    for y in place_of_similar_y_coordinates:
                                        x_y_coordinates.remove(x_y_coordinates[y - removed])
                                        removed = removed + 1
                                    similar_y_coordinates.clear()
                                    break
                            else:
                                for y in place_of_similar_y_coordinates:
                                    similar_y_coordinates.append(x_y_coordinates[y])
                                found_similar_y_coordinates = True

                            # print(similar_y_coordinates)

                        if found_similar_y_coordinates:
                            n = 1
                            while not found_similar_x_coordinates:
                                try:
                                    difference = similar_y_coordinates[0][0] - similar_y_coordinates[n][0]
                                except IndexError:
                                    index_error = True
                                    break
                                if -5 < difference < 5:
                                    place_of_similar_x_coordinates.append(n)
                                if len(place_of_similar_x_coordinates) < 3:
                                    if n < len(similar_y_coordinates) - 1:
                                        n = n + 1
                                    else:
                                        removed = 0
                                        for x in place_of_similar_x_coordinates:
                                            similar_y_coordinates.remove(similar_y_coordinates[x - removed])
                                            removed = removed + 1
                                        break
                                else:
                                    for x in place_of_similar_x_coordinates:
                                        similar_coordinates.append(similar_y_coordinates[x])
                                    found_similar_x_coordinates = True
                                    coordinates_found = True

                    if not index_error:
                        sum_x_coordinates = 0
                        sum_y_coordinates = 0
                        for i in range(3):
                            sum_x_coordinates = sum_x_coordinates + similar_coordinates[i][0]
                            sum_y_coordinates = sum_y_coordinates + similar_coordinates[i][1]

                        average_x_coordinate = sum_x_coordinates / 3
                        average_y_coordinate = sum_y_coordinates / 3

                        ball_coordinate = [average_x_coordinate, average_y_coordinate]

                        print(ball_coordinate)

    def get_blue_ball_position_with_grid(self):
        blue_ball_world_positions = []

        for ball in self.blue_ball_position:
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

            blue_ball_world_positions.append((x_difference[0][0], x_difference[0][1]))

        for balls in blue_ball_world_positions:
            image_coordinate, jacobian = cv2.projectPoints(
                np.array([(x_difference[0][0], x_difference[0][1], 0.0)]), self.rvecs, self.tvecs,
                self.camera_matrix, self.dist_coeff)
            estimated_point = (int(image_coordinate[0][0][0]), int(image_coordinate[0][0][1]))
            cv2.circle(self.dst, estimated_point, 2, (0, 0, 255), -1)

        if not self.found_blue_container and len(self.blue_container_position) > 0:
            y_coordinate_of_smallest_difference = []
            x_b, y_b = self.blue_container_position[0]
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

            self.blue_container_world_position.append((x_difference[0][0], x_difference[0][1]))
            self.found_blue_container = True

        print("Positionen der blauen Bälle: ", blue_ball_world_positions, "\nPosition des blauen Behälters: ", self.blue_container_world_position)

    def get_red_ball_position_with_grid(self):
        red_ball_world_positions = []

        for ball in self.red_ball_position:
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

            red_ball_world_positions.append((x_difference[0][0], x_difference[0][1]))

        for balls in red_ball_world_positions:
            image_coordinate, jacobian = cv2.projectPoints(
                np.array([(x_difference[0][0], x_difference[0][1], 0.0)]), self.rvecs, self.tvecs,
                self.camera_matrix, self.dist_coeff)
            estimated_point = (int(image_coordinate[0][0][0]), int(image_coordinate[0][0][1]))
            cv2.circle(self.dst, estimated_point, 2, (0, 0, 255), -1)

        if not self.found_red_container and len(self.red_container_position) > 0:
            y_coordinate_of_smallest_difference = []
            x_b, y_b = self.red_container_position[0]
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

            self.red_container_world_position.append((x_difference[0][0], x_difference[0][1]))
            self.found_red_container = True

        print("Positionen der roten Bälle: ", red_ball_world_positions, "\nPosition des roten Behälters: ", self.red_container_world_position)

    def get_ball_position_with_matrix(self):
        if len(self.blue_ball_position) > 0:
            u, v = self.blue_ball_position[0]
            uv_1 = np.array([[u, v, 1]], dtype=np.float32)
            uv_1 = uv_1.T
            inverse_newcam_mtx = np.linalg.inv(self.new_camera_matrix)
            inverse_R_mtx = np.linalg.inv(self.rmatrix)
            Rt = inverse_R_mtx.dot(self.tvecs)
            RM = inverse_R_mtx.dot(inverse_newcam_mtx)
            RMuv = RM.dot(uv_1)
            s = (0 + Rt[2]) / RMuv[2]
            a = s*inverse_newcam_mtx.dot(uv_1) - self.tvecs
            XYZ = inverse_R_mtx.dot(a)
            print(XYZ)


if __name__ == "__main__":
    camera = Camera()
    camera.show_image()
