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
        blue_lower = [51, 0, 0]
        blue_upper = [255, 62, 62]
        self.boundaries = [(blue_lower, blue_upper)]
        self.cX = None
        self.cY = None

        # Variablen zur Positionierung
        self.ball_position = []
        self.qr_centres = []
        self.world_points = []
        self.qr_codes = None
        self.image_points_of_qr_codes = None
        self.localization_qr_codes = []
        self.world_localization_qr_codes = []

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
            cv2.imshow("blue_only", self.show_blue_color)

            # Kamera schließen
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            elif cv2.waitKey(1) & 0xFF == ord('t'):
                cv2.imwrite("/home/lennart/dorna/camera/images/gps.bmp", self.dst)

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

            self.show_blue_color = cv2.bitwise_and(bgr, bgr, mask=self.mask)
        self.draw_contours()

    def draw_contours(self):
        self.ball_position.clear()
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
            if 0.25 < circularity < 1.5:
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
            world_points = np.array(self.world_points, dtype="float")

            # draw the barcode data and barcode type on the image
            text = "{}".format(data)
            cv2.putText(self.dst, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # prüfen, ob die qr Codes gut verteilt sind
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

        # print(self.world_points)
        # print(self.localization_qr_codes)

        if len(self.qr_centres) >= 4 and valid_qr_spread:
            # print(world_points)
            self.image_points_of_qr_codes = np.array(self.qr_centres, dtype="float")
            # print(world_points)
            # print(self.image_points_of_qr_codes)

            _, rvecs, tvecs = cv2.solvePnP(world_points, self.image_points_of_qr_codes, self.camera_matrix, self.dist_coeff)

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
        if len(self.localization_qr_codes) == 3:
            world_distance_of_qr_codes = []
            for i in range(len(self.world_points)):
                x_1, y_1, z_1 = self.world_points[i]
                if i < len(self.world_points) - 1:
                    j = i + 1
                else:
                    break
                while j <= len(self.world_points):
                    # print(j)
                    x_2, y_2, z_2 = self.world_points[j]
                    world_distance_of_qr_codes.append(m.sqrt((m.pow((x_1 - x_2), 2) + m.pow((y_1 - y_2), 2))))
                    if j < len(self.world_points) - 1:
                        j = j + 1
                    else:
                        break
            #print(world_distance_of_qr_codes)

            sum = 0
            for x in world_distance_of_qr_codes:
                sum = sum + x
            average_world_distance_of_qr_codes = sum / len(world_distance_of_qr_codes)
            # print(average_world_distance_of_qr_codes)

            image_distance_of_qr_codes = []
            for i in range(len(self.image_points_of_qr_codes)):
                u_1 = self.image_points_of_qr_codes[i][0]
                v_1 = self.image_points_of_qr_codes[i][1]
                if i < len(self.world_points) - 1:
                    j = i + 1
                else:
                    break
                while j <= len(self.image_points_of_qr_codes):
                    u_2 = self.image_points_of_qr_codes[j][0]
                    v_2 = self.image_points_of_qr_codes[j][1]
                    image_distance_of_qr_codes.append(m.sqrt((m.pow((u_1 - u_2), 2) + m.pow((v_1 - v_2), 2))))
                    if j < len(self.world_points) - 1:
                        j = j + 1
                    else:
                        break
            # print(image_distance_of_qr_codes)

            sum = 0
            for x in image_distance_of_qr_codes:
                sum = sum + x
            average_image_distance_of_qr_codes = sum / len(image_distance_of_qr_codes)
            # print(average_image_distance_of_qr_codes)

            cm_per_pixel = average_world_distance_of_qr_codes / average_image_distance_of_qr_codes
            # print(cm_per_pixel)

            distance_balls_to_qr_code = []

            # print(len(self.ball_position))
            i = 0
            for (ball_x, ball_y) in self.ball_position:
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
            # print(distance_balls_to_qr_code)

            # print(self.world_localization_qr_codes)

            equations = []

            for j in range(len(self.world_localization_qr_codes)):
                x = self.world_localization_qr_codes[j][0]
                y = self.world_localization_qr_codes[j][1]
                r = (distance_balls_to_qr_code[0][j] * cm_per_pixel) ** 2
                equations.append([x, y, r])
            print(equations)

            x_y_coordinates = []
            for i in range(3):
                x_coordinates = []
                y_coordinates = []
                if i < 2:
                    a_1 = equations[i][0]
                    b_1 = equations[i][1]
                    r_1 = equations[i][2]
                    a_2 = equations[i + 1][0]
                    b_2 = equations[i + 1][1]
                    r_2 = equations[i + 1][2]
                else:
                    a_1 = equations[i][0]
                    b_1 = equations[i][1]
                    r_1 = equations[i][2]
                    a_2 = equations[0][0]
                    b_2 = equations[0][1]
                    r_2 = equations[0][2]

                a = a_1 - a_2

                c = (r_2 ** 2 - r_1 ** 2 + b_1 ** 2 - b_2 ** 2 - a ** 2) / (2 * a)
                d = -1 - (b_2 - b_1) ** 2 / a ** 2
                e = 2 * b_1 - 2 * (b_2 - b_1) * c / a
                f = r_1 ** 2 - b_1 ** 2 - c ** 2

                p = e / d
                q = f / d

                y_coordinates.append(-p / 2 + m.sqrt((p / 2) ** 2 - q))
                y_coordinates.append(-p / 2 - m.sqrt((p / 2) ** 2 - q))

                for y in y_coordinates:
                    x_coordinates.append(m.sqrt(r_1 ** 2 - y ** 2 + 2 * b_1 * y - b_1 ** 2) + a_1)
                    x_coordinates.append(-m.sqrt(r_1 ** 2 - y ** 2 + 2 * b_1 * y - b_1 ** 2) + a_1)

                for j in range(4):
                    if j < 2:
                        x_y_coordinates.append([x_coordinates[j], y_coordinates[0]])
                    else:
                        x_y_coordinates.append([x_coordinates[j], y_coordinates[1]])

            coordinates_found = False
            found_similar_x_coordinates = False
            found_similar_y_coordinates = False
            similar_y_coordinates = []
            similar_coordinates = []
            while not coordinates_found:
                place_of_similar_x_coordinates = [0]
                place_of_similar_y_coordinates = [0]
                n = 1
                # print(x_y_coordinates)
                while not found_similar_y_coordinates:
                    difference = x_y_coordinates[0][1] - x_y_coordinates[n][1]
                    if -1 < difference < 1:
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
                        difference = similar_y_coordinates[0][0] - similar_y_coordinates[n][0]
                        if -1 < difference < 1:
                            place_of_similar_x_coordinates.append(n)
                        if len(place_of_similar_x_coordinates) < 3:
                            if n < len(similar_y_coordinates) - 1:
                                n = n + 1
                            else:
                                removed = 0
                                for x in place_of_similar_x_coordinates:
                                    similar_y_coordinates.remove(similar_y_coordinates[x - removed])
                                    removed = removed + 1
                                n = 1
                                break
                        else:
                            for x in place_of_similar_x_coordinates:
                                similar_coordinates.append(similar_y_coordinates[x])
                            found_similar_x_coordinates = True
                            coordinates_found = True

            # print(similar_coordinates)

            sum_x_coordinates = 0
            sum_y_coordinates = 0
            for i in range(3):
                sum_x_coordinates = sum_x_coordinates + similar_coordinates[i][0]
                sum_y_coordinates = sum_y_coordinates + similar_coordinates[i][1]

            average_x_coordinate = sum_x_coordinates / 3
            average_y_coordinate = sum_y_coordinates / 3

            ball_coordinate = [average_x_coordinate, average_y_coordinate]

            print(ball_coordinate)
            

if __name__ == "__main__":
    camera = Camera()
    camera.show_image()
