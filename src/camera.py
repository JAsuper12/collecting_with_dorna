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

            '''for j in range(len(self.world_localization_qr_codes)):
                x = 1
                x_k = self.world_localization_qr_codes[j][0]
                y = 1
                y_k = self.world_localization_qr_codes[j][1]
                k = (distance_balls_to_qr_code[0][j] * cm_per_pixel) ** 2
                equations.append([x, x_k, y, y_k, k])
            print(equations)'''

            a_1 = 30
            b_1 = 30
            k_1 = 22.32
            a_2 = 0
            b_2 = 40
            k_2 = 19.41

            a = a_1 - a_2
            print(a)
            k = k_2 - k_1
            print(k)
            b = b_1 - b_2
            print(b ** 2)
            c = (k ** 2 + b ** 2 - a ** 2) / (2 * a)
            print(c)
            d = -1 - (b / a) ** 2
            print(d)
            e = 2 * b_1 + 2 * b * c / a
            print(e)
            f = c ** 2 + b_1 ** 2 - k_1 ** 2
            print(f)

            p = e / d
            print(p)
            q = -f / d
            print(q)

            y_1 = -p / 2 + m.sqrt((p / 2) ** 2 - q)
            y_2 = -p / 2 - m.sqrt((p / 2) ** 2 - q)
            print(y_1, y_2)


if __name__ == "__main__":
    camera = Camera()
    camera.show_image()
