from pyueye import ueye
import cv2
import numpy as np
import json
import math as m
from pyzbar import pyzbar
import os
import sys


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
        self.path_images = os.path.dirname(os.path.abspath(sys.argv[0])) + "/images/"

        # laden der intrinsischen Kalibriermatrix
        self.load_calibration_config()

        # Variabeln zur Farberkennung
        blue_lower = [50, 0, 0]
        blue_upper = [255, 75, 75]
        red_lower = [0, 0, 100]
        red_upper = [50, 50, 255]
        green_lower = [0, 75, 50]
        green_upper = [50, 255, 150]
        self.blue_boundaries = [(blue_lower, blue_upper)]
        self.red_boundaries = [(red_lower, red_upper)]
        self.green_boundaries = [(green_lower, green_upper)]

        # Variablen zur Positionsbestimmung
        self.found_blue_container = False
        self.found_red_container = False
        self.found_green_container = False
        self.qr_centres = []
        self.world_points = []
        self.localization_qr_codes = []
        self.world_localization_qr_codes = []
        self.results = 0

        # Farbbereiche
        self.blue = Color(self.blue_boundaries, "blue")
        self.red = Color(self.red_boundaries, "red",)
        self.green = Color(self.green_boundaries, "green")

    def show_image(self):
        # Kamera initialisieren
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
            # Daten der Kamera auslesen
            array = ueye.get_data(self.pcImageMemory, self.width, self.height, self.nBitsPerPixel, self.pitch,
                                  copy=False)
            # Bild zuschneiden
            frame = np.reshape(array, (self.height.value, self.width.value, self.bytes_per_pixel))
            frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            size = (self.height, self.width)
            # optimale Kameramatrix erstellen
            self.new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.dist_coeff, size, 1,
                                                                        size)
            # Bild entzerren
            dst = cv2.undistort(frame, self.camera_matrix, self.dist_coeff, None, self.new_camera_matrix)
            x, y, w, h = roi
            self.dst = dst[y:y + h, x:x + w]
            # Farbmasken erstellen
            self.blue.create_color_mask(self.dst, self.found_blue_container)
            self.red.create_color_mask(self.dst, self.found_red_container)
            self.green.create_color_mask(self.dst, self.found_green_container)
            # Kamera extrinsisch kalibrieren
            self.extrinsic_calibration()
            # Bild auf dem Bildschirm ausgeben
            cv2.imshow("camera", self.dst)

            # mit q Kamera schließen
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # mit t Bild aufnehemen
            elif cv2.waitKey(100) & 0xFF == ord('t'):
                cv2.imwrite(self.path_images + "image.bmp", self.dst)
                print("Bild aufgenommen")
            # mit l Position der Behälter zurücksetzen
            elif cv2.waitKey(100) & 0xFF == ord('l'):
                self.found_blue_container = False
                self.found_red_container = False
                self.found_green_container = False
                print("Behälterposition zurückgesetzt")

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

    def extrinsic_calibration(self):
        self.qr_centres.clear()
        self.world_points.clear()
        self.localization_qr_codes.clear()
        self.world_localization_qr_codes.clear()
        self.qr_codes = pyzbar.decode(self.dst)
        # Schleife für jeden erkannten QR-Code ausführen
        for qr in self.qr_codes:
            # Position des Codes ermitteln
            (x, y, w, h) = qr.rect
            # Mittelpunkt des Codes berechnen und als Bildkoordinate speichern
            centre = (x + int((w / 2)), y + int((h / 2)))
            self.qr_centres.append(centre)
            # erkannten Code umrahmen
            cv2.rectangle(self.dst, (x, y), (x + w, y + h), (0, 0, 255), 2)
            # Mittelpunkt einzeichnen
            cv2.circle(self.dst, centre, 2, (0, 255, 0), -1)
            # Daten aus dem Codes als String speichern
            data = qr.data.decode("utf-8")
            # Weltkoordinate des Codes speichern
            self.qr_decoder(data, centre)
            # Daten über den Code einzeichnen
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

        # wenn genügend Codes erkannt wurden und diese gut verteilt sind, soll die Kamera extrinsisch kalibriert werden
        if len(self.qr_centres) >= 6 and valid_qr_spread:
            self.image_points_of_qr_codes = np.array(self.qr_centres, dtype="float")
            # Kamera extrinsisch Kalibrieren
            _, self.rvecs, self.tvecs = cv2.solvePnP(world_points, self.image_points_of_qr_codes, self.camera_matrix,
                                                     self.dist_coeff)
            # Bildkoordinaten der Koordinatenachsen berechnen
            self.origin, jacobian = cv2.projectPoints(np.array([(0.0, 0.0, 0.0)]), self.rvecs, self.tvecs,
                                                      self.camera_matrix, self.dist_coeff)
            z_axis, jacobian = cv2.projectPoints(np.array([(0.0, 0.0, 55.0)]), self.rvecs, self.tvecs,
                                                 self.camera_matrix, self.dist_coeff)
            x_axis, jacobian = cv2.projectPoints(np.array([(55.0, 0.0, 0.0)]), self.rvecs, self.tvecs,
                                                 self.camera_matrix, self.dist_coeff)
            y_axis, jacobian = cv2.projectPoints(np.array([(0.0, 55.0, 0.0)]), self.rvecs, self.tvecs,
                                                 self.camera_matrix, self.dist_coeff)
            axis = [x_axis, y_axis, z_axis]
            # Koordinatenachsen einzeichnen
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
            # Positionsbestimmung der Bälle und Behälter mit Vergleichsansatz
            self.found_blue_container = self.blue.get_ball_position_compare(self.rvecs, self.tvecs, self.camera_matrix, self.dist_coeff)
            self.found_red_container = self.red.get_ball_position_compare(self.rvecs, self.tvecs, self.camera_matrix, self.dist_coeff)
            self.found_green_container = self.green.get_ball_position_compare(self.rvecs, self.tvecs, self.camera_matrix, self.dist_coeff)

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


class Color:
    def __init__(self, boundaries, color):
        self.boundaries = boundaries
        self.color = color
        self.container_world_position = []
        self.ball_position = []
        self.container_position = []
        self.contours_rectangle = []
        self.increment = 0.25

    def create_color_mask(self, dst, found_container):
        self.dst = dst
        self.found_container = found_container
        hsv = cv2.cvtColor(self.dst, cv2.COLOR_BGR2HSV)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        for (lower, upper) in self.boundaries:
            lower = np.array(lower)
            upper = np.array(upper)
            # Maske aus den definierten Farbbereich erstellen
            self.mask = cv2.inRange(bgr, lower, upper)
            # Bild mit nur Pixeln der jeweiligen Farbe erstellen
            self.mask_image = cv2.bitwise_and(bgr, bgr, mask=self.mask)
        self.draw_contours()

    def draw_contours(self):
        self.ball_position.clear()
        contours_area = []
        contours_circles = []
        contours, hierarchy = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2:]

        if not self.found_container:
            self.contours_rectangle.clear()
            self.container_world_position.clear()
            self.container_position.clear()
        # Bereiche auf Größe überprüfen
        for con in contours:
            area = cv2.contourArea(con)
            if 200 < area < 5000:
                contours_area.append(con)
        # Form des Bereiches ermitteln
        for con in contours_area:
            perimeter = cv2.arcLength(con, True)
            area = cv2.contourArea(con)
            approx = cv2.approxPolyDP(con, 0.01 * perimeter, True)
            circularity = 4 * m.pi * (area / (perimeter ** 2))
            if len(approx) == 4 and not self.found_container:
                # Bereich wurde als Rechteck erkannt und wird daher als Behälter identifiziert
                self.contours_rectangle.append(con)
            elif 0.8 < circularity:
                # Bereich wurde als Kreis erkannt und wird daher als Ball identifiziert
                contours_circles.append(con)

        for cnt in contours_circles:
            # Mittelpunkte der Bälle berechnen und Bildposition speichern
            M = cv2.moments(cnt)
            self.cX = int(M["m10"] / M["m00"])
            self.cY = int(M["m01"] / M["m00"]) + 5
            self.ball_position.append((self.cX, self.cY))
            # Kontur und Mittelpunkte der Bälle einzeichnen
            if self.color == "blue":
                cv2.drawContours(self.dst, [cnt], 0, (0, 128, 255), 1)
                cv2.circle(self.dst, (self.cX, self.cY), 2, (0, 128, 255), -1)
            elif self.color == "red":
                cv2.drawContours(self.dst, [cnt], 0, (0, 255, 255), 1)
                cv2.circle(self.dst, (self.cX, self.cY), 2, (0, 255, 255), -1)
            elif self.color == "green":
                cv2.drawContours(self.dst, [cnt], 0, (255, 0, 255), 1)
                cv2.circle(self.dst, (self.cX, self.cY), 2, (255, 0, 255), -1)

        for cnt in self.contours_rectangle:
            # Mittelpunkt der Behälter berechnen und Bildposition speichern
            M = cv2.moments(cnt)
            self.cX_container = int(M["m10"] / M["m00"])
            self.cY_container = int(M["m01"] / M["m00"])
            self.container_position.append((self.cX_container, self.cY_container))
            # Kontur und Mittelpunkte der Behälter einzeichnen
            if self.color == "blue":
                cv2.drawContours(self.dst, [cnt], 0, (0, 128, 255), 1)
                cv2.circle(self.dst, (self.cX_container, self.cY_container), 1, (0, 128, 255), -1)
            elif self.color == "red":
                cv2.drawContours(self.dst, [cnt], 0, (0, 255, 255), 1)
                cv2.circle(self.dst, (self.cX_container, self.cY_container), 1, (0, 255, 255), -1)
            elif self.color == "green":
                cv2.drawContours(self.dst, [cnt], 0, (255, 0, 255), 1)
                cv2.circle(self.dst, (self.cX_container, self.cY_container), 1, (255, 0, 255), -1)

    # Vergleichsansatz
    def get_ball_position_compare(self, rvecs, tvecs, camera_matrix, dist_coeff):
        ball_world_positions = []
        # für jeden gefundenen Ball Schleife einmal durchlaufen
        for ball in self.ball_position:
            y_coordinate_of_smallest_difference = []
            x_b, y_b = ball
            x = -60
            x_difference = []
            # jede x-Koordinate des Koordinatensystems durchgehen
            while x <= 60:
                y = -10
                y_difference = []
                # jede y-Koordinate des Koordinatensystems einmal durchgehen und Bildkoordinaten berechnen
                while y <= 60:
                    image_coordinate, jacobian = cv2.projectPoints(np.array([(x, y, 0.0)]), rvecs, tvecs,
                                                                   camera_matrix, dist_coeff)
                    # Abstand zwischen berechneten Bildkoordinate und Bildkoordinate des Balles berechnen
                    y_difference.append([y, y_b - image_coordinate[0][0][1]])
                    # Abstände miteinander vergleichen und größeren Abstand löschen
                    if len(y_difference) == 2:
                        if 0 < y_difference[0][1] < y_difference[1][1] or y_difference[1][1] < y_difference[0][1] < 0:
                            y_difference.remove(y_difference[1])
                        else:
                            y_difference.remove(y_difference[0])
                    y = y + self.increment
                y_coordinate_of_smallest_difference.append(y_difference[0][0])
                x = x + self.increment

            for y in range(len(y_coordinate_of_smallest_difference)):
                # Bildkoordinate jeder x-Koordinate auf Höhe der y-Koordinate mit den kleinsten Abstand zum Ball
                # berechnen
                image_coordinate, jacobian = cv2.projectPoints(
                    np.array([(y * self.increment - 60, y_coordinate_of_smallest_difference[y], 0.0)]), rvecs,
                    tvecs, camera_matrix, dist_coeff)
                # Abstand zwischen berechneten Bildkoordinate und Bildkoordinate des Balles berechnen
                x_difference.append([y * self.increment - 60, y_coordinate_of_smallest_difference[y],
                                     x_b - image_coordinate[0][0][0]])
                # Abstände miteinander vergleichen und größeren Abstand löschen
                if len(x_difference) == 2:
                    if 0 < x_difference[0][2] < x_difference[1][2] or x_difference[1][2] < x_difference[0][2] < 0:
                        x_difference.remove(x_difference[1])
                    else:
                        x_difference.remove(x_difference[0])
            # Weltkoordinate in dem sich ein Ball befindet
            ball_world_positions.append((x_difference[0][0], x_difference[0][1]))
        # prognostizierte Koordinate einzeichnen
        for balls in ball_world_positions:
            image_coordinate, jacobian = cv2.projectPoints(
                np.array([(x_difference[0][0], x_difference[0][1], 0.0)]), rvecs, tvecs,
                camera_matrix, dist_coeff)
            estimated_point = (int(image_coordinate[0][0][0]), int(image_coordinate[0][0][1]))
            cv2.circle(self.dst, estimated_point, 2, (0, 0, 255), -1)

        # selber Ablauf mit dem Behälter nur wenn die Koordinate des Behälters noch nicht ermittelt wurde
        if not self.found_container and len(self.container_position) > 0:
            y_coordinate_of_smallest_difference = []
            x_b, y_b = self.container_position[0]
            x = -60
            x_difference = []
            while x <= 60:
                y = -10
                y_difference = []
                while y <= 60:
                    image_coordinate, jacobian = cv2.projectPoints(np.array([(x, y, 0.0)]), rvecs, tvecs,
                                                                   camera_matrix, dist_coeff)
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
                    np.array([(y * self.increment - 60, y_coordinate_of_smallest_difference[y], 0.0)]), rvecs,
                    tvecs,
                    camera_matrix, dist_coeff)
                x_difference.append(
                    [y * self.increment - 60, y_coordinate_of_smallest_difference[y], x_b - image_coordinate[0][0][0]])

                if len(x_difference) == 2:
                    if 0 < x_difference[0][2] < x_difference[1][2] or x_difference[1][2] < x_difference[0][2] < 0:
                        x_difference.remove(x_difference[1])
                    else:
                        x_difference.remove(x_difference[0])

            self.container_world_position.append((x_difference[0][0], x_difference[0][1]))
            self.found_container = True
        # Koordinaten ausgeben
        if self.color == "blue":
            print("Positionen der blauen Bälle: ", ball_world_positions, "\nPosition des blauen Behälters: ",
                  self.container_world_position)
        elif self.color == "red":
            print("Positionen der roten Bälle: ", ball_world_positions, "\nPosition des roten Behälters: ",
                  self.container_world_position)
        elif self.color == "green":
            print("Positionen der grünen Bälle: ", ball_world_positions, "\nPosition des grünen Behälters: ",
                  self.container_world_position)
        return self.found_container


if __name__ == "__main__":
    camera = Camera()
    camera.show_image()
