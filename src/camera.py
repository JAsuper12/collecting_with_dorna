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

        # laden der intrinsischen Kalibriermatrix
        self.load_calibration_config()

        # Variabeln zur Farberkennung
        self.found_container = False
        self.contours_rectangle = []
        blue_lower = [50, 0, 0]
        blue_upper = [255, 75, 75]
        self.boundaries = [(blue_lower, blue_upper)]
        self.cX = None
        self.cY = None
        self.cX_container = None
        self.cY_container = None

        # Variablen zur Positionierung
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
        self.results = 0

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
            array = ueye.get_data(self.pcImageMemory, self.width, self.height, self.nBitsPerPixel, self.pitch, copy=False)
            # Bild zuschneiden
            frame = np.reshape(array, (self.height.value, self.width.value, self.bytes_per_pixel))
            frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            size = (self.height, self.width)
            # optimale Kameramatrix erstellen
            self.new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.dist_coeff, size, 1, size)
            # Bild entzerren
            dst = cv2.undistort(frame, self.camera_matrix, self.dist_coeff, None, self.new_camera_matrix)
            x, y, w, h = roi
            self.dst = dst[y:y + h, x:x + w]

            # Farberkennung durchführen
            self.detect_colors()

            # Kamera extrinsisch kalibrieren
            self.extrinsic_calibration()

            # Bild der Kamera auf dem Bildschirm ausgeben
            cv2.imshow("camera", self.dst)
            # Bild mit nur blauen Pixeln ausgeben
            cv2.imshow("blue_only", self.show_blue_color)

            # mit q Kamera schließen
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # mit t derzeitiges Bild aufnehemen
            elif cv2.waitKey(100) & 0xFF == ord('t'):
                cv2.imwrite("C:\dorna\camera\images\image.bmp", self.dst)
                cv2.imwrite("C:\dorna\camera\images\mask.bmp", self.show_blue_color)

            # mit l Behälterposition zurücksetzen
            elif cv2.waitKey(100) & 0xFF == ord('l'):
                self.found_container = False
                self.container_world_position.clear()

        # Kamera freigeben
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
        hsv = cv2.cvtColor(self.dst, cv2.COLOR_BGR2HSV)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        for (lower, upper) in self.boundaries:
            lower = np.array(lower)
            upper = np.array(upper)
            # Maske erstellen aus den definierten Farbbereich
            self.mask = cv2.inRange(bgr, lower, upper)
            # Bild mit nur blauen Pixeln erstellen
            self.show_blue_color = cv2.bitwise_and(bgr, bgr, mask=self.mask)
        self.draw_contours()

    def draw_contours(self):
        self.ball_position.clear()
        contours_area = []
        contours_circles = []
        contours, hierarchy = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2:]

        if not self.found_container:
            self.contours_rectangle.clear()

        # Bereiche auf Größe überprüfen
        for con in contours:
            area = cv2.contourArea(con)
            if 200 < area < 10000:
                contours_area.append(con)

        # Form der Bereiches ermitteln
        for con in contours_area:
            perimeter = cv2.arcLength(con, True)
            area = cv2.contourArea(con)
            approx = cv2.approxPolyDP(con, 0.01 * perimeter, True)

            circularity = 4 * m.pi * (area / (perimeter ** 2))

            if len(approx) == 4 and not self.found_container:
                # Bereich wurde als Rechteck erkannt und wird als Behälter identifiziert
                self.contours_rectangle.append(con)

            elif 0.8 < circularity:
                # Bereich wurde als Kreis erkannt und wird als Ball identifiziert
                contours_circles.append(con)

        for cnt in contours_circles:
            # Mittelpunkte der Bälle berechnen und Bildposition speichern
            M = cv2.moments(cnt)
            self.cX = int(M["m10"] / M["m00"])
            self.cY = int(M["m01"] / M["m00"]) + 5
            self.ball_position.append((self.cX, self.cY))
            # Kontur des Balles einzeichnen
            cv2.drawContours(self.dst, [cnt], 0, (0, 255, 0), 1)
            # Mittelpunkt einzeichnen
            cv2.circle(self.dst, (self.cX, self.cY), 2, (0, 255, 0), -1)

        for cnt in self.contours_rectangle:
            # Mittelpunkt des Behälters berechnen und Bildposition speichern
            M = cv2.moments(cnt)
            self.cX_container = int(M["m10"] / M["m00"])
            self.cY_container = int(M["m01"] / M["m00"])
            self.container_position.append((self.cX_container, self.cY_container))
            # Kontur des Behälters einzeichnen
            cv2.drawContours(self.dst, [cnt], 0, (0, 128, 255), 1)
            # Mittelpunkt einzeichnen
            cv2.circle(self.dst, (self.cX_container, self.cY_container), 1, (0, 128, 255), -1)

    def extrinsic_calibration(self):
        self.qr_centres.clear()
        self.world_points.clear()
        self.localization_qr_codes.clear()
        self.world_localization_qr_codes.clear()
        self.qr_codes = pyzbar.decode(self.dst)
        # alle erkannten QR-Codes durchgehen
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
            # Daten über den Codes einzeichnen
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
            _, self.rvecs, self.tvecs = cv2.solvePnP(world_points, self.image_points_of_qr_codes, self.camera_matrix, self.dist_coeff)
            # Bildkoordinaten der Koordinatenachsen berechnen
            self.origin, jacobian = cv2.projectPoints(np.array([(0.0, 0.0, 0.0)]), self.rvecs, self.tvecs, self.camera_matrix, self.dist_coeff)
            z_axis, jacobian = cv2.projectPoints(np.array([(0.0, 0.0, 55.0)]), self.rvecs, self.tvecs, self.camera_matrix, self.dist_coeff)
            x_axis, jacobian = cv2.projectPoints(np.array([(55.0, 0.0, 0.0)]), self.rvecs, self.tvecs, self.camera_matrix, self.dist_coeff)
            y_axis, jacobian = cv2.projectPoints(np.array([(0.0, 55.0, 0.0)]), self.rvecs, self.tvecs, self.camera_matrix, self.dist_coeff)
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
            # Positionsbestimmung der Bälle mit Vergleichsansatz
            self.get_ball_position_compare()

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
            self.world_points.append((-57.5, 55, 0))
            if len(self.localization_qr_codes) < 3:
                self.localization_qr_codes.append(centre)
                self.world_localization_qr_codes.append([-57.5, 55, 0])
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

    # GPS-Ansatz
    def get_ball_position_with_circles(self):
        # wenn mindestens drei QR-Codes erkannt wurden
        if len(self.localization_qr_codes) == 3:
            distance_balls_to_qr_code = []

            i = 0
            # für jeden Ball der erkannt wurde die Schleife durchlaufen
            for (ball_x, ball_y) in self.ball_position:
                distance = []
                n_distance = 0
                # Abstand von den drei Codes zum Ball bestimmen und als Kreise einzeichnen
                # für jeden Ball eine andere Farbe verwenden zur Übersichtlichkeit
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

            # wenn mindestens ein Ball erkannt wurde
            if len(self.ball_position) > 0:
                math_error = False
                world_distance_origin_to_qr_code = []
                # Abstand der QR-Codes zum Koordinatenursprung ermitteln
                for [x, y, z] in self.world_localization_qr_codes:
                    world_distance_origin_to_qr_code.append(m.sqrt(x ** 2 + y ** 2))

                image_distance_origin_to_qr_code = []
                # und den dazugehörigen Bildkoordinatenabstand
                for [u, v] in self.localization_qr_codes:
                    image_distance_origin_to_qr_code.append(m.sqrt((u - self.origin[0][0][0]) ** 2 +
                                                                   (v - self.origin[0][0][1]) ** 2))

                cm_per_pixel = []
                # durchschnittliche cm pro Pixel im Bild bestimmen
                for distance in range(3):
                    cm_per_pixel.append(world_distance_origin_to_qr_code[distance]
                                        / image_distance_origin_to_qr_code[distance])

                equations = []
                # x, y und r der drei Gleichungssysteme bestimmen
                for j in range(len(self.world_localization_qr_codes)):
                    x = self.world_localization_qr_codes[j][0]
                    y = self.world_localization_qr_codes[j][1]
                    r = (distance_balls_to_qr_code[0][j] * cm_per_pixel[j])
                    equations.append([x, y, r])

                x_y_coordinates = []
                # die drei Gleichungssysteme lösen
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
                    # y-Koordinaten berechnen
                    try:
                        y_coordinates.append(-p / 2 + m.sqrt((p / 2) ** 2 - q))
                        y_coordinates.append(-p / 2 - m.sqrt((p / 2) ** 2 - q))
                    except ValueError:
                        math_error = True
                        break
                    # dazugehörigen x-Koordinaten berechnen
                    for y in y_coordinates:
                        x_coordinates.append(m.sqrt(r_1 ** 2 - y ** 2 + 2 * b_1 * y - b_1 ** 2) + a_1)
                        x_coordinates.append(-m.sqrt(r_1 ** 2 - y ** 2 + 2 * b_1 * y - b_1 ** 2) + a_1)
                    # x- und y-Koordinaten zusammenfügen
                    for j in range(4):
                        if j < 2:
                            x_y_coordinates.append([x_coordinates[j], y_coordinates[0]])
                        else:
                            x_y_coordinates.append([x_coordinates[j], y_coordinates[1]])

                # Koordinaten miteinander vergleichen und die drei Koordinaten ermitteln, welche nahe beieinander liegen
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

                    # Schwerpunkt der drei Koordinaten berechnen
                    if not index_error:
                        sum_x_coordinates = 0
                        sum_y_coordinates = 0
                        for i in range(3):
                            sum_x_coordinates = sum_x_coordinates + similar_coordinates[i][0]
                            sum_y_coordinates = sum_y_coordinates + similar_coordinates[i][1]

                        average_x_coordinate = sum_x_coordinates / 3
                        average_y_coordinate = sum_y_coordinates / 3

                        ball_coordinate = [average_x_coordinate, average_y_coordinate]
                        # Ballkoordinate ausgeben
                        print(ball_coordinate)

    # Vergleichsansatz
    def get_ball_position_compare(self):
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
                    image_coordinate, jacobian = cv2.projectPoints(np.array([(x, y, 0.0)]), self.rvecs, self.tvecs,
                                                                   self.camera_matrix, self.dist_coeff)
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
                image_coordinate, jacobian = cv2.projectPoints(np.array([(y * self.increment - 60, y_coordinate_of_smallest_difference[y], 0.0)]),
                                                               self.rvecs, self.tvecs, self.camera_matrix, self.dist_coeff)
                # Abstand zwischen berechneten Bildkoordinate und Bildkoordinate des Balles berechnen
                x_difference.append([y * self.increment - 60, y_coordinate_of_smallest_difference[y],
                                     x_b - image_coordinate[0][0][0]])
                # Abstände miteinander vergleichen und größeren Abstand löschen
                if len(x_difference) == 2:
                    if 0 < x_difference[0][2] < x_difference[1][2] or x_difference[1][2] < x_difference[0][2] < 0:
                        x_difference.remove(x_difference[1])
                    else:
                        x_difference.remove(x_difference[0])

            ball_world_positions.append((x_difference[0][0], x_difference[0][1]))
        # prognostizierte Koordinate einzeichnen
        for balls in ball_world_positions:
            image_coordinate, jacobian = cv2.projectPoints(
                np.array([(x_difference[0][0], x_difference[0][1], 0.0)]), self.rvecs, self.tvecs,
                self.camera_matrix, self.dist_coeff)
            estimated_point = (int(image_coordinate[0][0][0]), int(image_coordinate[0][0][1]))
            cv2.circle(self.dst, estimated_point, 2, (0, 0, 255), -1)

        # selber Ablauf mit dem Behälter
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
        # Koordinaten ausgeben
        print("Positionen der Bälle: ", ball_world_positions, "\nPosition des Behälters: ", self.container_world_position)

    # Berechnen der Weltkoordinaten
    def get_ball_position_with_matrix(self):
        # Rotationsmatrix berechnen
        rmatrix = cv2.Rodrigues(self.rvecs)[0]
        # für jeden gefundenen Ball die Weltkoordinaten berechnen
        for ball in self.ball_position:
            u, v = self.ball_position[0]
            uv_1 = np.array([[u, v, 1]], dtype=np.float32)
            uv_1 = uv_1.T
            inverse_cam_mtx = np.linalg.inv(self.camera_matrix)
            inverse_R_mtx = np.linalg.inv(rmatrix)
            Rt = inverse_R_mtx.dot(self.tvecs)
            RM = inverse_R_mtx.dot(inverse_cam_mtx)
            RMuv = RM.dot(uv_1)
            s = (0 + Rt[2]) / RMuv[2]
            a = s*inverse_cam_mtx.dot(uv_1) - self.tvecs
            XYZ = inverse_R_mtx.dot(a)
            # Ballkoordinaten ausgeben
            print("Ballkoordinate: ", XYZ[0], XYZ[1])


if __name__ == "__main__":
    camera = Camera()
    camera.show_image()
