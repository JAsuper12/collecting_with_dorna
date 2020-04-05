from pyueye import ueye
import numpy as np
import cv2
import json
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

        self.size_of_square = 3.6  # Zentimeter
        self.nx = 8
        self.ny = 6
        self.n_im = 0
        self.gray = None
        self.objectpoints = None
        self.imagepoints = None
        self.camera_matrix = None
        self.dist_coeff = None
        self.mean_error = 0

        # Pfad zum Speichern der Bilder angeben
        self.path = os.path.dirname(os.path.abspath(sys.argv[0])) + "/images/"

    def show_image(self):
        # Kamera initialisieren
        nRet = ueye.is_InitCamera(self.h_cam, None)
        nRet = ueye.is_SetDisplayMode(self.h_cam, ueye.IS_SET_DM_DIB)
        nRet = ueye.is_AOI(self.h_cam, ueye.IS_AOI_IMAGE_GET_AOI, self.rectAOI, ueye.sizeof(self.rectAOI))
        self.width = self.rectAOI.s32Width
        self.height = self.rectAOI.s32Height
        nRet = ueye.is_AllocImageMem(self.h_cam, self.width, self.height,
                                     self.nBitsPerPixel, self.pcImageMemory, self.MemID)
        nRet = ueye.is_SetImageMem(self.h_cam, self.pcImageMemory, self.MemID)
        nRet = ueye.is_SetColorMode(self.h_cam, self.ColorMode)
        nRet = ueye.is_CaptureVideo(self.h_cam, ueye.IS_DONT_WAIT)
        nRet = ueye.is_InquireImageMem(self.h_cam, self.pcImageMemory, self.MemID, self.width,
                                       self.height, self.nBitsPerPixel, self.pitch)

        while nRet == ueye.IS_SUCCESS:
            # Daten der Kamera auslesen
            array = ueye.get_data(self.pcImageMemory, self.width, self.height,
                                  self.nBitsPerPixel, self.pitch, copy=True)
            # Bild zuschneiden
            frame = np.reshape(array, (self.height.value, self.width.value, self.bytes_per_pixel))
            frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            # Kamerabild ausgeben
            cv2.imshow("camera", frame)
            # mit q Kamera schließen
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # mit t Bild aufnehmen und speichern, wenn ein Schachbrettmuster erkannt wird
            elif cv2.waitKey(1) & 0xFF == ord('t'):
                ret, corners = cv2.findChessboardCorners(frame, (self.nx, self.ny), None)
                if ret:
                    im_name = self.path + str(self.n_im) + ".bmp"
                    cv2.imwrite(im_name, frame)
                    self.n_im = self.n_im + 1
                    print("Bild", self.n_im, "aufgenommen.")
                else:
                    print("Kein Chessboard gefunden")

            # mit d Schachbrettlinien zeichnen und speichern
            elif cv2.waitKey(1) & 0xFF == ord('d'):
                n_img = eval(input("Wie viele Bilder sollen bearbeitet werden? "))
                self.draw_chessboard_corners(n_img, True)

            # mit c Kamera intrinsisch kalibrieren
            elif cv2.waitKey(1) & 0xFF == ord('c'):
                self.calibrate()

        ueye.is_FreeImageMem(self.h_cam, self.pcImageMemory, self.MemID)
        ueye.is_ExitCamera(self.h_cam)
        cv2.destroyAllWindows()

    def draw_chessboard_corners(self, n_img, save):
        print("Schabrettlinien werden erstellt.")
        criteria = (cv2.TermCriteria_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # Weltkoordinaten der Referenzpunkte des Schachbrettmusters speichern
        objectp = np.zeros((self.ny * self.nx, 3), np.float32)
        objectp[:, :2] = (self.size_of_square * np.mgrid[0:8, 0:6]).T.reshape(-1, 2)

        self.objectpoints = []
        self.imagepoints = []
        images = []

        for x in range(n_img):
            print("Bild Nummer", x, "wird erstellt")
            images.append(self.path + str(x) + ".bmp")
        x = 0
        for fname in images:
            # aktuelles Bild speichern
            img = cv2.imread(fname)
            self.gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Schachbrettmuster im Bild suchen
            ret, corners = cv2.findChessboardCorners(self.gray, (self.nx, self.ny), None)
            if ret:
                self.objectpoints.append(objectp)
                corners2 = cv2.cornerSubPix(self.gray, corners, (11, 11), (-1, -1), criteria)
                # Bildkoordinaten der Referenzpunkte des Schachbrettmusters speichern
                self.imagepoints.append(corners2)
                # Bild mit eingezeichneten Schachbrettmuster speichern
                if save:
                    img = cv2.drawChessboardCorners(img, (self.nx, self.ny), corners2, ret)
                    cb_img = self.path + str(x) + ".bmp"
                    cv2.imwrite(cb_img, img)
                    print("Bild gespeichert")
                    x = x + 1
        print("Schachbrettlinien erstellt.")

    def calibrate(self):
        self.draw_chessboard_corners(50, False)

        # Kamera intrinsisch kalibrieren
        print("Kamera wird kalibriert.")
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objectpoints, self.imagepoints,
                                                           self.gray.shape[::-1], None, None)
        self.camera_matrix = mtx
        self.dist_coeff = dist
        print(self.camera_matrix)
        print(self.dist_coeff)

        # Kameramatrix optimieren
        img = cv2.imread(self.path + "0.bmp")
        h, w = img.shape[:2]
        self.newcameramatrix, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

        # Bild entverzerren
        dst = cv2.undistort(img, mtx, dist, None, self.newcameramatrix)

        # Bild zuschneiden
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        cv2.imwrite(self.path + "calibrated_result.bmp", dst)
        print("Kamera kalibriert.")

        # Ergebnis überprüfen, sollte möglichst 0 sein
        for x in range(len(self.objectpoints)):
            imgpoints2, _ = cv2.projectPoints(self.objectpoints[x], rvecs[x], tvecs[x], mtx, dist)
            error = cv2.norm(self.imagepoints[x], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            self.mean_error = self.mean_error + error

        print("Fehlerquote: ", self.mean_error / len(self.objectpoints))
        # Kameramatrix und Verzerrungskoeffizienten speichern
        self.save_calibration_config()

    def save_calibration_config(self):
        data = {
            "camera_matrix": self.camera_matrix.tolist(),
            "dist_coeff": self.dist_coeff.tolist(),
            "mean_error": self.mean_error
        }
        path_config = os.path.dirname(os.path.abspath(sys.argv[0])) + "config/camera_calibration_config.json"
        with open(path_config, "w") as file:
            json.dump(data, file)


if __name__ == "__main__":
    camera = Camera()
    camera.show_image()
