import numpy as np
import cv2
import json
from pyueye import ueye


class DetectColors:
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

        self.nx = 8
        self.ny = 6
        self.size_of_square = 3.6

        self.load_calibration_config()

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
            self.draw()
            #cv2.imshow("orange", self.output)
            cv2.imshow("camera", self.dst)

            # Kamera schließen
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            elif cv2.waitKey(1) & 0xFF == ord('t'):
                cv2.imwrite("/home/lennart/dorna/camera/images/contour_only_circles.bmp", self.dst)
                #cv2.imwrite("/home/lennart/dorna/camera/images/only_orange.bmp", self.output)

        ueye.is_FreeImageMem(self.h_cam, self.pcImageMemory, self.MemID)
        ueye.is_ExitCamera(self.h_cam)
        cv2.destroyAllWindows()

    def load_calibration_config(self):
        with open("/home/lennart/dorna/camera/camera_calibration_config.json", "r") as file:
            data = json.load(file)
            self.camera_matrix = np.array(data["camera_matrix"])
            self.dist_coeff = np.array(data["dist_coeff"])
            self.mean_error = data["mean_error"]

    def draw(self):
        criteria = (cv2.TermCriteria_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objectp = np.zeros((self.ny * self.nx, 3), np.float32)
        objectp[:, :2] = (self.size_of_square * np.mgrid[0:8, 0:6]).T.reshape(-1, 2)

        axis = np.float32([[10, 0, 0], [0, 10, 0], [0, 0, -10]]).reshape(-1, 3)

        gray = cv2.cvtColor(self.dst, cv2.COLOR_BGR2GRAY)
        # Find the rotation and translation vectors.
        ret, corners = cv2.findChessboardCorners(gray, (self.nx, self.ny), None)

        if ret:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            _, rvecs, tvecs, inliers = cv2.solvePnPRansac(objectp, corners2, self.camera_matrix, self.dist_coeff)

            # project 3D points to image plane
            imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, self.camera_matrix, self.dist_coeff)

            corner = tuple(corners2[0].ravel())
            self.dst = cv2.line(self.dst, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 5)
            self.dst = cv2.line(self.dst, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
            self.dst = cv2.line(self.dst, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 5)


if __name__ == "__main__":
    camera = DetectColors()
    camera.show_image()