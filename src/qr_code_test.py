import numpy as np
import cv2
import json
from pyueye import ueye
from pyzbar import pyzbar


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

        self.load_calibration_config()

        self.object_point = np.array([(0, 45, 0), (5.25, 39.75, 0), (5.25, 50.25, 0), (-5.25, 39.75, 0), (-5.25, 50.25, 0)])
        self.axis = np.float32([[10, 0, 0], [0, 10, 0], [0, 0, -10]]).reshape(-1, 3)

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
            array = ueye.get_data(self.pcImageMemory, self.width, self.height, self.nBitsPerPixel, self.pitch,
                                  copy=False)
            frame = np.reshape(array, (self.height.value, self.width.value, self.bytes_per_pixel))
            frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            size = (self.height, self.width)
            new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.dist_coeff, size, 1, size)
            dst = cv2.undistort(frame, self.camera_matrix, self.dist_coeff, None, new_camera_matrix)
            x, y, w, h = roi
            self.dst = dst[y:y + h, x:x + w]
            self.qr_decoder()
            # cv2.imshow("orange", self.output)
            cv2.imshow("camera", self.dst)

            # Kamera schließen
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            elif cv2.waitKey(1) & 0xFF == ord('t'):
                cv2.imwrite("/home/lennart/dorna/camera/images/contour_only_circles.bmp", self.dst)
                # cv2.imwrite("/home/lennart/dorna/camera/images/only_orange.bmp", self.output)

        ueye.is_FreeImageMem(self.h_cam, self.pcImageMemory, self.MemID)
        ueye.is_ExitCamera(self.h_cam)
        cv2.destroyAllWindows()

    def load_calibration_config(self):
        with open("/home/lennart/dorna/camera/camera_calibration_config.json", "r") as file:
            data = json.load(file)
            self.camera_matrix = np.array(data["camera_matrix"])
            self.dist_coeff = np.array(data["dist_coeff"])
            self.mean_error = data["mean_error"]

    '''def display_code(self, bbox):
        n = len(bbox)
        for j in range(n):
            cv2.line(self.dst, tuple(bbox[j][0]), tuple(bbox[(j + 1) % n][0]), (255, 0, 0), 3)

    def qr_decoder(self):
        qrDecoder = cv2.QRCodeDetector()
        data, bbox, rectifiedImage = qrDecoder.detectAndDecode(self.dst)
        if len(data) > 0:
            print("Decoded Data : {}".format(data))
            self.display_code(bbox)
            rectifiedImage = np.uint8(rectifiedImage)
        else:
            print("QR Code not detected")'''

    def qr_decoder(self):
        centres = []
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
            barcodeData = qr.data.decode("utf-8")

            # draw the barcode data and barcode type on the image
            text = "{}".format(barcodeData)
            cv2.putText(self.dst, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            print(centres)

        if len(centres) > 4:
            image_points = np.array(centres, dtype="float")
            print(image_points)

            _, rvecs, tvecs, inliers = cv2.solvePnPRansac(self.object_point, image_points, self.camera_matrix, self.dist_coeff)

            imgpts, jacobian = cv2.projectPoints(self.axis, rvecs, tvecs, self.camera_matrix, self.dist_coeff)

            p1 = (int(image_points[0][0]), int(image_points[0][1]))
            p2 = (int(imgpts[0][0][0]), int(imgpts[0][0][1]))
            self.dst = cv2.line(self.dst, p1, p2, (255, 0, 0), 5)


if __name__ == "__main__":
    camera = DetectColors()
    camera.show_image()
