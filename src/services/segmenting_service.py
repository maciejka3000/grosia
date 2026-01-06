from typing import Union


from src.utils.config_loader import load_settings
from ultralytics import YOLO, SAM
import cv2
import numpy as np
from src.utils import img_processing_utils

class Segmentator:
    def __init__(self):
        self.settings = load_settings()
        self.segmentation_type = self.settings['segmentation_type']
        self.model_path = self.settings["segmentation_model_path"]
        self.segment_settings = self.settings["segmentation_settings"]

        self.model = None
        self.init_models()

    def init_models(self):
        if self.segmentation_type == "SAM":
            self.model = SAM(self.model_path)
        elif self.segmentation_type == "YOLO":
            self.model = YOLO(self.model_path)
        else:
            raise KeyError('settings.yaml/segmentation_type: must be "YOLO" or "SAM"')

    def segment(self, image: Union[np.ndarray, str], verbose: bool = False):
        if self.segmentation_type == "YOLO":
            return self._segment_yolo(image)
        elif self.segmentation_type == "SAM":
            return self._segment_sam(image, verbose)
        else:
            raise KeyError('settings.yaml/segmentation_type: must be "YOLO" or "SAM"')

    def _segment_sam(self, image, verbose: bool = False):
        image = img_processing_utils.image_type_check(image)

        imsz = image.shape[:2]
        result = self.model(image, points = [imsz[1] // 2, imsz[0] // 2], **self.segment_settings)
        if result[0].masks is None:
            print('No mask found')
            return None
        xy_pre = result[0].masks.xy[0]
        mask = np.zeros((imsz[0], imsz[1]), dtype=np.uint8)
        mask = cv2.fillPoly(mask, [xy_pre.astype(np.int32)], 1)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.erode(mask, kernel)
        mask = cv2.dilate(mask, kernel)

        cnt, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_cnt = max(cnt, key=cv2.contourArea)

        xy_polygon = largest_cnt.reshape(-1, 2).astype(np.float32)

        if verbose:
            from matplotlib import pyplot as plt
            plt.figure()
            plt.imshow(image)
            plt.plot(xy_polygon[:, 0], xy_polygon[:, 1], 'r-')
            plt.show()
        return [xy_polygon]



    def _segment_yolo(self, image):
        result = self.model(image, **self.segment_settings)
        if result[0].masks is None:
            print('No mask found')
            return None
        return result[0].masks.xy

if __name__ == "__main__":
    segmentator = Segmentator()
    print(segmentator.model_path)
    print(segmentator.segment('/home/maciejka/Documents/projects/grosia_app/02_test_images/1.jpg'))