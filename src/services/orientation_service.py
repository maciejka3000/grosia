import numpy as np

from src.utils.config_loader import load_settings
from src.utils.runtime_loader import runtime_loader
from doctr.models import page_orientation_predictor
from src.utils.img_processing_utils import rotate_image, image_type_check, normalize_image


class OrientationService:
    def __init__(self):
        self.settings = load_settings()
        device = runtime_loader()
        self.orientation_predictor = page_orientation_predictor(pretrained=True).to(device)

    def get_angle(self, image):
        image = image_type_check(image)
        image = normalize_image(image, 0, 255, np.uint8)
        _, angle, conf = self.orientation_predictor([image])
        angle = angle[0]
        conf = conf[0]
        if angle == -90:
            angle = 270
        return angle, conf

    def get_angle_rotate_image(self, image):
        image = image_type_check(image)
        angle, _ =  self.get_angle(image)
        print('angle before rotation: ' + str(angle))
        return rotate_image(image, angle)



if __name__ == "__main__":
    from matplotlib import pyplot as plt
    service = OrientationService()
    plt.imshow((service.get_angle_rotate_image('/home/maciejka/Documents/projects/grosia_app/02_test_images/lidl-01.png')))
    plt.show()