import cv2
import numpy as np


def rotate_image(image, angle):
    if angle == 0: return image
    elif angle == 90: return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif angle == 180: return cv2.rotate(image, cv2.ROTATE_180)
    elif angle == 270: return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    else: raise ValueError("Angle must be 0, 90, 180, 270")

def image_type_check(image):
    if type(image) is str:
        img = cv2.imread(image, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    if type(image) is np.ndarray:
        lenimage = len(image.shape)
        if lenimage == 3:
            return image
        elif lenimage == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    raise ValueError("Image type incorrect")

def normalize_image(image:np.ndarray, min_val=0, max_val=255, out_dtype=np.uint8):
    norm_min, norm_max = np.min(image), np.max(image)
    n_img = (image - norm_min) / (norm_max - norm_min) * max_val
    n_img = np.clip(n_img, min_val, max_val)
    n_img = n_img.astype(out_dtype)
    return n_img


if __name__ == "__main__":
    image_type_check(cv2.imread("/home/maciejka/Documents/projects/grosia_app/02_test_images/IMG_5422.HEIC.jpg", cv2.IMREAD_GRAYSCALE))