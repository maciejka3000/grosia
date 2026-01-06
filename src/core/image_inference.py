from src.services import (
database_service,
extract_service,
image_preprocessing_service,
llm_extraction_service,
orientation_service,
segmenting_service,
straightening_service,
)
import numpy as np
import glob
import os
import cv2
from matplotlib import pyplot as plt

segmentator = segmenting_service.Segmentator()
straightener = straightening_service.StraighteningService()
rotator = orientation_service.OrientationService()

in_path = '/home/maciejka/Documents/projects/grosia_app/03_final_test_images'
out_path = '/home/maciejka/Documents/projects/grosia_app/04_preprocessed_images'

images_paths = glob.glob(os.path.join(in_path, '*.JPEG'))
verbose = False
for n, image_path in enumerate(images_paths):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    segment = segmentator.segment(img, verbose=False)
    straight_img = straightener.straighten_simple(img, segment)
    out_image = rotator.get_angle_rotate_image(straight_img)
    out_name = f"{n}.jpg"
    print(out_name)
    out_image_path = os.path.join(out_path, out_name)
    cv2.imwrite(out_image_path, out_image)

    if verbose:
        plt.figure(figsize=(5, 10))
        plt.imshow(out_image)
        plt.axis('off')
        plt.tight_layout()
        plt.show()



