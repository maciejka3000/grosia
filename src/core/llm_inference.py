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

ocr = extract_service.ExtractService()
llm = llm_extraction_service.LLMExtractionService()
db = database_service.DatabaseService()

images = glob.glob('/home/maciejka/Documents/projects/grosia_app/04_preprocessed_images/*.*')

image_str = images[0]
for image_str in images:
    img = cv2.imread(image_str)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    scan = ocr.extract(img)
    print(scan)
    out = llm.extract_and_categorize(scan, return_as_dict=True)
    print(out)
    out_metadata = llm.extract_date_shop_name(scan)
    print(out_metadata)

    db.add_new_receipt(out, out_metadata, image_str)
