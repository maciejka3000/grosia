from doctr.models import ocr_predictor
from src.utils.config_loader import load_settings
from src.utils.runtime_loader import runtime_loader
from src.utils.img_processing_utils import image_type_check




class ExtractService:

    def __init__(self):
        settings = load_settings()
        self.ocr_settings = settings["ocr_settings"]
        self.verbose = settings["ocr_verbose"]
        device = runtime_loader()
        self.predictor = ocr_predictor(**self.ocr_settings).to(device)

    def text_postprocessing(self, out):
        out_full_string = ""
        for n, line in enumerate(out.pages[0].blocks[0].lines):
            out_str = ""
            for word in line.words:
                out_str += word.value + " "
            out_full_string += out_str + '\n'

        return out_full_string

    def extract(self, image, return_raw=False):
        img = image_type_check(image)
        predicted_text = self.predictor([img])
        if return_raw:
            return predicted_text
        else:
            return self.text_postprocessing(predicted_text)

if __name__ == '__main__':
    service = ExtractService()
    im = '/home/maciejka/Documents/projects/grosia_app/02_test_images/lidl-00.png'
    result = service.extract(im)
    print(result)



