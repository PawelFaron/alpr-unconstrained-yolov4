import cv2
from plate_finder import PlateFinder
from darknet_ocr import DarknetOCR

weights = 'ocr_models/yolov4_csp_sam/model.weights'
netcfg = 'ocr_models/yolov4_csp_sam/model.cfg'
dataset = 'ocr_models/yolov4_csp_sam/model.data'

class PlateReader:
    def __init__(self, ocr_config, ocr_data, ocr_weights, ocr_threshold=0.5, ocr_hier_thresh=0.5, ocr_nms=0.5, lp_finder_threshold=0.5, alphas=[0.5]):
        self.ocr = DarknetOCR(ocr_weights, ocr_config, ocr_data, ocr_threshold, ocr_hier_thresh, ocr_nms)
        self.plate_finder = PlateFinder(lp_finder_threshold, alphas)

    def read_license_plate(self, image):
        max_prob = 0
        best_plate = ""
        for lp_image in self.plate_finder.find(image):
            bboxes = self.ocr.predict(lp_image)
            prob = sum([v[-1] for v in bboxes])
            if prob > max_prob:
                best_plate = "".join([b[4] for b in bboxes])
                max_prob = prob
        return best_plate

    def read_license_plate_from_file(self, image_path):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.read_license_plate(img)