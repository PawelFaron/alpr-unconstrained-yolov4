import os
os.environ['DARKNET_PATH'] = 'darknet'

from plate_reader import PlateReader


weights = 'ocr_models/yolov4_csp_sam/model.weights'
config = 'ocr_models/yolov4_csp_sam/model.cfg'
data = 'ocr_models/yolov4_csp_sam/model.data'

ocr_threshold=0.5
ocr_hier_thresh=0.5
ocr_nms=0.5
lp_finder_threshold=0.5
alphas=[0.5, 0.1, 0.4]

plate_reader = PlateReader(config, data, weights, ocr_threshold, ocr_hier_thresh, ocr_nms, lp_finder_threshold, alphas)

plate = plate_reader.read_license_plate_from_file('data/example.jpg')
print(plate)