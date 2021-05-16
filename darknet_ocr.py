from ocr_improver import OCRImprover
from darknet.darknet import pointer, IMAGE, POINTER, c_float, c_int, predict_image, \
                        get_network_boxes, do_nms_sort, free_detections, load_net, load_meta
import numpy as np
import cv2

class DarknetOCR:
    def __init__(self, weights, netcfg, dataset, ocr_threshold=0.5, hier_thresh=0.5, nms=0.5):
        self.ocr_threshold = ocr_threshold
        self.hier_thresh = hier_thresh
        self.nms = nms
        self.ocr_improver = OCRImprover()

        self.net = load_net(netcfg.encode('utf-8'), weights.encode('utf-8'), 0)
        self.meta = load_meta(dataset.encode('utf-8'))
        self.class_names = [self.meta.names[i].decode("ascii") for i in range(self.meta.classes)]

    @staticmethod
    def array_to_image(arr):
        # need to return old values to avoid python freeing memory
        arr = arr.transpose(2,0,1)
        c = arr.shape[0]
        h = arr.shape[1]
        w = arr.shape[2]
        arr = np.ascontiguousarray(arr.flat, dtype=np.float32)
        data = arr.ctypes.data_as(POINTER(c_float))
        im = IMAGE(w,h,c,data)
        return im, arr

    def to_bboxes(self, detections, num):
        bboxes = []
        for j in range(num):
            for idx, name in enumerate(self.class_names):
                confidence = detections[j].prob[idx]
                if confidence > 0:
                    box = detections[j].bbox
                    x_min = int(box.x - box.w * 0.5)
                    y_min = int(box.y - box.h * 0.5)
                    x_max = int(box.x + box.w * 0.5)
                    y_max = int(box.y + box.h * 0.5)

                    bboxes.append([x_min, y_min, x_max, y_max, name, confidence])
                    break
        return bboxes

    def predict(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.expand_dims(img, axis=-1)  
        im, arr = self.array_to_image(img)
        pnum = pointer(c_int(0))
        predict_image(self.net, im)
        bboxes = []
        try:
            detections = get_network_boxes(self.net, im.w, im.h,
                                        self.ocr_threshold, self.hier_thresh, None, 0, pnum, 0)
            num = pnum[0]
            if self.nms:
                do_nms_sort(detections, num, len(self.class_names), self.nms)
            bboxes = self.to_bboxes(detections, num)
            bboxes.sort(key=lambda x: x[0])
            
            bboxes = self.ocr_improver.fix_multiline(bboxes)
        finally:
            free_detections(detections, num)
        
        return bboxes
