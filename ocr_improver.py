import statistics

class OCRImprover:
    def fix_multiline(self, ocr_bboxes):
        if len(ocr_bboxes) == 0:
            return ocr_bboxes
        medium_box_size = statistics.mean([box[3] - box[1] for box in ocr_bboxes])
        line1 = [ocr_bboxes[0]]
        line2 = []
        prev_line = 1

        for i, box in enumerate(ocr_bboxes[:-1]):
            next_box = ocr_bboxes[i+1]
            y_difference = box[1] - next_box[1]
            x_difference = next_box[0] - box[0]

            if abs(y_difference) > medium_box_size * 0.6:
                if prev_line == 1:
                    line2.append(next_box)
                    prev_line = 2
                else:
                    line1.append(next_box)
                    prev_line = 1
            else:
                if prev_line == 1:
                    line1.append(next_box)
                else:
                    line2.append(next_box)

        if len(line1) == 0 or len(line2) == 0:
            return ocr_bboxes

        line1 = sorted(line1, key=lambda value: value[0]) 
        line2 = sorted(line2, key=lambda value: value[0]) 

        if line1[0][1] > line2[0][1]:
            ocr_bboxes = line2 + line1
        else:
            ocr_bboxes = line1 + line2

        return ocr_bboxes