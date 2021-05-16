from src.keras_utils import load_model, detect_lp
import numpy as np

multiplier = 1. / 255.
class PlateFinder:
    def __init__(self, lp_finder_threshold, alphas):
        self.model = load_model("data/lp-detector/wpod-net_update1.h5")
        self.lp_threshold = lp_finder_threshold
        self.alphas = alphas

    @staticmethod
    def im2single(I):
        return I.astype('float32') * multiplier

    def find(self, image):
        ratio = float(max(image.shape[:2])) / min(image.shape[:2])
        side = int(ratio * 288.)
        bound_dim = min(side + (side % (2 ** 4)), 608)

        lp_images = detect_lp(self.model, self.im2single(image), bound_dim, 2 ** 4, (240, 80), self.lp_threshold, self.alphas)

        return lp_images
