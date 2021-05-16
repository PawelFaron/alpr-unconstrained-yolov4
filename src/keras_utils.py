import numpy as np
import cv2

from os.path import splitext

import tensorflow as tf

from src.label import Label
from src.projection_utils import getRectPts, find_T_matrix
from tensorflow.keras.models import model_from_json
from tensorflow.keras import backend as K

class DLabel (Label):
    def __init__(self, cl, pts, prob):
        self.pts = pts
        tl = np.amin(pts, 1)
        br = np.amax(pts, 1)
        Label.__init__(self, cl, tl, br, prob)

def getWH(shape):
	return np.array(shape[1::-1]).astype(float)

def reconstruct(Iorig, I, Y, out_size, threshold=.9, alphas=[0.6]):
    net_stride = 2**4
    side = ((208. + 40.)/2.)/net_stride  # 7.75

    Probs = Y[..., 0]
    Affines = Y[..., 2:]

    xx, yy = np.where(Probs > threshold)

    WH = getWH(I.shape)
    MN = WH/net_stride

    res_data = []
    for alpha in alphas:
        vxx = vyy = alpha
        
        def base(vx, vy): return np.matrix(
            [[-vx, -vy, 1.], [vx, -vy, 1.], [vx, vy, 1.], [-vx, vy, 1.]]).T

        labels = []

        for i in range(len(xx)):
            y, x = xx[i], yy[i]
            affine = Affines[y, x]
            prob = Probs[y, x]

            mn = np.array([float(x) + .5, float(y) + .5])

            A = np.reshape(affine, (2, 3))
            A[0, 0] = max(A[0, 0], 0.)
            A[1, 1] = max(A[1, 1], 0.)

            pts = np.array(A*base(vxx, vyy))  # *alpha
            pts_MN_center_mn = pts * side
            pts_MN = pts_MN_center_mn + mn.reshape((2, 1))

            pts_prop = pts_MN/MN.reshape((2, 1))

            labels.append(DLabel(0, pts_prop, prob))

        labels.sort(key=lambda a: a._Label__prob, reverse=True)

        for i, label in enumerate(labels[:3]):
            ptsh = np.concatenate((label.pts*getWH(Iorig.shape).reshape((2, 1)), np.ones((1, 4))))

            # x1 = max(0, int(ptsh[0][0]))
            # y1 = max(0, int(ptsh[1][0]))
            # x3 = min(Iorig.shape[1], int(ptsh[0][2]))
            # y3 = min(Iorig.shape[0], int(ptsh[1][2]))

            # width = (x3 - x1)

            t_ptsh = getRectPts(0, 0, out_size[0], out_size[1])
            H = find_T_matrix(ptsh, t_ptsh)
            Ilp = cv2.warpPerspective(Iorig, H, out_size, borderValue=.0)

            res_data.append(Ilp)

    return res_data


def load_model(path):
    per_process_gpu_memory_fraction=0.1
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=per_process_gpu_memory_fraction)
    config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
    # config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)

    path = splitext(path)[0]
    with open('%s.json' % path, 'r') as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json)
    model.load_weights('%s.h5' % path)
    return model


def detect_lp(model, image, max_dim, net_step, out_size, threshold, alphas):
    min_dim_img = min(image.shape[:2])
    factor = float(max_dim) / min_dim_img

    w, h = (np.array(image.shape[1::-1], dtype=float)
            * factor).astype(int).tolist()
    w += (w % net_step != 0) * (net_step - w % net_step)
    h += (h % net_step != 0) * (net_step - h % net_step)
    image_resized = cv2.resize(image, (w, h))

    T = image_resized.copy()
    T = T.reshape((1, T.shape[0], T.shape[1], T.shape[2]))

    Yr = model.predict(T)
    Yr = np.squeeze(Yr)

    return reconstruct(image, image_resized, Yr, out_size, threshold, alphas)
