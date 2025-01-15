# pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tf2onnx
# pip install -i https://pypi.tuna.tsinghua.edu.cn/simple onnxmltools
# pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tensorflow>=2.2.0

import onnxmltools
import onnx
import numpy as np
import onnxruntime
# import cv2
import keras
# from keras.models import load_model
from keras.models import load_model


def h5_to_onnx(input_h5, output_onnx):
    model = load_model(input_h5)
    onnx_model = onnxmltools.convert_keras(model, model.name)
    onnx.save_model(onnx_model, output_onnx)


'''
def h5_to_onnx(input_h5, output_onnx):
    model = load_model(input_h5)
    onnx_model =keras2onnx.convert_keras(model, model.name)
    onnx.save_model(onnx_model, output_onnx)
'''

input_h5 = './models/newest_PE.h5'
output_onnx = './models/newest_PE.onnx'

h5_to_onnx(input_h5, output_onnx)
