import onnx
import pandas as pd
from pathlib import Path
import tensorflow as tf
import onnxruntime as rt
import numpy as np
# import onnx_tf
# from onnx_tf.backend import prepare
# Load the onnx model with onnx.load
from sklearn.metrics import accuracy_score

onnx_model = onnx.load("./models/newest_PEseg.onnx")
onnx.checker.check_model(onnx_model)


# Create inference session using ort.InferenceSession
def proc_img(filepath):
    """ Create a DataFrame with the filepath and the labels of the pictures
    """

    labels = [str(filepath[i]).split("\\")[-2] for i in range(len(filepath))]

    filepath = pd.Series(filepath, name='Filepath').astype(str)
    labels = pd.Series(labels, name='Label')

    # Concatenate filepaths and labels
    df = pd.concat([filepath, labels], axis=1)

    # Shuffle the DataFrame and reset index
    df = df.sample(frac=1, random_state=0).reset_index(drop=True)

    return df


dir_ = Path('./data/PE/BM/test')
test_filepaths = list(dir_.glob(r'**/*.BMP'))
print(len(test_filepaths))
test_df = proc_img(test_filepaths)

print(f'Number of pictures in the test dataset: {test_df.shape[0]}\n')

test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    # preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
)

test_images = test_generator.flow_from_dataframe(
    dataframe=test_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(64, 64),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=1,
    shuffle=False
)
# test_images
type(test_images[0][0])
# print(test_images[0].shape)
print(test_images[0][0].shape)
print(len(test_images))
# tf_rep = prepare(onnx_model)
# tf_rep

session = rt.InferenceSession("./models/newest_PEseg.onnx", None, providers=['CPUExecutionProvider'])

pred = []
for i in range(len(test_images)):
    test_image = test_images[i]
    results = session.run([], {session.get_inputs()[0].name: test_image[0]})
    res = np.argmax(np.array(results[0]), axis=1)
    pred.extend(res)
    # print()
    # print(len(pred))

# pred = onnx_model.run(test_images)
# pred = np.argmax(pred, axis=1)


# Map the label
labels = (test_images.class_indices)
labels = dict((v, k) for k, v in labels.items())
pred = [labels[k] for k in pred]

# Get the accuracy on the test set
y_test = list(test_df.Label)

acc = accuracy_score(y_test, pred)
print(f'# Accuracy on the test set: {acc * 100:.2f}%')
"""
test_images[0][0].shape
Out[57]: (28, 224, 224, 3)
"""
