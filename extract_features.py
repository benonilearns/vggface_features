import pandas as pd
from numpy import expand_dims
from matplotlib import pyplot
from PIL import Image
from numpy import asarray
from keras.models import Model
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from keras_vggface.utils import decode_predictions
import os
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)

###cropped_faces = directory with 224x224 face images###

cropped_faces = "/FACE/DIR/HERE"
cropped_paths = os.listdir(cropped_faces)
vgg_model = VGGFace() # pooling: None, avg or max
out = vgg_model.get_layer('fc7').output
vgg_model_fc7 = Model(vgg_model.input, out)


def get_features(file):
    pixels = pyplot.imread(file)
    face_array = asarray(pixels)
    face_array = face_array.astype('float32')
    samples = expand_dims(face_array, axis=0)
    # prepare the face for the model, e.g. center pixels
    samples = preprocess_input(samples, version=1)
    vgg_model_fc7_preds = vgg_model_fc7.predict(samples)
    return vgg_model_fc7_preds

main_df = pd.DataFrame()
count = 0
goal = len(cropped_paths)
with out.get_writer() as writer:
    for file in cropped_paths:
        try:
            preds = get_features(cropped_faces + file)
            df = pd.DataFrame()
            df['preds'] = preds.tolist()
            df['image'] = str(file)
            main_df = main_df.append(df, ignore_index = True)
            count +=1
            if count % 1000 == 0:
                print("Done: " + str(count))
                print("To go: " + str(goal - count))
        except Exception as e:
            print(e)
