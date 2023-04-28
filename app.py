import time

import numpy as np
import streamlit as st
from PIL import Image
from streamlit_image_comparison import image_comparison

from lib_utils.utils import draw_bbox
from main import ObjectDetector

@st.cache_resource
def load_detector():
    detector = ObjectDetector("yolov5s")
    return detector

detector = load_detector()


#Create the windown
st.title("Demo App")

#Upload file
file1 = st.file_uploader("Select the image", type = ['png', 'jpg', 'jpeg'])


#Read image
def read_image(file):
    if file is not None:
        image = Image.open(file)
        image = np.array(image)
    else:
        image = None
    return image

if file1 is not None:
    #Main
    image = read_image(file1)
    slice_width = st.slider("slice width", 0, image.shape[0], 320, 1)
    slice_height = st.slider("slice height", 0, image.shape[1], 320, 1)
    overlap_ratio = st.slider("overlap ratio", 0., 1., 0.2, 0.01)

    pred_sig = st.button("Predict")
    if pred_sig:
        start_time = time.time()
        res = detector.predict(image)[0].numpy()
        res_image = draw_bbox(image, res)
        end_time = time.time() - start_time

        slice_start_time = time.time()
        slice_res = detector.predict_slices(
            image,
            slice_width,
            slice_height,
            overlap_ratio
        )
        slice_res_image = draw_bbox(image, slice_res)
        slice_end_time = time.time() - slice_start_time

        #Compare
        image_comparison(
            res_image,
            slice_res_image,
            label1="pipeline cơ bản: %d objects in %.2f secs" % (len(res), end_time),
            label2="pipeline chia nhỏ ảnh: %d objects in %.2f secs" % (len(slice_res), slice_end_time))
