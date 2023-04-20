import streamlit as st
from streamlit_image_comparison import image_comparison
from PIL import Image


#Create the windown
st.title("Demo App")


#Upload file
file1 = st.file_uploader("Select the image 1", type = ['png', 'jpg'])
file2 = st.file_uploader("Select the image 2", type = ['png', 'jpg'])

#Read image
def read_image(file):
    if file is not None:
        image = Image.open(file)
    
    else:
        image = None
        
    return image
   



#Main
image1 = read_image(file1)
image2 = read_image(file2)
#Compare
if (image1 and image2) is not None:
    image_comparison(image1, image2, label1 = "pipeline cơ bản", label2 = "pipeline chia nhỏ ảnh")
