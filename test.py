import yaml
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec



def read_file(file_path):
    with open(file_path, 'r') as f:
        data = list(yaml.safe_load_all(f))
    return data



def show_parameter_config(image, data ,thickness = 2):
  colour = (255,0,0) #draw a blue line
  start_point = (0,0)
  end_point = (image.shape[1], image.shape[0])  
  image = cv2.putText(image, str(data[0]), (0,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, thickness = 1)
  return image



def show_image(image,i):
    fig.add_subplot(gs[i])
    plt.axis('off')
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    
path = r"C:\Users\A715\Desktop\Python_code\config"


image_path = r"C:\Users\A715\Desktop\Python_code\tomato.jpg"



fig = plt.figure()
gs = gridspec.GridSpec(2,3)
gs.update(wspace = 0.0, hspace = 0.0)



for index, file in enumerate(os.listdir(path)):
    file_path = f"{path}/{file}"
    data = read_file(file_path)
    image_input = cv2.imread(image_path)
    image = show_parameter_config(image_input, data = data)
    show_image(image = image, i = index)

plt.show()       






