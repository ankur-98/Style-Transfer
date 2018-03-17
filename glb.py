from compute_content_cost import *
from compute_layer_style_cost import *
from compute_style_cost import *
from gram_matrix import *
from total_cost import *

import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
import cv2
import numpy as np
import tensorflow as tf
from tkinter import *
from tkinter.filedialog import askopenfilename

from nst_utils import *

def getImg(path):
    IMG_SIZE = 300
    img = scipy.misc.imread(path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), cv2.INTER_LINEAR)
    return img

# Reset the graph
tf.reset_default_graph()

# Start interactive session
sess = tf.Session()

root = Tk()
root.withdraw()

content_img_path = askopenfilename(title = "Select Content image")
content_image = getImg(content_img_path)
#content_image = scipy.misc.imread("/media/ankur98/0F5E1B3E0F5E1B3E/Projects/art generation/images/content300.jpg")
content_image = reshape_and_normalize_image(content_image)

style_img_path = askopenfilename(title = "Select Style image")
style_image = getImg(style_img_path)
#style_image = scipy.misc.imread("/media/ankur98/0F5E1B3E0F5E1B3E/Projects/art generation/images/stone_style.jpg")
style_image = reshape_and_normalize_image(style_image)

root.destroy()

generated_image = generate_noise_image(content_image)
imshow(generated_image[0])

model = load_vgg_model("/media/ankur98/0F5E1B3E0F5E1B3E/Projects/art generation/pretrained-model/imagenet-vgg-verydeep-19.mat")

# Assign the content image to be the input of the VGG model.  
sess.run(model['input'].assign(content_image))

# Select the output tensor of layer conv4_2
out = model['conv4_3']

# Set a_C to be the hidden layer activation from the layer we have selected
a_C = sess.run(out)

# Set a_G to be the hidden layer activation from same layer. Here, a_G references model['conv4_2'] 
# and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
# when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
a_G = out
    
# Compute the content cost
J_content = compute_content_cost(a_C, a_G)

# Assign the input of the model to be the "style" image 
sess.run(model['input'].assign(style_image))

# Compute the style cost
J_style = compute_style_cost(model, STYLE_LAYERS, sess)

J = total_cost(J_content, J_style, alpha = 9, beta = 63)

# define optimizer (1 line)
optimizer = tf.train.AdamOptimizer()

# define train_step (1 line)
train_step = optimizer.minimize(J)
