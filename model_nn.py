from compute_content_cost import *
from compute_layer_style_cost import *
from compute_style_cost import *
from gram_matrix import *
from total_cost import *
from glb import *

import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
from nst_utils import *
import numpy as np
import tensorflow as tf

def model_nn(sess, input_image, num_iterations = 200):
    
    saver = tf.train.Saver()
    
    config = tf.ConfigProto(allow_soft_placement=True)
#    config.gpu_options.allocator_type = 'BFC'
#    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.85
    with tf.device("/gpu:0"):
     with tf.Session(config=config) as sess:
#    
#    with tf.Session() as sess:
    
        # Initialize global variables (you need to run the session on the initializer)
        ### START CODE HERE ### (1 line)
        sess.run(tf.global_variables_initializer())
        ### END CODE HERE ###
        
        # Run the noisy input image (initial generated image) through the model. Use assign().
        ### START CODE HERE ### (1 line)
        sess.run(model['input'].assign(input_image))
        ### END CODE HERE ###
        
        for i in range(num_iterations):
        
            # Run the session on the train_step to minimize the total cost
            ### START CODE HERE ### (1 line)
            sess.run(train_step)
            ### END CODE HERE ###
            
#            if (os.path.exists("/media/ankur98/0F5E1B3E0F5E1B3E/Projects/art generation/chk_pt/cp.ckpt"))
#                saver.restore(sess, "/media/ankur98/0F5E1B3E0F5E1B3E/Projects/art generation/chk_pt/cp.ckpt")
#                print("Model restored.")

            
            # Compute the generated image by running the session on the current model['input']
            ### START CODE HERE ### (1 line)
            generated_image = sess.run(model['input'])
            ### END CODE HERE ###
    
            # Print every 20 iteration.
            if i%10 == 0:
                Jt, Jc, Js = sess.run([J, J_content, J_style])
                print("Iteration " + str(i) + " :")
                print("total cost = " + str(Jt))
                print("content cost = " + str(Jc))
                print("style cost = " + str(Js))
                
#                save_path = saver.save(sess, "/media/ankur98/0F5E1B3E0F5E1B3E/Projects/art generation/chk_pt/cp.ckpt")
#                print("Model saved in file: %s" % save_path)  
                
                # save current generated image in the "/output" directory
                save_image("output/" + str(i) + ".png", generated_image)
        
        # save last generated image
        save_image('output/generated_image.jpg', generated_image)
    
    return generated_image