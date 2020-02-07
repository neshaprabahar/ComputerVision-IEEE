import numpy as np
import cv2
from matplotlib import pyplot as plt
######
import skimage
from skimage.viewer import ImageViewer
import sys
######

# image = skimage.io.imread(fname='./simulated_images_input/'+'s_' + 1 + '.png')
# viewer = ImageViewer(image)
# viewer.show()






##sigma = float(sys.argv[2])



# Create FAST object with default values
fast = cv2.FastFeatureDetector_create()

# Plot features on images from the simulator
for i in range(10):
    num = str(i + 1)
    filename_i = 's_' + num + '.png'
    img = cv2.imread('./simulated_images_input/' + filename_i,1)
    # find and draw the keypoints
    kp = fast.detect(img, None)
    img2 = cv2.drawKeypoints(img, kp, None, color=(0,255,0))
    # plt.imshow(img2),plt.show()


    ##Gaussian Blur Application
    

    # blur = cv2.GaussianBlur(img,(5,5),0)
    # cv2.imshow("Gaussian Smoothing",np.hstack((img, blur)))





    filename_o = 'FAST_s_' + num + '.png'
    cv2.imwrite('./FAST_output/FAST_simulated_output/' + filename_o,img2)

# Plot features on images from playing field



##### look into changing the intensity for FAST - is there a parameter i can provide/change

for i in range(10):
    num = str(i + 1)
    filename_i = 'r_' + num + '.png'   
    img = cv2.imread('./real_images_input/' + filename_i,1)
    # find and draw the keypoints
    kp = fast.detect(img, None)

    ##Gaussian Blur Application
    # blurred = skimage.filters.gaussian(img, (5,5), truncate=3.5, multichannel=True)
    # viewer = ImageViewer(blurred)
    blurred = cv2.blur(img,(100, 100))
    # viewer.show()



    img2 = cv2.drawKeypoints(blurred, kp, None, color=(0,255,0))
    plt.imshow(img2),plt.show()

    filename = 'FAST_r_' + num + '.png'
    cv2.imwrite('./FAST_output/FAST_real_output/' + filename,img2)