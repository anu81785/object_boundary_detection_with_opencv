#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
from rembg.bg import remove


 #image_path
image_path="/home/anusaini/Downloads/ProductizeTech/TESTIMAGES/2.jpg"

original_image = cv2.imread(image_path)
original_image = cv2.resize(original_image, (1000, 500))

i=0
while i==0:   
    final_image=original_image.copy()
    if cv2.waitKey(5000) & 0xff==ord("q"):
        print("Key 'q' is pressed, You can not select the 'Region of Interest', now")
        break       
        
    #select ROI function
    roi = cv2.selectROI('image', original_image)
    
    # store x,y,w,h values for roi
    x, y, w, h=roi

    #Crop selected roi from raw image
    roi_cropped = original_image[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]

    img_rembg=remove(roi_cropped)

    gray=cv2.cvtColor(img_rembg, cv2.COLOR_RGB2GRAY)
    _,binary=cv2.threshold(gray, 5,250, cv2.THRESH_BINARY_INV)

    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    area=[cv2.contourArea(c) for c in contours]


    max_area_index=np.argmax(area)


    area.pop(max_area_index)

    contours.pop(max_area_index)

    max_area_index=np.argmax(area)
    
    
    #Create a white background image with same size of original image
    array = np.zeros([h, w, 3], dtype = np.uint8)
    array[:, :] = [255, 255, 255]
    
    #Draw the contours on white background 

    im=cv2.drawContours(array, [contours[max_area_index]], 0, (0, 255, 0), 3)
    
    #In this special case, take the alpha channel of the overlay image, and
    # check for value 0; idx is a Boolean array
    idx = im[:, :, 2] == 0

    # Position for overlay image
    left, top, height, width = (x, y, h, w)

    # Access region of interest with overlay image's dimensions at position
    #   img[top:top+h, left:left+w]   and there, use Boolean array indexing
    # to set the color to red (for example)
    final_image[top:top+height, left:left+width, :][idx] = (0, 255, 0)

    # Show the final image
    cv2.imshow('image', final_image)
    print("Press key 'c' to clear the outline")
    print("Press key 'q' to stop the process")
    k=cv2.waitKey(5000) & 0xff
    if k==ord("c"):
        print("Key 'c' is pressed, outline is removed")
        cv2.imshow('image', original_image)
        continue
    original_image=final_image







