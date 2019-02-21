
from urllib.request import urlopen
from datetime import datetime

from keras.models import model_from_json
import numpy as np
import os
import datetime
import cv2
from PIL import Image

model_filename = 'lego.json'
weghts_filename = 'lego.h5'
labels_filename = 'lego_clases.txt'

network_input_size = 200

output_layer = 'loss:0'
input_node = 'Placeholder:0'

loaded_model = None
classes = {}

def initialize():
    global loaded_model
    global classes
    loaded_model = None
    classes = None
    classes = {}
    print('Loading model...', end='')
    # load json and create model
    json_file = open('lego.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("lego.h5")

    loaded_model.compile(loss='categorical_crossentropy',
                        optimizer='rmsprop',
                        metrics=['accuracy']) 
    print('Success!')
    print('Loading labels...', end='')
    with open("lego_classes.txt", 'r', encoding='utf-8') as classes_file:
        for line in classes_file:
            item = line.split(":")
            classes.update({int(item[0]):item[1]})
    print(len(classes), 'found. Success!')

def log_msg(msg):
    print("{}: {}".format(datetime.datetime.now(),msg))

def extract_bilinear_pixel(img, x, y, ratio, xOrigin, yOrigin):
    xDelta = (x + 0.5) * ratio - 0.5
    x0 = int(xDelta)
    xDelta -= x0
    x0 += xOrigin
    if x0 < 0:
        x0 = 0;
        x1 = 0;
        xDelta = 0.0;
    elif x0 >= img.shape[1]-1:
        x0 = img.shape[1]-1;
        x1 = img.shape[1]-1;
        xDelta = 0.0;
    else:
        x1 = x0 + 1;
    
    yDelta = (y + 0.5) * ratio - 0.5
    y0 = int(yDelta)
    yDelta -= y0
    y0 += yOrigin
    if y0 < 0:
        y0 = 0;
        y1 = 0;
        yDelta = 0.0;
    elif y0 >= img.shape[0]-1:
        y0 = img.shape[0]-1;
        y1 = img.shape[0]-1;
        yDelta = 0.0;
    else:
        y1 = y0 + 1;

    #Get pixels in four corners
    bl = img[y0, x0]
    br = img[y0, x1]
    tl = img[y1, x0]
    tr = img[y1, x1]
    #Calculate interpolation
    b = xDelta * br + (1. - xDelta) * bl
    t = xDelta * tr + (1. - xDelta) * tl
    pixel = yDelta * t + (1. - yDelta) * b
    return pixel.astype(np.uint8)

def extract_and_resize(img, targetSize):
    determinant = img.shape[1] * targetSize[0] - img.shape[0] * targetSize[1]
    if determinant < 0:
        ratio = float(img.shape[1]) / float(targetSize[1])
        xOrigin = 0
        yOrigin = int(0.5 * (img.shape[0] - ratio * targetSize[0]))
    elif determinant > 0:
        ratio = float(img.shape[0]) / float(targetSize[0])
        xOrigin = int(0.5 * (img.shape[1] - ratio * targetSize[1]))
        yOrigin = 0
    else:
        ratio = float(img.shape[0]) / float(targetSize[0])
        xOrigin = 0
        yOrigin = 0
    resize_image = np.empty((targetSize[0], targetSize[1], img.shape[2]), dtype=np.uint8)
    for y in range(targetSize[0]):
        for x in range(targetSize[1]):
            resize_image[y, x] = extract_bilinear_pixel(img, x, y, ratio, xOrigin, yOrigin)
    return resize_image

def extract_and_resize_to_200_square(image):
    h, w = image.shape[:2]
    log_msg("crop_center: " + str(w) + "x" + str(h) +" and resize to " + str(200) + "x" + str(200))
    return extract_and_resize(image, (200, 200))

def crop_center(img,cropx,cropy):
    h, w = img.shape[:2]
    startx = max(0, w//2-(cropx//2) - 1)
    starty = max(0, h//2-(cropy//2) - 1)
    log_msg("crop_center: " + str(w) + "x" + str(h) +" to " + str(cropx) + "x" + str(cropy))
    return img[starty:starty+cropy, startx:startx+cropx]

def resize_down_to_1600_max_dim(image):
    w,h = image.size
    if h < 1600 and w < 1600:
        return image

    new_size = (1600 * w // h, 1600) if (h > w) else (1600, 1600 * h // w)
    log_msg("resize: " + str(w) + "x" + str(h) + " to " + str(new_size[0]) + "x" + str(new_size[1]))
    if max(new_size) / max(image.size) >= 0.5:
        method = Image.BILINEAR
    else:
        method = Image.BICUBIC
    return image.resize(new_size, method)

def predict_url(imageUrl):
    log_msg("Predicting from url: " +imageUrl)
    with urlopen(imageUrl) as testImage:
        image = Image.open(testImage)
        return predict_image(image)

def convert_to_nparray(image):
    # RGB -> BGR
    log_msg("Convert to numpy array")
    image = np.array(image)
    return image[:, :, (2,1,0)]

def update_orientation(image):
    exif_orientation_tag = 0x0112
    if hasattr(image, '_getexif'):
        exif = image._getexif()
        if exif != None and exif_orientation_tag in exif:
            orientation = exif.get(exif_orientation_tag, 1)
            log_msg('Image has EXIF Orientation: ' + str(orientation))
            # orientation is 1 based, shift to zero based and flip/transpose based on 0-based values
            orientation -= 1
            if orientation >= 4:
                image = image.transpose(Image.TRANSPOSE)
            if orientation == 2 or orientation == 3 or orientation == 6 or orientation == 7:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
            if orientation == 1 or orientation == 2 or orientation == 5 or orientation == 6:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
    return image
                
def predict_image(image):
        
    log_msg('Predicting image')
    try:
        if image.mode != "RGB":
            log_msg("Converting to RGB")
            image = image.convert("RGB")

        w,h = image.size
        log_msg("Image size: " + str(w) + "x" + str(h))
        
        # Update orientation based on EXIF tags
        image = update_orientation(image)

        # If the image has either w or h greater than 1600 we resize it down respecting
        # aspect ratio such that the largest dimention is 1600
        image = resize_down_to_1600_max_dim(image)

        # Convert image to numpy array
        image = convert_to_nparray(image)
        
        # Crop the center square and resize that square down to 200x200
        resized_image = extract_and_resize_to_200_square(image)

        # Crop the center for the specified network_input_Size
        cropped_image = crop_center(resized_image, network_input_size, network_input_size)

        img = np.reshape(cropped_image,[1,200,200,3])

        initialize()

        predicted_classes = loaded_model.predict_classes(img)

        predicted_class = str(classes[0])

        response = { 
            'id': '',
            'project': '',
            'iteration': '',
            'created': datetime.datetime.utcnow().isoformat(),
            'predictions': predicted_class 
        }

        log_msg("Results: " + str(response))
        return response
            
    except Exception as e:
        log_msg(str(e))
        return 'Error: Could not preprocess image for prediction. ' + str(e)
