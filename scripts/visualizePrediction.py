# Load required packages
import sys
sys.path.append('/env/lib/python3.6/site-packages')

import cv2
import numpy as np
from keras import backend as K
from keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import tensorflow as tf
from matplotlib import pyplot as plt
import os
import pandas
from scipy.ndimage.interpolation import zoom


# Load required scripts
from .config import initConfigFile, readConfigFile, getWidth, getHeight
from .model import loadModelDefault


def visualizePredictionsDefault(minPercentage = 50, layerName = "conv5_block3_out", classIndex=0):
    # Get default values 
    model = loadModelDefault()
    outputPredictDirectory = readConfigFile("DIRECTORY", "outputPredict")
    filePath = os.path.join(outputPredictDirectory, "prediction.csv")
    width = getWidth()
    height = getHeight()


    # Run routine
    visualizePredictions(model, filePath, width, height, minPercentage, outputPredictDirectory, layerName, classIndex)


def visualizeTestPredictionsDefault(minPercentage = 50, layerName = "conv5_block3_out", classIndex=0):
    # Get default values 
    model = loadModelDefault()
    outputPredictDirectory = readConfigFile("DIRECTORY", "outputTest")
    filePath = os.path.join(outputPredictDirectory, "prediction.csv")
    width = getWidth()
    height = getHeight()


    # Run routine
    visualizePredictions(model, filePath, width, height, minPercentage, outputPredictDirectory, layerName, classIndex)


def visualizePredictions(model, filePath, width, height, minPercentage, outputPredictDirectory, layerName = "res5c_branch2c", classIndex=0):
    # Disable eager execution
    tf.compat.v1.disable_eager_execution()
    predictions = pandas.read_csv(filePath, sep=";" )    
    predictions = predictions[predictions.percentage >= minPercentage]
    for index, row in predictions.iterrows():
        visualizeGradCam(model, layerName, row['filePath'], width, height, outputPredictDirectory, classIndex)


def visualizeGradCam(model, layerName, filePath, width, height, outputPredictDirectory, classIndex=0):
    img = loadImage(filePath, height, width)
    #gradcam = computeGradCam(model, img, width, height, classIndex, layerName)
    gradcam = computeGradCamPlus(model, img, width, height, classIndex, layerName)
    jetcam = computeJetCam(gradcam, filePath, width, height)
    cv2.imwrite(os.path.join(outputPredictDirectory, os.path.basename(filePath)), jetcam)
    

def loadImage(filePath, width, height, preprocess=True):
    img = tf.keras.utils.load_img(filePath, target_size=(width, height))
    if preprocess:
        img = tf.keras.utils.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
    return img


def computeGradCam(model, image, width, height, classIndex, layerName):
    y_c = model.output[0, classIndex]
    conv_output = model.get_layer(layerName).output
    grads = K.gradients(y_c, conv_output)[0]
    gradient_function = K.function([model.input], [conv_output, grads])

    output, grads_val = gradient_function([image])
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    weights = np.mean(grads_val, axis=(0, 1))
    gradcam = np.dot(output, weights)

    gradcam = cv2.resize(gradcam, (width, height), cv2.INTER_LINEAR)
    gradcam = np.maximum(gradcam, 0)
    gradcamMax = gradcam.max() 
    if gradcamMax != 0: 
        gradcam = gradcam / gradcamMax
    return gradcam    


def computeGradCamPlus(model, image, width, height, classIndex, layerName):
    y_c = model.output[0, classIndex]
    conv_output = model.get_layer(layerName).output
    grads = K.gradients(y_c, conv_output)[0]

    first = K.exp(y_c)*grads
    second = K.exp(y_c)*grads*grads
    third = K.exp(y_c)*grads*grads*grads

    gradient_function = K.function([model.input], [y_c,first,second,third, conv_output, grads])
    y_c, conv_first_grad, conv_second_grad,conv_third_grad, conv_output, grads_val = gradient_function([image])
    global_sum = np.sum(conv_output[0].reshape((-1,conv_first_grad[0].shape[2])), axis=0)

    alpha_num = conv_second_grad[0]
    alpha_denom = conv_second_grad[0]*2.0 + conv_third_grad[0]*global_sum.reshape((1,1,conv_first_grad[0].shape[2]))
    alpha_denom = np.where(alpha_denom != 0.0, alpha_denom, np.ones(alpha_denom.shape))
    alphas = alpha_num/alpha_denom

    weights = np.maximum(conv_first_grad[0], 0.0)

    alpha_normalization_constant = np.sum(np.sum(alphas, axis=0),axis=0)

    alphas /= alpha_normalization_constant.reshape((1,1,conv_first_grad[0].shape[2]))

    deep_linearization_weights = np.sum((weights*alphas).reshape((-1,conv_first_grad[0].shape[2])),axis=0)
    grad_CAM_map = np.sum(deep_linearization_weights*conv_output[0], axis=2)

    # Passing through ReLU
    cam = np.maximum(grad_CAM_map, 0)
    cam = zoom(cam,height/cam.shape[0])
    # scale 0 to 1.0
    cam = cam / np.max(cam)
    return cam


def computeJetCam(gradcam, filePath, width, height):
    jetcam = cv2.applyColorMap(np.uint8(255 * gradcam), cv2.COLORMAP_JET)
    jetcam = (np.float32(jetcam) + loadImage(filePath, width, height, preprocess=False)) / 2
    jetcam = np.uint8(jetcam)
    return jetcam
