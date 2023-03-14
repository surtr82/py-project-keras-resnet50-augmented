# Load required modules
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import pandas
from keras import backend as K
from keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import tensorflow as tf

# Load required scripts
from .config import readConfigFile, getWidth, getHeight
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
    predictions = pandas.read_csv(filePath, sep=";" )    
    predictions = predictions[predictions.percentage >= minPercentage]
    for index, row in predictions.iterrows():
        visualizeGradCam(model, layerName, row['filePath'], width, height, outputPredictDirectory, classIndex)


def visualizeGradCam(model, layerName, filePath, width, height, outputPredictDirectory, classIndex=0):
    img = loadImage(filePath, height, width)
    gradcam = computeGradCam(model, img, width, height, classIndex, layerName)
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

    # Process CAM
    gradcam = cv2.resize(gradcam, (width, height), cv2.INTER_LINEAR)
    gradcam = np.maximum(gradcam, 0)
    gradcamMax = gradcam.max() 
    if gradcamMax != 0: 
        gradcam = gradcam / gradcamMax
    return gradcam
    

def normalize(grads):
    # Normalize tensor by its L2 norm
    return (grads + 1e-10) / (K.sqrt(K.mean(K.square(grads))) + 1e-10)


def computeJetCam(gradcam, filePath, width, height):
    jetcam = cv2.applyColorMap(np.uint8(255 * gradcam), cv2.COLORMAP_JET)
    jetcam = (np.float32(jetcam) + loadImage(filePath, width, height, preprocess=False)) / 2
    jetcam = np.uint8(jetcam)
    return jetcam

