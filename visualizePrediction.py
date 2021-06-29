
# Load required packages
import sys

sys.path.append('/env/lib/python3.6/site-packages')
from cv2 import cv2
import numpy as np
from keras import backend as K
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
import tensorflow as tf
from matplotlib import pyplot as plt
import os
import pandas


# Load required scripts
from scripts.config import initConfigFile, readConfigFile, getWidth, getHeight
from scripts.model import loadModelDefault


def visualizePredictionsDefault(minPercentage = 50, layerName = "conv5_block3_out", classIndex=0):
    """visualizePredictionsDefault
        Arguments:
            minPercentage: Minimum prediction percentage to filter
            layerName: Layer to visualize
            classIndex: Index of class to visualize
    """     
    # Get default values 
    model = loadModelDefault()
    outputPredictDirectory = readConfigFile("DIRECTORY", "outputPredict")
    filePath = os.path.join(outputPredictDirectory, "prediction.csv")
    width = getWidth()
    height = getHeight()


    # Run routine
    visualizePredictions(model, filePath, width, height, minPercentage, outputPredictDirectory, layerName, classIndex)


def visualizeTestPredictionsDefault(minPercentage = 50, layerName = "conv5_block3_out", classIndex=0):
    """visualizeTestPredictionsDefault
        Arguments:
            minPercentage: Minimum prediction percentage to filter
            layerName: Layer to visualize
            classIndex: Index of class to visualize
    """     
    # Get default values 
    model = loadModelDefault()
    outputPredictDirectory = readConfigFile("DIRECTORY", "outputTest")
    filePath = os.path.join(outputPredictDirectory, "prediction.csv")
    width = getWidth()
    height = getHeight()


    # Run routine
    visualizePredictions(model, filePath, width, height, minPercentage, outputPredictDirectory, layerName, classIndex)


def visualizePredictions(model, filePath, width, height, minPercentage, outputPredictDirectory, layerName = "res5c_branch2c", classIndex=0):
    """visualizePredictions
        Arguments:
            model: ConvNet model
            filePath: Image to visualize
            width: Image width
            height: Image height
            minPercentage: Minimum prediction percentage to filter
            layerName: Layer to visualize
            classIndex: Class to visualize
    """ 
    predictions = pandas.read_csv(filePath, sep=";" )    
    predictions = predictions[predictions.percentage >= minPercentage]
    for index, row in predictions.iterrows():
        visualizeGradCam(model, layerName, row['filePath'], width, height, outputPredictDirectory, classIndex)


def visualizeGradCam(model, layerName, filePath, width, height, outputPredictDirectory, classIndex=0):
    """visualizeGradCam
        Arguments:
            model: ConvNet model
            layerName: Layer to visualize            
            filePath: Image to visualize 
            width: Image width
            height: Image height
            outputPredictDirectory: Output directory
            classIndex: Class to visualize
    """     
    img = loadImage(filePath, height, width)
    gradcam = computeGradCam(model, img, width, height, classIndex, layerName)
    jetcam = computeJetCam(gradcam, filePath, width, height)
    cv2.imwrite(os.path.join(outputPredictDirectory, os.path.basename(filePath)), jetcam)
    

def loadImage(filePath, width, height, preprocess=True):
    """Load and preprocess image
        Arguments:
            filePath: File to load
            width: Image width
            height: Image height
            preprocess: Prprocess image 
        Return:
            Image tensor
    """
    img = image.load_img(filePath, target_size=(width, height))
    if preprocess:
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
    return img


def computeGradCam(model, image, width, height, classIndex, layerName):
    """GradCAM method for visualizing input saliency
        Arguments:
            model: ConvNet model
            image: Image
            width: Image width
            height: Image height
            classIndex: Class to highlight
            layerName: ConvNet Layer
        Return:
            Grad-CAM tensor
    """    
    y_c = model.output[0, classIndex]
    conv_output = model.get_layer(layerName).output
    grads = K.gradients(y_c, conv_output)[0]
    # grads = tf.GradientType(y_c, conv_output)[0]
    # grads = normalize(grads)
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
    """normalize
        Arguments:
            grads: Tensor
        Return:
            Normalized tensor by its L2 norm
    """
    return (grads + 1e-10) / (K.sqrt(K.mean(K.square(grads))) + 1e-10)


def computeJetCam(gradcam, filePath, width, height):
    """getJetCam
        Arguments:
            gradcam: Grad-CAM
            filePath: File to apply grad-CAM
        Return:
            jetcam
    """
    jetcam = cv2.applyColorMap(np.uint8(255 * gradcam), cv2.COLORMAP_JET)
    jetcam = (np.float32(jetcam) + loadImage(filePath, width, height, preprocess=False)) / 2
    jetcam = np.uint8(jetcam)
    return jetcam


def main():
    """main

    """    
    visualizePredictionsDefault()


# Execute main routine
if __name__ == '__main__':
    tf.compat.v1.disable_eager_execution()
    main()

