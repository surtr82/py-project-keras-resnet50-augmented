# Install required packages
import configparser
import os


def initConfigFile(filePath="config.ini"):
    if not os.path.isfile(filePath):
        config = configparser.ConfigParser()
        config["DIRECTORY"] = {"input": "input",
                               "inputPredict": "../dataset-mardin-plain/base-dataset/predict",
                               "inputTrain": "../dataset-mardin-plain/base-dataset/train",
                               "dataset": "../dataset-mardin-plain",
                               "datasetPredict": "../dataset-mardin-plain/training-dataset/predict",
                               "datasetPredictSubset": "../dataset-mardin-plain/training-dataset/predict/subset$",
                               "datasetValidate": "../dataset-mardin-plain/training-dataset/validate",
                               "datasetValidateFold": "../dataset-mardin-plain/training-dataset/validate/fold$",    
                               "datasetValidateFoldTrain": "../dataset-mardin-plain/training-dataset/validate/fold$/train",
                               "datasetValidateFoldValidate": "../dataset-mardin-plain/training-dataset/validate/fold$/validate",                               
                               "datasetTrain": "../dataset-mardin-plain/training-dataset/train",
                               "datasetTest": "../dataset-mardin-plain/training-dataset/test",
                               "output": "output",
                               "outputTrain": "output/train",
                               "outputValidate": "output/validate",
                               "outputTest": "output/test",
                               "outputPredict": "output/predict"}

        config["TRAIN"] = {"noOfTestFiles": "100",
                           "noOfFolds": "5",
                           "shuffleFiles": "yes",
                           "batchSize": "64"}

        config["PREDICT"] = {"maxfiles": "2000"}
                          
        config["IMAGE"] = {"width": "224",
                           "height": "224",
                           "depth": "3"}

        with open(filePath, "w") as configfile:
            config.write(configfile)


def readConfigFile(section, key, filePath="config.ini"):
    config = configparser.ConfigParser()
    config.read(filePath)
    return config[section][key]


def getNoOfTestFiles():
    return int(readConfigFile("TRAIN", "noOfTestFiles"))


def getNoOfFolds():
    return int(readConfigFile("TRAIN", "noOfFolds"))


def useShuffledFiles():
    return (readConfigFile("TRAIN", "shuffleFiles") == "yes")


def getBatchSize():
    return int(readConfigFile("TRAIN", "batchSize"))


def getMaxFiles():
    return int(readConfigFile("PREDICT", "maxFiles"))


def getWidth():
    return int(readConfigFile("IMAGE", "width"))


def getHeight():
    return int(readConfigFile("IMAGE", "height"))


def getDepth():
    return int(readConfigFile("IMAGE", "depth"))
