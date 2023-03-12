# Install required packages
import configparser
import os


def initConfigFile(filePath="config.ini"):
    if not os.path.isfile(filePath):
        config = configparser.ConfigParser()
        config["DIRECTORY"] = {"input": "../dataset-mardin/input",
                               "inputPredict": "../dataset-mardin/input/predict",
                               "inputTrain": "../dataset-mardin/input/train",
                               "dataset": "../dataset-mardin/dataset",
                               "datasetPredict": "../dataset-mardin/dataset/predict",
                               "datasetPredictSubset": "../dataset-mardin/dataset/predict/subset$",
                               "datasetValidate": "../dataset-mardin/dataset/validate",
                               "datasetTrain": "../dataset-mardin/dataset/train",
                               "datasetTest": "../dataset-mardin/dataset/test",
                               "output": "output",
                               "outputTrain": "output/train",
                               "outputValidate": "output/validate",
                               "outputTest": "output/test",
                               "outputPredict": "output/predict"}

        config["TRAIN"] = {"noOfTestFiles": "100",
                           "percentageOfValidationFiles": "20",
                           "shuffleFiles": "yes",
                           "batchSize": "40"}

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


def getPercentageOfValidationFiles():
    return int(readConfigFile("TRAIN", "percentageOfValidationFiles"))


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
