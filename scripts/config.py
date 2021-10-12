# Install required packages
import configparser
import os


def initConfigFile(filePath="config.ini"):
    """initConfigFile
        Arguments:
            filePath: Configfile Path
    """
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
    """readConfigFile
        Arguments:
            section: SinitConfigFileection in init file
            key: key of key-value-pair
            filePath: Configfile Path
        Return:
            value of key-value-pair
    """
    config = configparser.ConfigParser()
    config.read(filePath)
    return config[section][key]


def getNoOfTestFiles():
    """getNoOfTestFiles
        Return:
            Returns number of test files
    """
    return int(readConfigFile("TRAIN", "noOfTestFiles"))


def getPercentageOfValidationFiles():
    """getPercentageOfValidationFiles
        Return:
            Percentage of training files for validation
    """
    return int(readConfigFile("TRAIN", "percentageOfValidationFiles"))


def useShuffledFiles():
    """useShuffledFiles
        Return:
            Use shuffled files
    """
    return (readConfigFile("TRAIN", "shuffleFiles") == "yes")


def getBatchSize():
    """getBatchSize
        Return:
            Returns batch size
    """
    return int(readConfigFile("TRAIN", "batchSize"))


def getMaxFiles():
    """getMaxFiles
        Return:
            Returns maximum amount of files to predict
    """
    return int(readConfigFile("PREDICT", "maxFiles"))


def getWidth():
    """getWidth
        Return:
            Returns image width
    """
    return int(readConfigFile("IMAGE", "width"))


def getHeight():
    """getHeight
        Return:
            Returns image height
    """
    return int(readConfigFile("IMAGE", "height"))


def getDepth():
    """getDepth
        Return:
            Returns image depth
    """
    return int(readConfigFile("IMAGE", "depth"))
