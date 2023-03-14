# Load required modules
import math
import os
import random

# Load required scripts
from scripts.config import readConfigFile, getNoOfFolds, getNoOfTestFiles, getNoOfFolds, useShuffledFiles, getBatchSize, getMaxFiles
from scripts.file import getFiles, copyFiles, getDirectories, createDirectory, deleteSubdirectories


def recreateDatasetDefault():
    recreateTrainAndTestDataDefault()
    recreateCrossValidationDataDefault()
    recreatePredictDataDefault()


def recreateTrainAndTestDataDefault():
    # Get default values
    inputTrainDirectory = readConfigFile("DIRECTORY", "inputTrain")
    datasetTrainDirectory = readConfigFile("DIRECTORY", "datasetTrain")
    datasetTestDirectory =  readConfigFile("DIRECTORY", "datasetTest")
    noOfTestFiles = getNoOfTestFiles() 
    shuffleFiles = useShuffledFiles()

    # Run routine
    recreateTrainAndTestData(inputTrainDirectory, datasetTrainDirectory, datasetTestDirectory, noOfTestFiles, shuffleFiles)


def recreateTrainAndTestData(inputTrainDirectory, datasetTrainDirectory, datasetTestDirectory, noOfTestFiles, shuffleFiles):
    deleteTrainAndTestData(datasetTrainDirectory, datasetTestDirectory)
    createTrainAndTestData(inputTrainDirectory, datasetTrainDirectory, datasetTestDirectory, noOfTestFiles, shuffleFiles)


def deleteTrainAndTestDataDefault():
    # Get default values
    datasetTrainDirectory = readConfigFile("DIRECTORY", "datasetTrain")
    datasetTestDirectory =  readConfigFile("DIRECTORY", "datasetTest")    

    # Run routine
    deleteTrainAndTestData(datasetTrainDirectory, datasetTestDirectory)


def deleteTrainAndTestData(datasetTrainDirectory, datasetTestDirectory):
    deleteSubdirectories(datasetTrainDirectory)
    deleteSubdirectories(datasetTestDirectory)


def createTrainAndTestDataDefault():
    # Get default values
    inputTrainDirectory = readConfigFile("DIRECTORY", "inputTrain")
    datasetTrainDirectory = readConfigFile("DIRECTORY", "datasetTrain")
    datasetTestDirectory =  readConfigFile("DIRECTORY", "datasetTest")
    noOfTestFiles = getNoOfTestFiles() 
    shuffleFiles = useShuffledFiles()

    # Run routine
    createTrainAndTestData(inputTrainDirectory, datasetTrainDirectory, datasetTestDirectory, noOfTestFiles, shuffleFiles)


def createTrainAndTestData(inputTrainDirectory, datasetTrainDirectory, datasetTestDirectory, noOfTestFiles, shuffleFiles):
    classDirectories = getDirectories(inputTrainDirectory)
    for classDirectory in classDirectories:
        inputTrainClassDirectory = os.path.join(inputTrainDirectory, classDirectory)  
        inputTrainClassFilePaths = getFiles(inputTrainClassDirectory, True)

        if shuffleFiles: 
            random.shuffle(inputTrainClassFilePaths)

        datasetTrainClassDirectory = os.path.join(datasetTrainDirectory, classDirectory)
        createDirectory(datasetTrainClassDirectory)
        copyFiles(inputTrainClassFilePaths[noOfTestFiles: ], datasetTrainClassDirectory)

        datasetTestClassDirectory = os.path.join(datasetTestDirectory, classDirectory)
        createDirectory(datasetTestClassDirectory)
        copyFiles(inputTrainClassFilePaths[ :noOfTestFiles], datasetTestClassDirectory)


def recreateCrossValidationDataDefault():
    # Get default values
    datasetTrainDirectory = readConfigFile("DIRECTORY", "datasetTrain")
    datasetValidateDirectory = readConfigFile("DIRECTORY", "datasetValidate")
    datasetValidateFoldTrainDirectory = readConfigFile("DIRECTORY", "datasetValidateFoldTrain")
    datasetValidateFoldValidateDirectory = readConfigFile("DIRECTORY", "datasetValidateFoldValidate")
    noOfFolds = getNoOfFolds()
    shuffleFiles = useShuffledFiles()
    batchSize = getBatchSize()    
    
    # Run routine
    recreateCrossValidationData(datasetTrainDirectory, datasetValidateDirectory, datasetValidateFoldTrainDirectory, datasetValidateFoldValidateDirectory, noOfFolds, shuffleFiles, batchSize)


def recreateCrossValidationData(datasetTrainDirectory, datasetValidateDirectory, datasetValidateFoldTrainDirectory, datasetValidateFoldValidateDirectory, noOfFolds, shuffleFiles, batchSize):
    deleteCrossValidationData(datasetValidateDirectory)
    createCrossValidationData(datasetTrainDirectory, datasetValidateFoldTrainDirectory, datasetValidateFoldValidateDirectory, noOfFolds, shuffleFiles, batchSize)
    

def deleteCrossValidationDataDefault():
    # Get default values
    datasetValidateDirectory = readConfigFile("DIRECTORY", "datasetValidate")
    
    # Run routine
    deleteCrossValidationData(datasetValidateDirectory)


def deleteCrossValidationData(datasetValidateDirectory):
    deleteSubdirectories(datasetValidateDirectory)


def createCrossValidationDataDefault():
    # Get deault values
    datasetTrainDirectory = readConfigFile("DIRECTORY", "datasetTrain")
    datasetValidateFoldTrainDirectory = readConfigFile("DIRECTORY", "datasetValidateFoldTrain")
    datasetValidateFoldValidateDirectory = readConfigFile("DIRECTORY", "datasetValidateFoldValidate")
    noOfFolds = getNoOfFolds()
    shuffleFiles = useShuffledFiles()
    batchSize = getBatchSize()    
    
    # Run routine
    createCrossValidationData(datasetTrainDirectory, datasetValidateFoldTrainDirectory, datasetValidateFoldValidateDirectory, noOfFolds, shuffleFiles, batchSize)


def createCrossValidationData(datasetTrainDirectory, datasetValidateFoldTrainDirectory, datasetValidateFoldValidateDirectory, noOfFolds, shuffleFiles, batchSize):
    datasetValidateFoldTrainDirectoryTemplate = datasetValidateFoldTrainDirectory
    datasetValidateFoldValidateDirectoryTemplate = datasetValidateFoldValidateDirectory
    classDirectories = getDirectories(datasetTrainDirectory)
    for classDirectory in classDirectories:
        datasetTrainClassDirectory = os.path.join(datasetTrainDirectory, classDirectory)
        datasetTrainClassFilePaths = getFiles(datasetTrainClassDirectory, True)
        if shuffleFiles: 
            random.shuffle(datasetTrainClassFilePaths)

        for i in range(noOfFolds):
            datasetValidateFoldValidateDirectory = datasetValidateFoldValidateDirectoryTemplate.replace("$", str(i))
            datasetValidateFoldValidateClassDirectory = os.path.join(datasetValidateFoldValidateDirectory, classDirectory)
            createDirectory(datasetValidateFoldValidateClassDirectory)

            noOfClassFiles = round(len(datasetTrainClassFilePaths) / noOfFolds)
            datasetValidateFoldValidatenClassFilePaths = datasetTrainClassFilePaths[(i * noOfClassFiles):((i+1) * noOfClassFiles)]
            copyFiles(datasetValidateFoldValidatenClassFilePaths, datasetValidateFoldValidateClassDirectory)
  
            datasetValidateFoldTrainDirectory = datasetValidateFoldTrainDirectoryTemplate.replace("$", str(i))
            datasetValidateFoldTrainClassDirectory = os.path.join(datasetValidateFoldTrainDirectory, classDirectory)
            createDirectory(datasetValidateFoldTrainClassDirectory)

            if i != 0:
                datasetValidateFoldTrainClassFilePaths1 = datasetTrainClassFilePaths[ :(i * noOfClassFiles)]
            else:
                datasetValidateFoldTrainClassFilePaths1 = []
            if i != noOfFolds-1:
                datasetValidateFoldTrainClassFilePaths2 = datasetTrainClassFilePaths[((i+1) * noOfClassFiles): ]
            else:
                datasetValidateFoldTrainClassFilePaths2 = []
            datasetValidateFoldTrainClassFilePaths = datasetValidateFoldTrainClassFilePaths1 + datasetValidateFoldTrainClassFilePaths2
            copyFiles(datasetValidateFoldTrainClassFilePaths, datasetValidateFoldTrainClassDirectory)


def recreatePredictDataDefault():
    deletePredictDataDefault()
    createPredictDataDefault()


def deletePredictDataDefault():
    # Get default values
    datasetPredictDirectory = readConfigFile("DIRECTORY", "datasetPredict")
    
    # Run routine
    deletePredictData(datasetPredictDirectory)    


def deletePredictData(datasetPredictDirectory):
    deleteSubdirectories(datasetPredictDirectory)    


def createPredictDataDefault():
    # Get default values
    inputPredictDirectory = readConfigFile("DIRECTORY", "inputPredict")
    datasetPredictDirectory = readConfigFile("DIRECTORY", "datasetPredict")
    datasetPredictSubsetDirectory = readConfigFile("DIRECTORY", "datasetPredictSubset")
    maxFiles = getMaxFiles()

    # Run routine
    createPredictData(inputPredictDirectory, datasetPredictDirectory, datasetPredictSubsetDirectory, maxFiles)   


def createPredictData(inputPredictDirectory, datasetPredictDirectory, datasetPredictSubsetDirectory, maxFiles):
    datasetPredictSubsetTemplateDirectory = datasetPredictSubsetDirectory
    classDirectories = getDirectories(inputPredictDirectory)
    for classDirectory in classDirectories:
        inputPredictClassDirectory = os.path.join(inputPredictDirectory, classDirectory)
        filePaths = getFiles(inputPredictClassDirectory, True)
        noOfFiles = len(filePaths)
        noOfSubsets = math.ceil(noOfFiles / maxFiles)
        for i in range(noOfSubsets):
            datasetPredictSubsetDirectory = datasetPredictSubsetTemplateDirectory.replace("$", str(i).zfill(3))
            datasetPredictSubsetDirectory = os.path.join(datasetPredictSubsetDirectory, classDirectory)
            createDirectory(datasetPredictSubsetDirectory)
            copyFiles(filePaths[(i * maxFiles):((i+1) * maxFiles)], datasetPredictSubsetDirectory)