# Load required modules
import math
import os
import random
import shutil

# Load required scripts
from scripts.config import readConfigFile, getNoOfTestFiles, getPercentageOfValidationFiles, useShuffledFiles, getBatchSize, getMaxFiles
from scripts.file import getFiles, copyFiles, moveFiles, getDirectories, createDirectory, deleteSubdirectories


def recreateDatasetDefault():
    recreateTrainAndTestDataDefault()
    recreateValidationDataDefault()
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


def recreateValidationDataDefault():
    # Get default values
    datasetTrainDirectory = readConfigFile("DIRECTORY", "datasetTrain")
    datasetValidateDirectory = readConfigFile("DIRECTORY", "datasetValidate")
    percentageOfValidationFiles = getPercentageOfValidationFiles()
    shuffleFiles = useShuffledFiles()
    batchSize = getBatchSize()    
    
    # Run routine
    recreateValidationData(datasetTrainDirectory, datasetValidateDirectory, percentageOfValidationFiles, shuffleFiles, batchSize)


def recreateValidationData(datasetTrainDirectory, datasetValidateDirectory, percentageOfValidationFiles, shuffleFiles, batchSize):
    deleteValidationData(datasetValidateDirectory)
    createValidationData(datasetTrainDirectory, datasetValidateDirectory, percentageOfValidationFiles, shuffleFiles, batchSize)
    

def deleteValidationDataDefault():
    # Get default values
    datasetValidateDirectory = readConfigFile("DIRECTORY", "datasetValidate")
    
    # Run routine
    deleteValidationData(datasetValidateDirectory)


def deleteValidationData(datasetValidateDirectory):
    deleteSubdirectories(datasetValidateDirectory)


def createValidationDataDefault():
    # Get deault values
    datasetTrainDirectory = readConfigFile("DIRECTORY", "datasetTrain")
    datasetValidateDirectory = readConfigFile("DIRECTORY", "datasetValidate")
    percentageOfValidationFiles = getPercentageOfValidationFiles()
    shuffleFiles = useShuffledFiles()
    batchSize = getBatchSize()    
    
    # Run routine
    createValidationData(datasetTrainDirectory, datasetValidateDirectory, percentageOfValidationFiles, shuffleFiles, batchSize)


def createValidationData(datasetTrainDirectory, datasetValidateDirectory, percentageOfValidationFiles, shuffleFiles, batchSize):
    classDirectories = getDirectories(datasetTrainDirectory)
    for classDirectory in classDirectories:
        datasetTrainClassDirectory = os.path.join(datasetTrainDirectory, classDirectory)
        datasetTrainClassFilePaths = getFiles(datasetTrainClassDirectory, True)
        if shuffleFiles: 
            random.shuffle(datasetTrainClassFilePaths)

        noOfValidationFiles = int(len(datasetTrainClassFilePaths) / 100 * percentageOfValidationFiles)
        datasetTrainClassFilePaths = datasetTrainClassFilePaths[ :noOfValidationFiles]
        datasetValidationClassDirectory = os.path.join(datasetValidateDirectory, classDirectory)    
        createDirectory(datasetValidationClassDirectory)           
        moveFiles(datasetTrainClassFilePaths, datasetValidationClassDirectory)


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
