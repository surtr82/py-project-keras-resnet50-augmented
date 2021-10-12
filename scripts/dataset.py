# Load required packages
import math
import os
import random
import shutil


# Load required py-scripts
from scripts.config import readConfigFile, getNoOfTestFiles, getPercentageOfValidationFiles, useShuffledFiles, getBatchSize, getMaxFiles
from scripts.file import getFiles, getFilesIncludingSubdirectories, copyFiles, moveFiles, getDirectories, createDirectory, deleteSubdirectories


def recreateDatasetDefault():
    """recreateDatasetDefault

    """ 
    recreateTrainAndTestDataDefault()
    recreateValidationDataDefault()
    recreatePredictDataDefault()


def recreateTrainAndTestDataDefault():
    """createTrainAndTestDataDefault

    """      
    # Get default values
    inputTrainDirectory = readConfigFile("DIRECTORY", "inputTrain")
    datasetTrainDirectory = readConfigFile("DIRECTORY", "datasetTrain")
    datasetTestDirectory =  readConfigFile("DIRECTORY", "datasetTest")
    noOfTestFiles = getNoOfTestFiles() 
    shuffleFiles = useShuffledFiles()

    # Run routine
    recreateTrainAndTestData(inputTrainDirectory, datasetTrainDirectory, datasetTestDirectory, noOfTestFiles, shuffleFiles)


def recreateTrainAndTestData(inputTrainDirectory, datasetTrainDirectory, datasetTestDirectory, noOfTestFiles, shuffleFiles):
    """createTrainAndTestData
        Arguments:
            inputTrainDirectory: Input train directory
            datasetTrainDirectory: Dataset train directory
            datasetTestDirectory: Dataset test directory
            noOfTestFiles: Number of test files to separate
            shuffleFiles: Shuffle files before splitting test files
    """      
    deleteTrainAndTestData(datasetTrainDirectory, datasetTestDirectory)
    createTrainAndTestData(inputTrainDirectory, datasetTrainDirectory, datasetTestDirectory, noOfTestFiles, shuffleFiles)


def deleteTrainAndTestDataDefault():
    """deleteTrainAndTestDataDefault

    """   
    # Get default values
    datasetTrainDirectory = readConfigFile("DIRECTORY", "datasetTrain")
    datasetTestDirectory =  readConfigFile("DIRECTORY", "datasetTest")    

    # Run routine
    deleteTrainAndTestData(datasetTrainDirectory, datasetTestDirectory)


def deleteTrainAndTestData(datasetTrainDirectory, datasetTestDirectory):
    """deleteTrainAndTestData
        Arguments:
            datasetTrainDirectory: Dataset train directory
            datasetTestDirectory: Dataset test directory        
    """      
    deleteSubdirectories(datasetTrainDirectory)
    deleteSubdirectories(datasetTestDirectory)


def createTrainAndTestDataDefault():
    """createTrainAndTestDataDefault

    """      
    # Get default values
    inputTrainDirectory = readConfigFile("DIRECTORY", "inputTrain")
    datasetTrainDirectory = readConfigFile("DIRECTORY", "datasetTrain")
    datasetTestDirectory =  readConfigFile("DIRECTORY", "datasetTest")
    noOfTestFiles = getNoOfTestFiles() 
    shuffleFiles = useShuffledFiles()

    # Run routine
    createTrainAndTestData(inputTrainDirectory, datasetTrainDirectory, datasetTestDirectory, noOfTestFiles, shuffleFiles)


def createTrainAndTestData(inputTrainDirectory, datasetTrainDirectory, datasetTestDirectory, noOfTestFiles, shuffleFiles):
    """createTrainAndTestData
        Arguments:
            inputTrainDirectory: Input train directory
            datasetTrainDirectory: Dataset train directory
            datasetTestDirectory: Dataset test directory
            noOfTestFiles: Number of test files to separate
            shuffleFiles: Shuffle files before splitting test files
    """    
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
    """recreateValidationDataDefault

    """     
    # Get default values
    datasetTrainDirectory = readConfigFile("DIRECTORY", "datasetTrain")
    datasetValidateDirectory = readConfigFile("DIRECTORY", "datasetValidate")
    percentageOfValidationFiles = getPercentageOfValidationFiles()
    shuffleFiles = useShuffledFiles()
    batchSize = getBatchSize()    
    
    # Run routine
    recreateValidationData(datasetTrainDirectory, datasetValidateDirectory, percentageOfValidationFiles, shuffleFiles, batchSize)


def recreateValidationData(datasetTrainDirectory, datasetValidateDirectory, percentageOfValidationFiles, shuffleFiles, batchSize):
    """recreateValidationData
        Arguments:
            datasetTrainDirectory: Dataset train directory
            datasetValidateDirectory: Dataset validate directory
            percentageOfValidationFiles: Percentage of training files used for validation
            shuffleFiles: Shuffle files for cross validation
            batchSize: Round number of files in each folder according to the later batch size
    """  
    deleteValidationData(datasetValidateDirectory)
    createValidationData(datasetTrainDirectory, datasetValidateDirectory, percentageOfValidationFiles, shuffleFiles, batchSize)
    

def deleteValidationDataDefault():
    """deleteValidationDataDefault

    """   
    # Get default values
    datasetValidateDirectory = readConfigFile("DIRECTORY", "datasetValidate")
    
    # Run routine
    deleteValidationData(datasetValidateDirectory)


def deleteValidationData(datasetValidateDirectory):
    """deleteCrossValidationDataDefault
        Arguments:
            datasetValidateDirectory: Dataset Validate Directory 
    """      
    deleteSubdirectories(datasetValidateDirectory)


def createValidationDataDefault():
    """createValidationDataDefault

    """    
    # Get deault values
    datasetTrainDirectory = readConfigFile("DIRECTORY", "datasetTrain")
    datasetValidateDirectory = readConfigFile("DIRECTORY", "datasetValidate")
    percentageOfValidationFiles = getPercentageOfValidationFiles()
    shuffleFiles = useShuffledFiles()
    batchSize = getBatchSize()    
    
    # Run routine
    createValidationData(datasetTrainDirectory, datasetValidateDirectory, percentageOfValidationFiles, shuffleFiles, batchSize)


def createValidationData(datasetTrainDirectory, datasetValidateDirectory, percentageOfValidationFiles, shuffleFiles, batchSize):
    """createValidationData
        Arguments:
            datasetTrainDirectory: Dataset train directory
            datasetValidateDirectory: Dataset validate directory    
            percentageOfValidationFiles: Percentage of training files used for validation       
            shuffleFiles: Shuffle files for cross validation
            batchSize: Round number of files in each folder according to the later batch size
    """ 
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
    """recreatePredictDataDefault

    """   
    deletePredictDataDefault()
    createPredictDataDefault()


def deletePredictDataDefault():
    """deletePredictDataDefault

    """   
    # Get default values
    datasetPredictDirectory = readConfigFile("DIRECTORY", "datasetPredict")
    
    # Run routine
    deletePredictData(datasetPredictDirectory)    


def deletePredictData(datasetPredictDirectory):
    """deletePredictData
        Arguments:
            datasetPredictDirectory: Dataset Predict Directory 
    """      
    deleteSubdirectories(datasetPredictDirectory)    


def createPredictDataDefault():
    """createPredictDataDefault

    """  
    # Get default values
    inputPredictDirectory = readConfigFile("DIRECTORY", "inputPredict")
    datasetPredictDirectory = readConfigFile("DIRECTORY", "datasetPredict")
    datasetPredictSubsetDirectory = readConfigFile("DIRECTORY", "datasetPredictSubset")
    maxFiles = getMaxFiles()

    # Run routine
    createPredictData(inputPredictDirectory, datasetPredictDirectory, datasetPredictSubsetDirectory, maxFiles)   


def createPredictData(inputPredictDirectory, datasetPredictDirectory, datasetPredictSubsetDirectory, maxFiles):
    """createPredictData
        Arguments:
            inputPredictDirectory: Input predict directory
            datasetPredictDirectory: Dataset predict directory
            datasetPredictSubsetDirectory: Dataset predict subset directory
            maxFiles: Maximum amount of files to predict            

    """      
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