# Load required modules
import os
import shutil


def getFiles(rootDirectory, fullPath=False, extension=".png"):
    filePaths = []
    for contentName in os.listdir(rootDirectory):
        contentPath = os.path.join(rootDirectory, contentName)
        if os.path.isfile(contentPath):
             if contentName.endswith(extension):
                if fullPath:
                    filePaths.append(contentPath)
                else:
                    filePaths.append(contentName)
    return sorted(filePaths)


def getFilesIncludingSubdirectories(rootDirectory, fullPath=False, extension=".png"):
    filePaths = list()
    for (directoryPaths, directoryNames, fileNames) in os.walk(rootDirectory):
        for fileName in fileNames:
            if fileName.endswith(extension):
                if fullPath:
                    filePaths.append(os.path.join(directoryPaths, fileName))
                else:
                    filePaths.append(fileName)
    return sorted(filePaths)


def copyFiles(sourceFilePaths, destinationDirectory):
    for sourceFilePath in sourceFilePaths:
        shutil.copy(sourceFilePath, destinationDirectory)    


def moveFiles(sourceFilePaths, destinationDirectory):
    for sourceFilePath in sourceFilePaths:
        shutil.move(sourceFilePath, destinationDirectory)    


def getDirectories(rootDirectory, fullPath=False):
    directories = []
    for contentName in os.listdir(rootDirectory):
        contentPath = os.path.join(rootDirectory, contentName)
        if os.path.isdir(contentPath):
            if fullPath:
                directories.append(contentPath)
            else:
                directories.append(contentName)
    return sorted(directories)


def createDirectory(directory):
    os.makedirs(directory, exist_ok=True)


def deleteDirectory(directory, ignoreErrors=False):
    if os.path.isdir(directory):
        shutil.rmtree(directory, ignore_errors=ignoreErrors)           
    else:
        raise Exception("Invalid directory")


def deleteDirectories(directories, ignoreErrors=False):
    for directory in directories: 
        deleteDirectory(directory, ignoreErrors)           


def deleteSubdirectories(rootDirectory, ignoreErrors=False):
    subdirectories = getDirectories(rootDirectory, True)
    deleteDirectories(subdirectories, ignoreErrors)
