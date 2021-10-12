# Load required packages
import os
import shutil


def getFiles(rootDirectory, fullPath=False, extension=".png"):
    """getFiles
        Arguments:
            rootDirectory: Search directory
            fullPath: Return full path of the files
            extension: File extension
        Return:
            List of files
    """
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
    """getFiles
        Arguments:
            rootDirectory: Search directory
            fullPath: Return full path of the files
            extension: File extension
        Return:
            List of files
    """    
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
    """copyFiles
        Arguments:
            sourceFilePaths: List of files to copy
            directory: Destination directory
    """         
    for sourceFilePath in sourceFilePaths:
        shutil.copy(sourceFilePath, destinationDirectory)    


def moveFiles(sourceFilePaths, destinationDirectory):
    """moveFiles
        Arguments:
            sourceFilePaths: List of files to move
            directory: Destination directory
    """         
    for sourceFilePath in sourceFilePaths:
        shutil.move(sourceFilePath, destinationDirectory)    


def getDirectories(rootDirectory, fullPath=False):
    """getDirectories
        Arguments:
            rootDirectory: Search directory
            fullPath: Return full path of the directories
        Return:
            List of directories
    """
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
    """createDirectory
        Arguments:
            directory: Directory path to create
    """    
    os.makedirs(directory, exist_ok=True)


def deleteDirectory(directory, ignoreErrors=False):
    """deleteDirectory
        Arguments:
            directory: Directory to delete
            ignoreErrors: Ignore errors
    """
    if os.path.isdir(directory):
        shutil.rmtree(directory, ignore_errors=ignoreErrors)           
    else:
        raise Exception("Invalid directory")


def deleteDirectories(directories, ignoreErrors=False):
    """deleteDirectory
        Arguments:
            directories: List of diretories to delete
            ignoreErrors: Ignore errors
    """       
    for directory in directories: 
        deleteDirectory(directory, ignoreErrors)           


def deleteSubdirectories(rootDirectory, ignoreErrors=False):
    """deleteSubdirectories
        Arguments:
            directory: Directory to delete subdirectories from
            ignoreErrors: Ignore errors
    """        
    subdirectories = getDirectories(rootDirectory, True)
    deleteDirectories(subdirectories, ignoreErrors)
