# Load required packages
from numpy import array, mean
import matplotlib.pyplot as plt
import os
import pandas
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.optimizers import Adam	
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import plot_model


# Load required py-scripts
from scripts.config import readConfigFile, getWidth, getHeight, getDepth, getBatchSize
from scripts.file import getFiles, getFilesIncludingSubdirectories, getDirectories, createDirectory, deleteSubdirectories
from scripts.hist import histPrediction, histPredictionTest


def getModelBaseDefault():
    """getModelBaseDefault
        Return:
            Resnet50 model base           
    """
    width = getWidth()
    height = getHeight()
    depth = getDepth()
    return getModelBase(width, height, depth)


def getModelBase(width, height, depth):
    """getModelBase
        Arguments:
            width: Image width
            height: Image height
            depth: Color depth
        Return:
            Resnet50 model base
    """
    modelBase = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=[width, height, depth]
    )
    return modelBase


def trainModelDefault():
    """trainModelDefault

    """
    # Get default values
    datasetTrainDirectory = readConfigFile("DIRECTORY", "datasetTrain")
    datasetValidateDirectory = readConfigFile("DIRECTORY", "datasetValidate")
    outputTrainDirectory = readConfigFile("DIRECTORY", "outputTrain")
    outputValidateDirectory = readConfigFile("DIRECTORY", "outputValidate")
    width = getWidth()
    height = getHeight()
    depth = getDepth()
    batchSize = getBatchSize()

    # Run routine
    return trainModel(datasetTrainDirectory, datasetValidateDirectory, outputTrainDirectory, outputValidateDirectory, width, height, depth, batchSize)


def trainModel(datasetTrainDirectory, datasetValidateDirectory, outputTrainDirectory, outputValidateDirectory, width, height, depth, batchSize):
    """trainModel
        Arguments:
            datasetTrainDirectory: Dataset train directory
            datasetValidateDirectory: Datasete validate directory
            outputTrainDirectory: Output train directory
            outputValidateDirectory: Output validate directory
            width: Image width
            height: Image height
            depth: Color depth
            batchSize: Batch size
    """ 
    # define the ImageNet mean subtraction (in RGB order) and set the
    # the mean subtraction value for each of the data augmentation objects
    mean = array([123.68, 116.779, 103.939], dtype="float32")

    # Process train files
    trainDataGenerator = ImageDataGenerator(
        preprocessing_function = preprocess_input,
        rotation_range=5,
        zoom_range=[0.95, 1],
        horizontal_flip=True, 
        # vertical_flip=True,
        width_shift_range=0.05,
        height_shift_range=0.1,
        brightness_range=[0.7, 1.0],
        fill_mode="constant",
        cval=75
    )

    trainDataGenerator.mean = mean
    trainData = trainDataGenerator.flow_from_directory(
        datasetTrainDirectory,
        batch_size = batchSize,
        color_mode = "rgb",
        class_mode = "binary",
        shuffle = True,        
        target_size = (width, height)
    )

    # Process validate files
    validateDataGenerator = ImageDataGenerator(
        preprocessing_function = preprocess_input
    )

    validateDataGenerator.mean = mean
    validateData = validateDataGenerator.flow_from_directory(
        datasetValidateDirectory,
        batch_size = batchSize,
        class_mode = "binary",
        color_mode = "rgb",
        shuffle = True,
        target_size = (width, height)
    )

    # Define steps per epoch
    epochs = 100
    trainStepsPerEpoch = int(trainData.n / batchSize)
    validateStepsPerEpcoh = int(validateData.n / batchSize)     

    # Build model
    model, history = buildModel(width, height, depth, trainData, epochs, trainStepsPerEpoch, validateData, validateStepsPerEpcoh)

    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validate'], loc='upper left')
    plt.savefig(os.path.join(outputValidateDirectory, 'model_accuracy.pdf'))
    plt.close()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validate'], loc='upper left')
    plt.savefig(os.path.join(outputValidateDirectory, 'model_loss.pdf'))
    plt.close()
    
    # Save model
    # filePath = os.path.join(outputTrainDirectory, "model.h5")
    # model.save(filePath)

    return model


def buildModel(width, height, depth, trainData, epochs, trainStepsPerEpoch, validateData = None, validateStepsPerEpoch = None):
    """buildModel
        Arguments: 
            width: Image width
            height: Image height
            depth: Image depth
            batchSize: Batch size
            trainData: Training data
            trainStepsPerEpoch: Train steps per epoch
            validationData: Optional validation data
            validateStepsPerEpoch: Optional validation steps per epoch
        Return:
            Built model
    """
    # Get resnet50 model base
    modelBase = getModelBase(width, height, depth)

    # Freeze layers of base model which wont get updated during training
    freezeLayers = 143
    for layer in modelBase.layers[:freezeLayers]:
        layer.trainable = False
    for layer in modelBase.layers[freezeLayers:]:
        layer.trainable = True

    # Construct new head
    modelHead = modelBase.output
    modelHead = MaxPooling2D(pool_size=(2, 2))(modelHead)
    modelHead = Flatten(name="flatten")(modelHead)
    modelHead = Dense(256, activation="relu")(modelHead)
    modelHead = Dropout(0.2)(modelHead)
    modelHead = Dense(256, activation="relu")(modelHead)
    modelHead = Dropout(0.2)(modelHead)
    modelHead = Dense(1, activation="sigmoid")(modelHead)

    # Place head on base model
    model = Model(inputs=modelBase.input, outputs=modelHead)

    # Compile the model
    model.compile(
        optimizer = Adam(lr=0.00001),
        loss = "binary_crossentropy",
        metrics = ["accuracy"],
    )

    # Train the model with early stopping
    earlyStoppingCallback = EarlyStopping(monitor='val_loss', patience=20)
    checkpointCallback = ModelCheckpoint('output/train/model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    history = model.fit_generator(
        trainData,
        steps_per_epoch = trainStepsPerEpoch,
        epochs = epochs,
        validation_data = validateData,
        validation_steps = validateStepsPerEpoch,
        callbacks=[earlyStoppingCallback, checkpointCallback]
    )

    return model, history


def loadModelDefault():
    """loadModelDefault
        Return:
            Keras model object
    """
    # Get default values
    outputTrainDirectory = readConfigFile("DIRECTORY", "outputTrain")

    # Run routine
    return loadModel(outputTrainDirectory)


def loadModel(outputTrainDirectory, fileName="model.h5"):
    """loadModel
        Arguments:
            dsEntry: datasetEntry object
            fileNanme: default file name
        Return:
            Keras model object
    """
    filePath = os.path.join(outputTrainDirectory, fileName)
    if os.path.exists(filePath):
        model = load_model(filePath)
        return model
    else:
        raise Exception("No saved model found.")


def predictFilesDefault(model):
    """predictFiles
        Arguments:
            model: Trained model to use
        Return:
            DataFrame of all predictions
    """    
    # Get default values
    datasetPredictDirectory = readConfigFile("DIRECTORY", "datasetPredict")
    outputPredictDirectory = readConfigFile("DIRECTORY", "outputPredict")
    width = getWidth()
    height = getHeight()
    depth = getDepth()  

    # Run routine 
    dict = {
        'name':[],
        'lat':[],
        'lon':[],
        'percentage':[],
        'filePath':[]
    }

    predictions = pandas.DataFrame(dict)
    deleteSubdirectories(outputPredictDirectory)
    subsetDirectories = getDirectories(datasetPredictDirectory)
    for subsetDirectory in subsetDirectories:
        datasetPredictSubsetDirectory = os.path.join(datasetPredictDirectory, subsetDirectory)
        outputPredictSubsetDirectory = os.path.join(outputPredictDirectory, subsetDirectory)
        createDirectory(outputPredictSubsetDirectory)

        predictionSubset = predictFiles(model, datasetPredictSubsetDirectory, width, height, depth)
        predictionSubset.to_csv(path_or_buf=os.path.join(outputPredictSubsetDirectory, "prediction.csv"), index=False, sep=";")
        histPrediction(predictionSubset, os.path.join(outputPredictSubsetDirectory, "histogram.pdf"))
        predictions = pandas.concat([predictions, predictionSubset])

    predictions.to_csv(path_or_buf=os.path.join(outputPredictDirectory, "prediction.csv"), index=False, sep=";")
    histPrediction(predictions, os.path.join(outputPredictDirectory, "histogram.pdf"))
        

def predictTestFilesDefault(model):
    """predictTestFilesDefault
        Arguments:
            model: Trained model to use
        Return:
            DataFrame of all predictions
    """    
    # Get default values
    datasetTestDirectory = readConfigFile("DIRECTORY", "datasetTest")
    outputTestDirectory = readConfigFile("DIRECTORY", "outputTest")
    width = getWidth()
    height = getHeight()
    depth = getDepth()

    # Run routine and save output
    predictions = predictFiles(model, datasetTestDirectory, width, height, depth)
    predictions.to_csv(path_or_buf=os.path.join(outputTestDirectory, "prediction.csv"), index=False, sep=";")
    histPredictionTest(predictions, os.path.join(outputTestDirectory, "histogram.pdf"))


def predictFiles(model, datasetPredictDirectory, width, height, depth):
    """predictFiles
        Arguments:
            model: Trained model to use
            datasetPredictDirectory: Path to directory, containing the image files to predict
            width: Image width
            height: Image height
            depth: Color depth
        Return:
            DataFrame of all predictions
    """
    mean = array([123.68, 116.779, 103.939], dtype="float32")

    # initialize the testing generator
    predictDataGenerator = ImageDataGenerator()
    predictDataGenerator.mean = mean

    predictData = predictDataGenerator.flow_from_directory(
        datasetPredictDirectory,
        batch_size=10,
        class_mode="binary",
        color_mode="rgb",
        shuffle=False,
        target_size=(width, height)
    )
    
    # Print data    
    predictData.reset()
    predictions = model.predict_generator(predictData, steps=(predictData.n // 10))   

    # Prepare results
    names = list()
    lats = list()
    lons = list()
    percentages = list()
    filePaths = list()

    index = 0  
    for fileName in predictData.filenames:
        try:
            # Get indices
            startIndex = fileName.index('[')
            endIndex = fileName.index(']')

            # Extract name
            name = fileName[0:startIndex-1]
            name = name.rstrip()
            name = name.lstrip()

            # Extract lat & lon
            coordinates = fileName[startIndex+1:endIndex]
            startIndex = coordinates.index(',')
            lat = coordinates[0:startIndex]
            lat = lat.rstrip()
            lat = lat.lstrip()
            lat = float(lat)
            lon = coordinates[startIndex+1:]
            lon = lon.rstrip()
            lon = lon.lstrip()
            lon = float(lon)        

            # Get percentage
            prct = (predictions[index,0] * 100)
        except:
            name = 'Unknown'
            lat = 0.00
            lon = 0.00
            prct = 0            
        finally:
            names.append(name)
            lats.append(lat)
            lons.append(lon)
            percentages.append(prct)
            filePaths.append(os.path.join(datasetPredictDirectory, fileName))
            index += 1

    dict = {
        'name':names,
        'lat':lats,
        'lon':lons,
        'percentage':percentages,
        'filePath':filePaths
    }

    df = pandas.DataFrame(dict)
    return df

