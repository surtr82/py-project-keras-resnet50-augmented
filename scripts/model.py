# Load required modules
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

# Load required scripts
from scripts.config import readConfigFile, getWidth, getHeight, getDepth, getBatchSize, getNoOfFolds
from scripts.file import getDirectories, createDirectory, deleteSubdirectories
from scripts.hist import histPrediction, histPredictionTest


def getModelBaseDefault():
    width = getWidth()
    height = getHeight()
    depth = getDepth()
    return getModelBase(width, height, depth)


def getModelBase(width, height, depth):
    modelBase = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=[width, height, depth]
    )
    return modelBase


def executeCrossValidationDefault():
    # Get default values
    datasetValidateFoldTrainDirectory = readConfigFile("DIRECTORY", "datasetValidateFoldTrain")
    datasetValidateFoldValidateDirectory = readConfigFile("DIRECTORY", "datasetValidateFoldValidate")
    outputValidateDirectory = readConfigFile("DIRECTORY", "outputValidate")
    noOfFolds = getNoOfFolds()
    width = getWidth()
    height = getHeight()
    depth = getDepth()
    batchSize = getBatchSize()

    # Run routine
    executeCrossValidation(datasetValidateFoldTrainDirectory, datasetValidateFoldValidateDirectory, outputValidateDirectory, noOfFolds, width, height, depth, batchSize)


def executeCrossValidation(datasetValidateFoldTrainDirectory, datasetValidateFoldValidateDirectory, outputValidateDirectory, noOfFolds, width, height, depth, batchSize):
    # Terminal output
    print("")
    print("=== Cross Validation ===")

    # Init plot results  
    historiesAccuracy = []
    historiesValidationAccuracy = []
    historiesLoss = []
    historiesValidationLoss = []

    # define the ImageNet mean subtraction (in RGB order) and set the
    # the mean subtraction value for each of the data augmentation objects
    mean = array([123.68, 116.779, 103.939], dtype="float32")

    for i in range(noOfFolds):
        print("Fold No. %s" % str(i))

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
        trainDirectory = datasetValidateFoldTrainDirectory.replace("$", str(i))
        trainData = trainDataGenerator.flow_from_directory(
            trainDirectory,
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
        validateDirectory = datasetValidateFoldValidateDirectory.replace("$", str(i))
        validateData = validateDataGenerator.flow_from_directory(
            validateDirectory,
            batch_size = batchSize,
            class_mode = "binary",
            color_mode = "rgb",
            shuffle = True,
            target_size = (width, height)
        )

        # Define steps per epoch
        epochs = 30
        trainStepsPerEpoch = int(trainData.n / batchSize)   
        validateStepsPerEpcoh = int(validateData.n / batchSize)     

        # Build model
        model, history = buildModel(width, height, depth, trainData, epochs, trainStepsPerEpoch, validateData, validateStepsPerEpcoh)


        # Plot dictory
        outputValidateFoldDirectory = os.path.join(outputValidateDirectory, "fold$".replace("$", str(i)))
        createDirectory(outputValidateFoldDirectory)

        # Plot training & validation accuracy values
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validate'], loc='upper left')
        plt.savefig(os.path.join(outputValidateFoldDirectory, 'model_accuracy.pdf'))
        plt.close()

        # Plot training & validation loss values
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validate'], loc='upper left')
        plt.savefig(os.path.join(outputValidateFoldDirectory, 'model_loss.pdf'))
        plt.close()

        # Save model
        filePath = os.path.join(outputValidateFoldDirectory, "model.h5")
        model.save(filePath)

        # Save measurments for mean calculation
        historyAccuracy = history.history['accuracy']
        historiesAccuracy.append(historyAccuracy)
        historyValidationAccuracy = history.history['val_accuracy']        
        historiesValidationAccuracy.append(historyValidationAccuracy)
        historyLoss = history.history['loss']
        historiesLoss.append(historyLoss)
        historyValidationLoss = history.history['val_loss']
        historiesValidationLoss.append(historyValidationLoss)

    # Plot measurement means
    columns = [str(i) for i in range(1, len(historyAccuracy)+1)]
    historiesAccuracyDataFrame = pandas.DataFrame(historiesAccuracy, columns=columns)
    historiesValidationAccuracyDataFrame = pandas.DataFrame(historiesValidationAccuracy, columns=columns)
    historiesLossDataFrame = pandas.DataFrame(historiesLoss, columns=columns)
    historiesValidationLossDataFrame = pandas.DataFrame(historiesValidationLoss, columns=columns)

    try:
        historiesAccuracyDataFrame.to_csv(os.path.join(outputValidateDirectory, 'train_accuracy.csv'))
        historiesValidationAccuracyDataFrame.to_csv(os.path.join(outputValidateDirectory, 'validation_accuracy.csv'))
        historiesLossDataFrame.to_csv(os.path.join(outputValidateDirectory, 'train_loss.csv'))    
        historiesValidationLossDataFrame.to_csv(os.path.join(outputValidateDirectory, 'validation_loss.csv')) 
    except:
        print("Save of accuracy and loss files failed.")

    # Plot training & validation accuracy mean values
    plt.plot(historiesAccuracyDataFrame.mean())
    plt.plot(historiesValidationAccuracyDataFrame.mean())
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validate'], loc='upper left')
    plt.savefig(os.path.join(outputValidateDirectory, 'model_accuracy.pdf'))
    plt.close()

    # Plot training & validation loss mean values
    plt.plot(historiesLossDataFrame.mean())
    plt.plot(historiesValidationLossDataFrame.mean())
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validate'], loc='upper left')
    plt.savefig(os.path.join(outputValidateDirectory, 'model_loss.pdf'))
    plt.close()


def trainFinalModelDefault():
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
    return trainFinalModel(datasetTrainDirectory, datasetValidateDirectory, outputTrainDirectory, outputValidateDirectory, width, height, depth, batchSize)


def trainFinalModel(datasetTrainDirectory, datasetValidateDirectory, outputTrainDirectory, outputValidateDirectory, width, height, depth, batchSize):
    # Terminal output
    print("")
    print("=== Train final model ===")

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

    # Define steps per epoch
    epochs = 30
    trainStepsPerEpoch = int(trainData.n / batchSize)
                
    # Build model
    model, history = buildModel(width, height, depth, trainData, epochs, trainStepsPerEpoch)

    # Save history
    try:
        history_acc = pandas.DataFrame(history.history) 
        history_acc.to_csv(os.path.join(outputValidateDirectory, 'model_accuracy.csv'))
    except:
        print("model_accuracy.csv save failed.")

    # Save model
    filePath = os.path.join(outputTrainDirectory, "model.h5")
    model.save(filePath)

    return model


def buildModel(width, height, depth, trainData, epochs, trainStepsPerEpoch, validateData = None, validateStepsPerEpoch = None):
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
        optimizer = Adam(learning_rate=1e-5),
        loss = "binary_crossentropy",
        metrics = ["accuracy"],
    )

    # Train the model
    history = model.fit(
        trainData,
        steps_per_epoch = trainStepsPerEpoch,
        epochs = epochs,
        validation_data = validateData,
        validation_steps = validateStepsPerEpoch
    )

    return model, history


def loadModelDefault():
    # Get default values
    outputTrainDirectory = readConfigFile("DIRECTORY", "outputTrain")

    # Run routine
    return loadModel(outputTrainDirectory)


def loadModel(outputTrainDirectory, fileName="model.h5"):
    filePath = os.path.join(outputTrainDirectory, fileName)
    if os.path.exists(filePath):
        model = load_model(filePath)
        return model
    else:
        raise Exception("No saved model found.")


def predictFilesDefault(model):
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
    predictions = model.predict(predictData, steps=(predictData.n // 10))   

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

