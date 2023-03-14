# Load required modules
import tensorflow as tf

# Load required scripts
from scripts.config import initConfigFile
from scripts.model import executeCrossValidationDefault, trainFinalModelDefault, predictTestFilesDefault
from scripts.visualizePrediction import visualizeTestPredictionsDefault


def main():
    # Init config file
    initConfigFile()

    # Execute cross validation and train model
    executeCrossValidationDefault()
    model = trainFinalModelDefault()

    # Predict test files
    predictTestFilesDefault(model)

    # Visualize positive predictions
    tf.compat.v1.disable_eager_execution()
    visualizeTestPredictionsDefault()


# Execute main routine
if __name__ == '__main__':
    main()
