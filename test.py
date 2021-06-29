# Load py-scripts
from visualizePrediction import visualizeTestPredictionsDefault
from scripts.config import initConfigFile
from scripts.model import loadModelDefault, predictTestFilesDefault
import tensorflow as tf


def main():
    """main

    """
    # Init config file
    initConfigFile()

    # Train model
    model = loadModelDefault()

    # Predict test files
    predictTestFilesDefault(model)
    visualizeTestPredictionsDefault()


# Execute main routine
if __name__ == '__main__':
    tf.compat.v1.disable_eager_execution()
    main()
