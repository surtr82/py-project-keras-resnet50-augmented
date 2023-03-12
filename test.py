# Load required modules
import tensorflow as tf

# Load py-scripts
from scripts.config import initConfigFile
from scripts.model import loadModelDefault, predictTestFilesDefault
from scripts.visualizePrediction import visualizeTestPredictionsDefault


def main():
    # Init config file
    initConfigFile()

    # Train model
    model = loadModelDefault()

    # Predict test files
    predictTestFilesDefault(model)

    # Visualize positive predictions
    visualizeTestPredictionsDefault()


# Execute main routine
if __name__ == '__main__':
    tf.compat.v1.disable_eager_execuion()
    main()
