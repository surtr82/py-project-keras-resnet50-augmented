# Load required modules
import tensorflow as tf

# Load required scripts
from scripts.config import initConfigFile
from scripts.model import loadModelDefault, predictFilesDefault
from scripts.visualizePrediction import visualizePredictionsDefault


def main():
    # Init config file
    initConfigFile()
    
    # Get base and load trained model
    model = loadModelDefault()

    # Predict test files
    predictFilesDefault(model)

    # Visualize positive predictions
    tf.compat.v1.disable_eager_execution()
    visualizePredictionsDefault()


# Execute main routine
if __name__ == '__main__':
    main()
