# Load py-scripts
from scripts.config import initConfigFile
from scripts.model import loadModelDefault, predictFilesDefault


def main():
    """main

    """
    # Init config file
    initConfigFile()
    
    # Get base and load trained model
    model = loadModelDefault()

    # Predict test files
    predictFilesDefault(model)


# Execute main routine
if __name__ == '__main__':
    main()
