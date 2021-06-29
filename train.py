# Load py-scripts
from scripts.config import initConfigFile
from scripts.model import trainModelDefault, predictTestFilesDefault


def main():
    """main

    """
    # Init config file
    initConfigFile()

    # Train model
    model = trainModelDefault()

    # Predict test files
    predictTestFilesDefault(model)


# Execute main routine
if __name__ == '__main__':
    main()
