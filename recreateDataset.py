# Load py-scripts
from scripts.config import initConfigFile
from scripts.dataset import recreateDatasetDefault


def main():
    # Init config file
    initConfigFile()

    # Recreate dataset from input data
    recreateDatasetDefault()


# Execute main routine
if __name__ == '__main__':
    main()
