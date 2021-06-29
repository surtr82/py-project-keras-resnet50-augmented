# Load required py-packages
import matplotlib.pyplot as plt
import os
import pandas


def histPrediction(predictions, filePath = None):
    """histPrediction
        Arguments:
            predictions: DataFrame with predictions to visualize
    """    
    plt.clf()
    plt.hist(list(predictions.percentage), label=["Tell"])
    plt.legend(loc='upper center', frameon=False)    
    if filePath == None:
        plt.show()
    else:
        plt.savefig(filePath)
       

def histPredictionTest(predictions, filePath = None):
    """histPredictionTest
        Arguments:
            predictions: DataFrame with predictions to visualize
            filePath: Optional file path to save image to
    """
    other = predictions[predictions['name'].str.contains('Other|Map', case=True, regex=True)]
    tell = predictions[~predictions['name'].str.contains('Other|Map', case=True, regex=True, na = False)]

    plt.clf()
    plt.style.use('seaborn-deep')
    plt.hist([tell.percentage, other.percentage],  label=['Tell', 'Other'])
    plt.legend(loc='upper center', frameon=False)
    if filePath == None:
        plt.show()
    else:
        plt.savefig(filePath)
