# Load required modules
import matplotlib.pyplot as plt


def histPrediction(predictions, filePath = None):
    plt.clf()
    plt.hist(list(predictions.percentage), label=["Tell"])
    plt.legend(loc='upper center', frameon=False)    
    if filePath == None:
        plt.show()
    else:
        plt.savefig(filePath)
       

def histPredictionTest(predictions, filePath = None):
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
