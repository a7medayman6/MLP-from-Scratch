import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn import datasets
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from MLP import MLP


def load_dataset():
    print("[INFO] loading MNIST (sample) dataset...")
    digits = datasets.load_digits()
    data = digits.data.astype("float")

    # normalize the data
    data = (data - data.min()) / (data.max() - data.min())
    print("[INFO] samples: {}, dim: {}".format(data.shape[0], data.shape[1]))

    

    # Dataset Split
    return train_test_split(data, digits.target, test_size=0.25)

def one_hot_encoding(y):
    return LabelBinarizer().fit_transform(y)
    
def train(x, y, classes, layers, epochs=1000, activation='sigmoid', lr=0.01, bias=False):
    print("[INFO] training network...")

    # add input and output layers
    layers.insert(0, x.shape[1])
    layers.append(len(classes))

    network = MLP(layers, lr=lr, activation=activation, bias=bias)

    network.fit(x, y,epochs=epochs)

    print("[INFO] finished training network...")

    return network

def eval(network, x, y):
    print("[INFO] evaluating network...")
    predictions = network.predict(x)
    predictions = predictions.argmax(axis=1)
    print(classification_report(y.argmax(axis=1), predictions))

    accuracy = accuracy_score(y.argmax(axis=1), predictions)

    return predictions, accuracy

def display_confusion_matrix(y, yhat, classes):

    cm = confusion_matrix(y.argmax(axis=1), yhat)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot()
    plt.show()
