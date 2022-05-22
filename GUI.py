import matplotlib.pyplot as plt
from tkinter import *
from Helper import *

#from Controller import Controller


class GUI:
    def __init__(self, root):
        
        # Variables
        self.numberOfLayers = 0
        self.nodesList = list()
        self.nodesIndex = 0

        # model variables
        self.network = None
        self.x_train, self.x_test, self.y_train, self.y_test = load_dataset()
        self.y_train = one_hot_encoding(self.y_train)
        self.y_test = one_hot_encoding(self.y_test)
        self.classes = [0, 1, 2, 3, 4, 5, 6, 7, 8 ,9]
        self.predictions = None
        self.accuracy = 0

        # Frames
        self.numberOfLayersFrame = Frame(root)
        self.numberOfNodesFrame = Frame(root)
        self.activationFunctionFrame = Frame(root)

        # Buttons
        self.showPlotButton = Button(root, text="Confusion Matrix", command=self.plotButtonClicked)
        self.trainButton = Button(root, text="train", command=self.trainButtonClicked)
        self.testButton = Button(root, text="test", command=self.testButtonClicked)
        self.addNumberOfLayersButton = Button(self.numberOfLayersFrame, text="Add",
                                              command=self.addNumberOfLayersButtonClicked)
        self.addNodeButton = Button(self.numberOfNodesFrame, text="Add Nodes", command=self.addNodeButtonClicked)

        # labels string var
        self.learningRateStringVar = StringVar()
        self.epochsStringVar = StringVar()
        self.numberOfLayersStringVar = StringVar()
        self.numberOfNodesStringVar = StringVar()
        self.activationFunctionStringVar = StringVar()

        # labels
        self.learningRateLabel = Label(root, textvariable=self.learningRateStringVar)
        self.learningRateStringVar.set("learning rate(<=0.1)")
        self.epochsLabel = Label(root, textvariable=self.epochsStringVar)
        self.epochsStringVar.set("Number of epochs")
        self.numberOfLayersLabel = Label(root, textvariable=self.numberOfLayersStringVar)
        self.numberOfLayersStringVar.set("Number of layers")
        self.numberOfNodesLabel = Label(root, textvariable=self.numberOfNodesStringVar)
        self.numberOfNodesStringVar.set("Number of nodes")
        self.activationFunctionLabel = Label(root, textvariable=self.activationFunctionStringVar)
        self.activationFunctionStringVar.set("Activation function")

        # Entries string var
        self.learningRateEntryStringVar = StringVar()
        self.epochsEntryStringVar = StringVar()
        self.numberOfLayersEntryStringVar = StringVar()
        self.numberOfNodesEntryStringVar = StringVar()

        # Entries
        self.learningRateEntry = Entry(root, textvariable=self.learningRateEntryStringVar)
        self.epochsEntry = Entry(root, textvariable=self.epochsEntryStringVar)
        self.numberOfLayersEntry = Entry(self.numberOfLayersFrame, width=4,
                                         textvariable=self.numberOfLayersEntryStringVar)
        self.numberOfNodesEntry = Entry(self.numberOfNodesFrame, width=4, textvariable=self.numberOfNodesEntryStringVar)

        # bias checkbox
        self.biasIntVariable = IntVar()
        self.biasCheckBox = Checkbutton(root, text="Add Bias", variable=self.biasIntVariable)

        # activation function checkbox
        self.sigmoidActivationFunctionIntVar = IntVar()
        self.sigmoidActivationFunctionCheckbox = Checkbutton(self.activationFunctionFrame, text="Sigmoid",
                                                             variable=self.sigmoidActivationFunctionIntVar)
        self.tanhActivationFunctionIntVar = IntVar()
        self.tanhActivationFunctionCheckbox = Checkbutton(self.activationFunctionFrame, text="tanh",
                                                          variable=self.tanhActivationFunctionIntVar)
        self.sigmoidActivationFunctionCheckbox.pack(side="left")
        self.tanhActivationFunctionCheckbox.pack(side="left")

        # pack components of number of layers
        self.numberOfLayersEntry.pack(side="left", padx=10)
        self.addNumberOfLayersButton.pack(side="left")

        # pack components of adding node
        self.numberOfNodesEntry.pack(side="left", padx=10)
        self.addNodeButton.pack(side="left")


        # placing of components in window
        self.showPlotButton.grid(row=0, column=1)
        self.learningRateLabel.grid(row=1, column=0)
        self.learningRateEntry.grid(row=1, column=1)
        self.epochsLabel.grid(row=2, column=0)
        self.epochsEntry.grid(row=2, column=1)
        self.numberOfLayersLabel.grid(row=3, column=0)
        self.numberOfLayersFrame.grid(row=3, column=1)
        self.activationFunctionLabel.grid(row=4, column=0)
        self.activationFunctionFrame.grid(row=4, column=1)
        self.biasCheckBox.grid(row=6, column=0)

        self.trainButton.grid(row=7, column=0)
        self.testButton.grid(row=7, column=1)


    def plotButtonClicked(self):
        display_confusion_matrix(self.y_test, self.predictions, self.classes)
        print("\n\n[INFO] Overall Accuracy: {}".format(self.accuracy))
     
    def trainButtonClicked(self):
        # get data from components when pressed enter button
        learning_rate = float(self.learningRateEntryStringVar.get())
        epochs_ = int(self.epochsEntryStringVar.get())
        bias = True if self.biasIntVariable.get() == 1 else False
        sigmoid = self.sigmoidActivationFunctionIntVar.get()
        tanh = self.tanhActivationFunctionIntVar.get()

        if sigmoid == 1:
            activationFunction = "sigmoid"
        else:
            activationFunction = "tanh"
        
        self.network = train(self.x_train, self.y_train, self.classes, self.nodesList, epochs=epochs_, activation=activationFunction, lr=learning_rate)
        
    def testButtonClicked(self):
        # eval
        self.predictions, self.accuracy = eval(self.network, self.x_test, self.y_test)
         

    def addNumberOfLayersButtonClicked(self):
        self.getNumberOfLayers()
        self.showNodeInputComponents()

    def addNodeButtonClicked(self):
        self.getNumberOfNodesInEachLayer()

    def getNumberOfLayers(self):
        self.numberOfLayers = int(self.numberOfLayersEntryStringVar.get())

        # hide layer input components after number of layers is chosen
        self.numberOfLayersLabel.grid_forget()
        self.numberOfLayersFrame.grid_forget()

    def showNodeInputComponents(self):
        self.numberOfNodesLabel.grid(row=3, column=0)
        self.numberOfNodesFrame.grid(row=3, column=1)
        self.numberOfNodesStringVar.set("nodes for layer" + " " + str(self.nodesIndex + 1))

    def getNumberOfNodesInEachLayer(self):
        # append number of nodes of layer i in the list
        self.nodesList.append(int(self.numberOfNodesEntryStringVar.get()))
        self.nodesIndex += 1
        self.numberOfNodesEntryStringVar.set("")
        self.numberOfNodesStringVar.set("nodes for layer" + " " + str(self.nodesIndex + 1))

        # hide node input components after number of nodes fore each layer was chosen
        if self.nodesIndex >= self.numberOfLayers:
            self.numberOfNodesLabel.grid_forget()
            self.numberOfNodesFrame.grid_forget()


def main():
    top = Tk()
    top.geometry("400x400")
    top.title("MINST MLP")
    g = GUI(top)

    top.mainloop()


if __name__ == "__main__":
    main()
