import random
from .sinaps import Sinaps
from .neuron import Neuron, InputNeuron, HiddenNeuron, OutputNeuron, BiasNeuron


class NeuroNet:

    @classmethod
    def createNet(cls, neuronList, E, A, isBiasNeuronEnabled = False):
        weightList = []
        for i in range(1, len(neuronList)):
            layerWeightList = []
            for _ in range(neuronList[i]):
                neuronWeightList = []
                for _ in range(neuronList[i-1]):
                    neuronWeightList.append(NeuroNet.__randomWeight())
                layerWeightList.append(neuronWeightList)
            weightList.append(layerWeightList)
        return cls(weightList, E, A, isBiasNeuronEnabled)
    
    def __init__(self, weightList, E, A, isBiasNeuronEnabled):
        self.E = E #learning rate
        self.A = A #momentum
        self.inputs = []
        self.outputs = []
        self.idealOutput = []
        self.isBiasNeuronEnabled = isBiasNeuronEnabled
        
        
        # Creating input layer neurons
        currentLayer = []
        previousLayer = []
        for i in range(len(weightList[0][0])):
            self.inputs.append(InputNeuron())
        if isBiasNeuronEnabled:
            self.inputs.append(BiasNeuron())
        previousLayer = self.inputs

        # Creating hidden layer neurons
        for i in range(len(weightList) - 1):
            currentLayer = []
            for j in range(len(weightList[i])):
                neuron = HiddenNeuron()
                currentLayer.append(neuron) 
                for k in range(len(weightList[i][j])):
                    self.__connect(Sinaps(weightList[i][j][k]), previousLayer[k], neuron)
                if isBiasNeuronEnabled:
                    self.__connect(Sinaps(self.__randomWeight()), previousLayer[-1], neuron)
            if isBiasNeuronEnabled:
                currentLayer.append(BiasNeuron())
            previousLayer = currentLayer

        # Creating output layer neurons
        i = len(weightList) - 1
        self.outputs = []
        for j in range(len(weightList[i])):
            neuron = OutputNeuron()
            self.outputs.append(neuron) 
            for k in range(len(weightList[i][j])):
                self.__connect(Sinaps(weightList[i][j][k]), previousLayer[k], neuron)
            if isBiasNeuronEnabled:
                self.__connect(Sinaps(self.__randomWeight()), previousLayer[-1], neuron)

    def calculateWithInput(self, inputValues, idealOutput):
        self.idealOutput = idealOutput
        if len(inputValues) != (len(self.inputs) - int(self.isBiasNeuronEnabled)):
            print("WARNING: ---Incorrect input for neuroNet---")
            return 1

        for i in range(len(inputValues)):
            self.inputs[i].value = inputValues[i]
            self.inputs[i].throwThrough()
        
        self.calcMSE(idealOutput)

    def calcMSE(self, idealOutput):
        result = 0
        for i in range(len(self.outputs)):
            result += (idealOutput[i] - self.outputs[i].value)**2
        
        result = result/len(self.outputs)
        self.MSE = result

    # def getSinapsesWeightMap(self):
    #     allSinapses = []
    #     for i in range(1, len(self.neurons)):
    #         layerSinapses = []
    #         for neuron in self.neurons[i]:
    #             neuronSinaps = []
    #             for sinaps in neuron.inputSinaps:
    #                 neuronSinaps.append(sinaps.weight)
    #             layerSinapses.append(neuronSinaps)
    #         allSinapses.append(layerSinapses)
    #     return allSinapses

    ###
    ### Calculate deltas and update weights
    def calcDeltas(self):

        for i in range(len(self.outputs)):
            neuron = self.outputs[i]
            neuron.delta = (self.idealOutput[i]-neuron.value) * ((1-neuron.value)*neuron.value)

        # got layar from right. Take first neuron in layer and get its input sinapses to get lefter neurons
        rightNeuron = self.outputs[0]
        while (not isinstance(rightNeuron, InputNeuron)):
            for inputSinaps in rightNeuron.inputSinaps:
                neuron = inputSinaps.inputNeuron
                for sinaps in neuron.outputSinaps:
                    neuron.delta += sinaps.weight * sinaps.outNeuron.delta
                    grad = neuron.value * sinaps.outNeuron.delta
                    sinapsDelta = self.E * grad + self.A * sinaps.previousDelta
                    sinaps.weight += sinapsDelta
                    sinaps.previousDelta = sinapsDelta
                neuron.delta *= (1 - neuron.value) * neuron.value
            rightNeuron = rightNeuron.inputSinaps[0].inputNeuron

    def train(self, viborka):
        # TODO
        pass

    def validateNeuronList(self, viborka):
        # TODO
        pass

    @staticmethod
    def __randomWeight():
        return round(random.random() * 2 - 1, 2)

    def __connect(self, sinaps, inputNeuron, outputNeuron):
        """Connect sinaps witn inputNeuron and outputNeuron
        """

        sinaps.outNeuron = outputNeuron
        sinaps.inputNeuron = inputNeuron
        outputNeuron.addInputSinaps(sinaps)
        inputNeuron.addOutputSinaps(sinaps)
