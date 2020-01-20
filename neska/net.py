import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__)))

import random
from decimal import Decimal
from sinaps import Sinaps
from neuron import Neuron, InputNeuron, HiddenNeuron, OutputNeuron, BiasNeuron


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
        self.E = Decimal(E) #learning rate
        self.A = Decimal(A) #momentum
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
                    self.__connect(Sinaps(Decimal(weightList[i][j][k])), previousLayer[k], neuron)
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
                self.__connect(Sinaps(Decimal(weightList[i][j][k])), previousLayer[k], neuron)
            if isBiasNeuronEnabled:
                self.__connect(Sinaps(self.__randomWeight()), previousLayer[-1], neuron)

    def calcForInput(self, inputValues):
        """-> [float] : Calculate output for inputs and return array of floats. 
        """
        for i in range(len(inputValues)):
            inputValues[i] = Decimal(inputValues[i])

        for i in range(len(inputValues)):
            self.inputs[i].value = inputValues[i]
            self.inputs[i].throwThrough()

        result = []
        for output in self.outputs:
            result.append(float(output.value))
        return result

    def calcForInputWithIdealOutput(self, inputValues, idealOutput):
        """->[float]: Calculate for input with self.calcForInput(inputValues). 
        Calculate error with self.calcMSE()
        """

        self.idealOutput = []
        for a in idealOutput:
            self.idealOutput.append(Decimal(a))

        result = self.calcForInput(inputValues)
        self.calcMSE()
        return result

    def calcMSE(self, idealOutput = []):
        """->MSE. Calculate MSE for self.outputs and self.idealOuput
        """
        if not len(idealOutput):
            idealOutput = self.idealOutput
        else:
            for i in range(len(idealOutput)):
                idealOutput[i] = Decimal(i)
        result = Decimal(0)
        for i in range(len(self.outputs)):
            result += (idealOutput[i] - self.outputs[i].value)**2
        
        result = result/len(self.outputs)
        self.MSE = result
        return self.MSE

    def getSinapsesWeightMap(self):
        """Get sinapses weight map. 
        Every elemnt consist of sinapses between two neuron layers
        And each such layer consist of element, describing input sinapses weights of single neuron
        """

        allSinapses = []

        neuron = self.inputs[0]
        while (not isinstance(neuron, OutputNeuron)):
            layerSinapses = []
            for curSinaps in neuron.outputSinaps:
                neuronSinapses = []
                for sinaps in curSinaps.outNeuron.inputSinaps:
                    neuronSinapses.append(float(sinaps.weight))
                layerSinapses.append(neuronSinapses)
            allSinapses.append(layerSinapses)
            neuron = neuron.outputSinaps[0].outNeuron
        return allSinapses

    def calcDeltas(self):
        """Calculate deltas and update weights
        """

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

    def train(self, viborka, epoch):
        inputLen = len(viborka[0][0])
        outputLen = len(viborka[0][1])

        for v in viborka:
            if (len(v[0]) != inputLen) or (len(v[1]) != outputLen):
                print("WARNING: ---Incorrect VIBORKA for neuroNet---")
                return 1
        for _ in range(epoch):
            for v in viborka:
                self.calcForInputWithIdealOutput(v[0], v[1])
                self.calcDeltas()

    def validateNeuronList(self, viborka):
        # TODO
        pass

    @staticmethod
    def __randomWeight():
        """__randomWeight() -> x: random Decimal number 
        """
        return Decimal(random.random() * 4 - 2)

    def __connect(self, sinaps, inputNeuron, outputNeuron):
        """Connect sinaps witn inputNeuron and outputNeuron
        """
        sinaps.outNeuron = outputNeuron
        sinaps.inputNeuron = inputNeuron
        outputNeuron.addInputSinaps(sinaps)
        inputNeuron.addOutputSinaps(sinaps)
