import math


class NeuroNet:
    
    # TODO: add bias sinaps
    def __init__(self, neuronList, E, A, isBiasNeuronEnabled):
        self.E = E #learning rate
        self.A = A #momentum
        self.neurons = []
        self.output = []
        self.idealOutput = []
        self.isBiasNeuronEnabled = isBiasNeuronEnabled
        
        # Creating input layer neurons
        inputNeurons = []
        for i in range(len(neuronList[0][0])):
            inputNeurons.append(InputNeuron())
        if isBiasNeuronEnabled:
            inputNeurons.append(BiasNeuron())
        self.neurons.append(inputNeurons)

        # Creating hidden layer neurons
        for i in range(len(neuronList) - 1):
            hiddenNeuronsLayer = []
            for j in range(len(neuronList[i])):
                neuron = Neuron()
                hiddenNeuronsLayer.append(neuron) 
                for k in range(len(neuronList[i][j])):
                    sinaps = Sinaps(neuronList[i][j][k], neuron)
                    self.neurons[-1][k].addOutputSinaps(sinaps)
                if isBiasNeuronEnabled:
                    sinaps = Sinaps(1, neuron) ## TODO: weight
                    self.neurons[-1][-1].addOutputSinaps(sinaps)
            if isBiasNeuronEnabled:
                hiddenNeuronsLayer.append(BiasNeuron())
            self.neurons.append(hiddenNeuronsLayer)

        # Creating output layer neurons
        i = len(neuronList) - 1
        outputNeuronsLayer = []
        for j in range(len(neuronList[i])):
            neuron = Neuron()
            outputNeuronsLayer.append(neuron) 
            for k in range(len(neuronList[i][j])):
                sinaps = Sinaps(neuronList[i][j][k], neuron)
                self.neurons[-1][k].addOutputSinaps(sinaps)
            if isBiasNeuronEnabled:
                    sinaps = Sinaps(1, neuron) ## TODO: weight
                    self.neurons[-1][-1].addOutputSinaps(sinaps)
        self.neurons.append(outputNeuronsLayer)

    def calculateWithInput(self, inputValues, idealOutput):
        self.output = []
        self.idealOutput = idealOutput
        if len(inputValues) != len(self.neurons[0]):
            print("WARNING: ---Incorrect input for neuroNet---")

        for i in range(len(inputValues)):
            self.neurons[0][i].value = inputValues[i]
            self.neurons[0][i].throwThrough()
        
        for i in range(len(self.neurons[-1])):
            self.output.append(self.neurons[-1][i].value)
        self.calcMSE(idealOutput)

    def calcMSE(self, idealOutput):
        result = 0
        for i in range(len(self.output)):
            result += (idealOutput[i] - self.output[i])**2
        
        result = result/len(self.output)
        self.MSE = result

    def getSinapsesWeightMap(self):
        allSinapses = []
        for i in range(1, len(self.neurons)):
            layerSinapses = []
            for neuron in self.neurons[i]:
                neuronSinaps = []
                for sinaps in neuron.inputSinaps:
                    neuronSinaps.append(sinaps.weight)
                layerSinapses.append(neuronSinaps)
            allSinapses.append(layerSinapses)
        return allSinapses

    ###
    ### Calculate deltas and update weights
    def calcDeltas(self):

        for i in range(len(self.neurons[-1])):
            neuron = self.neurons[-1][i]
            neuron.delta = (self.idealOutput[i]-neuron.value) * ((1-neuron.value)*neuron.value)

        for i in range(len(self.neurons)-2, -1, -1):
            for neuron in self.neurons[i]:
                for sinaps in neuron.outputSinaps:
                    neuron.delta += sinaps.weight * sinaps.outNeuron.delta
                    grad = neuron.value * sinaps.outNeuron.delta
                    sinapsDelta = self.E * grad + self.A * sinaps.previousDelta
                    sinaps.weight += sinapsDelta
                    sinaps.previousDelta = sinapsDelta
                neuron.delta *= (1 - neuron.value) * neuron.value

    def train(self, viborka):
        # TODO
        pass

class Neuron(object):

    def __init__(self):
        self.value = 0
        self.inputSinaps = []
        self.outputSinaps = []
        self.__cacheOfInput = []
        self.delta = 0

    ###
    ### Add inputSinaps for initialization net
    def addInputSinaps(self, sinaps):
        if (self.__checkExist(sinaps)):
            print("WARNING: ---Trying add repeating INPUT sinaps---")
            return
        self.inputSinaps.append(sinaps)
        self.__cacheOfInput.append(sinaps)

    ###
    ### Add for initialization net
    def addOutputSinaps(self, sinaps):
        if (self.__checkExist(sinaps)):
            print("WARNING: ---Trying add repeating OUTPUT sinaps---")
            return
        self.outputSinaps.append(sinaps)
    
    ###
    ### Add value from sinaps with triggering sinaps
    def addValue(self, sinaps):
        self.inputSinaps.remove(sinaps)
        self.value += sinaps.value

        # Check calculate all input sinaps
        if (len(self.inputSinaps)):
            return
        
        self.inputSinaps = list(self.__cacheOfInput)
        self.__calcSigmoid()
        # Chacking exist output sinaps
        if (len(self.outputSinaps)):
            for sinaps in self.outputSinaps:
                sinaps.getValue(self.value)
            return 

        # this is output sinaps
        # print("-------Output neuron reached-------")
        return
    
    def __checkExist(self, checkingSinaps):
        for sinaps in self.inputSinaps:
            if (checkingSinaps == sinaps):
                print("Find in inputSinapses")
                return True

        for sinaps in self.inputSinaps:
            if (checkingSinaps == sinaps):
                print("Find in outputSinapses")
                return True
        
        return False
    
    ###
    ### Activation function
    def __calcSigmoid(self):
        self.value = 1/(1 + math.exp(-self.value))


class InputNeuron(Neuron):

    def throwThrough(self):
        for sinaps in self.outputSinaps:
            sinaps.getValue(self.value)


class HiddenNeuron(Neuron):
    pass


class OutputNeuron(Neuron):
    pass


class BiasNeuron(Neuron):
    
    def __init__(self):
        super(BiasNeuron, self).__init__()
        self.__value = 1


class Sinaps:

    ###
    ### Init sinaps with weight
    def __init__(self, weight, outNeuron):
        self.value = 0
        self.weight = weight
        self.outNeuron = outNeuron
        outNeuron.addInputSinaps(self)
        self.previousDelta = 0
    
    ###
    ### set value with triggering next neuron
    def getValue(self, value):
        self.value = self.weight * value
        self.outNeuron.addValue(self)


neuronList = [[[0.45, -0.12], [0.78, 0.13]],[[1.5, -2.3]]]
viborka = [[[0, 0], [0]], [[0, 1], [1]], [[1, 0], [1]], [[1, 1], [0]]]
net = NeuroNet(neuronList, 0.7, 0.3, True)
for i in range(100000):
    mse = 0
    for v in viborka:
        net.calculateWithInput(v[0], v[1])
        net.calcDeltas()
        mse += net.MSE
    if i>990000:
        print(mse/4)

print(net.getSinapsesWeightMap())
for v in viborka:
    net.calculateWithInput(v[0], v[1])
    print("\n")
    print("Input: ", v[0], " Expected output: ", v[1])
    print("Output: ", net.output)
    print("MSE: ", net.MSE)

# neuronList = [[[0.5]], [[-2.3]]]
# viborka = [[[0], [32]], [[8], [46.4]], [[15], [59]], [[22], [71,6]], [[38], [100.4]]]
# net = NeuroNet(neuronList, 0.7, 0.3, True)
# for i in range(1000):
#     mse = 0 
#     for v in viborka:
#         net.calculateWithInput(v[0], v[1])
#         net.calcDeltas
#         mse += net.MSE

#     print(mse)

# net.calculateWithInput([12], [53.6])
# print(net.output)




# neuronList = [[[-3.5143580532580674, -3.377054480250476], [-11.288578870314078, -11.261286908579724]], [[52.38430795334441, -62.52342952301519]]]
# net = NeuroNet(neuronList)
# print(net.getSinapsesWeightMap())
# net.calculateWithInput([0,0], [0])
# print(net.output)
# print(net.MSE)