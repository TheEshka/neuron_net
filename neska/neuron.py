import math

class Neuron(object):

    def __init__(self):
        self.value = 0
        self.inputSinaps = []
        self.outputSinaps = []
        self.__cacheOfInput = [] ## TODO: change kostil
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
        if ( (len(self.inputSinaps) == 1) and (isinstance(self.inputSinaps[0].inputNeuron, BiasNeuron)) ):
            self.value += self.inputSinaps[0].weight
            self.inputSinaps = []
        if (len(self.inputSinaps)):
            return
        
        self.inputSinaps = list(self.__cacheOfInput)
        self.__calcSigmoid()
        # Chacking exist output sinaps
        if (len(self.outputSinaps)):
            for sinaps in self.outputSinaps:
                sinaps.getInput(self.value)
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
            sinaps.getInput(self.value)


class HiddenNeuron(Neuron):
    pass


class OutputNeuron(Neuron):
    pass


class BiasNeuron(Neuron):
    
    def __init__(self):
        super(BiasNeuron, self).__init__()
        self.__value = 1