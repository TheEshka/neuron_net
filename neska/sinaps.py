class Sinaps:

    ###
    ### Init sinaps with weight
    def __init__(self, weight):
        self.weight = weight
        self.outNeuron = None
        self.inputNeuron = None
        self.value = 0
        self.previousDelta = 0
    
    ###
    ### get value with triggering next neuron
    def getInput(self, value):
        self.value = self.weight * value
        self.outNeuron.addValue(self)