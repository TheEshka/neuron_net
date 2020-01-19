from decimal import Decimal

class Sinaps:
    """Representation of sinaps for neuron net

    self.value : Decimal
    self.weight : Decimal
    self.previousDelta : Decimal
    self.outNeuron  : Neurom (from neuron.py)
    self.inputNeuron = None : Neuron (from neuron.py)
    """

    def __init__(self, weight):
        """Init sinaps with weight
        """
        
        self.weight = Decimal(weight)
        self.outNeuron = None
        self.inputNeuron = None
        self.value = Decimal(0)
        self.previousDelta = Decimal(0)
    
    def getInput(self, value):
        """get value with triggering next neuron
        """

        self.value = self.weight * Decimal(value)
        self.outNeuron.addValue(self)