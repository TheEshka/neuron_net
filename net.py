import math

layerNumber = 2

inputValues = [
    [1, 0]
]

expectedOutput = [1]

# Transfer weights from first layer to second. First index - "from", second - "to"
p1 = [
    [0.45, 0.78],
    [-0.12, 0.13]
]

# Transfer weights from second layer to third. First index - "from", second - "to"
p2 = [
    [1.5],
    [-2.3]
]

allWeights = [p1, p2] 

# activision function
def sigmoid(x):
    return 1/(1 + math.exp(-x))

# error calculation
def calcMSE(output, expectedOutput):
    mse = 0
    for num in range(len(expectedOutput)):
        mse += (expectedOutput[num] - output[num])**2
    mse = mse / (len(expectedOutput))
    return mse


# getting output
for layer in range(layerNumber):
    h1column = []
    for columnNumber in range(len(allWeights[layer][0])):
        inputSum = 0
        for lineNumber in range(len(allWeights[layer])):
            # print(allWeights[layer][lineNumber][columnNumber])
            # print(inputValues[layer][lineNumber])
            # print()
            inputSum += allWeights[layer][lineNumber][columnNumber]*inputValues[layer][lineNumber]

        h1column.append(sigmoid(inputSum))

    inputValues.append(h1column)

output = inputValues[len(inputValues)-1]

print(calcMSE(output, expectedOutput))