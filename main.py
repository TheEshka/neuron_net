from neska.net import NeuroNet

neuronList = [[[0.45, -0.12], [0.78, 0.13]],[[1.5, -2.3]]]
viborka = [[[0, 0], [0]], [[0, 1], [1]], [[1, 0], [1]], [[1, 1], [0]]]
net = NeuroNet(neuronList, 0.7, 0.3, False)
for i in range(1000):
    mse = 0
    for v in viborka:
        net.calculateWithInput(v[0], v[1])
        net.calcDeltas()
        mse += net.MSE
    if i>990000:
        print(mse/4)

# print(net.getSinapsesWeightMap())
mse = 0
for v in viborka:
    net.calculateWithInput(v[0], v[1])
    print("\n")
    print("Input: ", v[0], " Expected output: ", v[1])
    print("Output: ", net.outputs[0].value)
    print("MSE: ", net.MSE)
    mse += net.MSE
print(mse/4)

