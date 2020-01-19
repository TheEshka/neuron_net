from neska.net import NeuroNet

neuronList = [[[0.45, -0.12], [0.78, 0.13]],[[1.5, -2.3]]]
viborka = [[[0, 0], [0]], [[0, 1], [1]], [[1, 0], [1]], [[1, 1], [0]]]
net = NeuroNet(neuronList, 0.7, 0.3, False)
print(net.getSinapsesWeightMap())
# net = NeuroNet.createNet([2,2,1], 0.7, 0.3, False)
for i in range(10000):
    mse = 0
    for v in viborka:
        net.improveWithInputAndOutput(v[0], v[1])
        net.calcDeltas()
        mse += net.MSE
    if i>990000:
        print(mse/4)

# print(net.getSinapsesWeightMap())
mse = 0
for v in viborka:
    net.improveWithInputAndOutput(v[0], v[1])
    print("\n")
    print("Input: ", v[0], " Expected output: ", v[1])
    print("Output: ", net.outputs[0].value)
    print("MSE: ", net.MSE)
    mse += net.MSE
print(mse/4)







# neuronList = [[[0.5]], [[-2.3]]]
# viborka = [[[0], [1/32]], [[1/8], [1/46.4]], [[1/15], [1/59]], [[1/22], [1/71,6]], [[1/38], [1/100.4]]]
# # net = NeuroNet(neuronList, 0.4, 0.3, True)
# net = NeuroNet.createNet([1,2,1], 0.7, 0.3, True)
# for i in range(10000):
#     mse = 0 
#     for v in viborka:
#         net.improveWithInputAndOutput(v[0], v[1])
#         net.calcDeltas()
#         mse += net.MSE

#     # print(mse)

# net.improveWithInputAndOutput([12], [53.6])
# print(1/net.outputs[0].value)








# neuronList = [[[-3.5143580532580674, -3.377054480250476], [-11.288578870314078, -11.261286908579724]], [[52.38430795334441, -62.52342952301519]]]
# net = NeuroNet(neuronList, 0.7, 0.3, False)
# # print(net.getSinapsesWeightMap())
# net.improveWithInputAndOutput([1,0], [1])
# print(net.outputs[0].value)
# print(net.MSE) 
