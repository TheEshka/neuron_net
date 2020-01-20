from neska.net import NeuroNet

# neuronList = [[[0.45, -0.12], [0.78, 0.13]],[[1.5, -2.3]]]
# viborka = [[[0, 0], [0]], [[0, 1], [1]], [[1, 0], [1]], [[1, 1], [0]]]
# # net = NeuroNet(neuronList, 0.7, 0.3, False)
# net = NeuroNet.createNet([2,2,1], 0.7, 0.3, True)
# for i in range(10000):
#     mse = 0
#     for v in viborka:
#         net.improveWithInputAndOutput(v[0], v[1])
#         net.calcDeltas()
#         mse += net.MSE
#     if i>990000:
#         print(mse/4)

# print(net.getSinapsesWeightMap())
# mse = 0
# for v in viborka:
#     net.improveWithInputAndOutput(v[0], v[1])
#     print("\n")
#     print("Input: ", v[0], " Expected output: ", v[1])
#     print("Output: ", net.outputs[0].value)
#     print("MSE: ", net.MSE)
#     mse += net.MSE
# print(mse/4)







neuronList = [[[0.5]], [[-2.3]]]
viborka = [[[0], [1/32]], [[1/8], [1/46.4]], [[1/15], [1/59]], [[1/22], [1/71.6]], [[1/38], [1/100.4]]]
# net = NeuroNet(neuronList, 0.4, 0.3, True)
net = NeuroNet.createNet([1,3,1], 0.4, 0.3, True)
print(net.getSinapsesWeightMap())
net.train(viborka, 1000)
print(net.getSinapsesWeightMap())

mse = 0
for v in viborka:
    r = net.calcForInputWithIdealOutput(v[0], v[1])
    print("\n")
    print("Input: ", v[0][0], " Expected output: ", v[1][0])
    print("Output: ", 1/r[0])
    print("MSE: ", net.MSE)
print(mse/4)

net.calcForInput([12]) #53.6
print(1/net.outputs[0].value)








# neuronList = [[[-3.5143580532580674, -3.377054480250476], [-11.288578870314078, -11.261286908579724]], [[52.38430795334441, -62.52342952301519]]]
# net = NeuroNet(neuronList, 0.7, 0.3, False)
# # print(net.getSinapsesWeightMap())
# net.improveWithInputAndOutput([1,0], [1])
# print(net.outputs[0].value)
# print(net.MSE) 





# IDEAL IDEAL
# [[[1.8590141917364174, -1.9734587966575976, -0.8468825805668709], [11.564897035242248, -9.038225204717946, 1.9382703628784843]], [[20.70058792121787, -14.1097803663027, 0.7805902318449678]]]