from neska.net import NeuroNet
from decimal import Decimal

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



def normilize(qwe): #TODO: to net.py
    x_min = 0
    x_max = 105
    d_min = 0
    d_max = 1
    qwe = (qwe - x_min)*(d_max - d_min)/(x_max - x_min) + d_min
    return Decimal(qwe)

def denormalize(asd):
    x_min = 0
    x_max = 105
    d_min = 0
    d_max = 1
    asd = (asd - d_min)*(x_max - x_min)/(d_max - d_min) + x_min
    return asd

viborka = [
    [[normilize(0)], [normilize(32)]], 
    [[normilize(8)], [normilize(46.4)]],
    [[normilize(15)], [normilize(59)]],
    [[normilize(22)], [normilize(71.6)]], 
    [[normilize(38)], [normilize(100.4)]]
]

neuronList = [[[-1.1480110551817493, 1.37097896611021]], [[0.6893808129013457, 1.2392702577003245]]]
# neuronList = [[[-6.201413597258762, 1.961061194640626]], [[-7.347050137648256, 5.694153069720562]]]
# viborka = [[[0], [1/32]], [[1/8], [1/46.4]], [[1/15], [1/59]], [[1/22], [1/71.6]], [[1/38], [1/100.4]]]
# net = NeuroNet(neuronList, 0.4, 0.3, True)
net = NeuroNet.createNet([1, 1, 1], 0.4, 0.3, True)
print(net.getSinapsesWeightMap())
net.train(viborka, 4000)
print(net.getSinapsesWeightMap())

for v in viborka:
    r = net.calcForInputWithIdealOutput(v[0], v[1])
    print("\n")
    print("Input: ", denormalize(v[0][0]), " Expected output: ", denormalize(v[1][0]))
    print("Output: ", denormalize(r[0]))
    print("MSE: ", net.MSE)

net.calcForInput([normilize(12)]) #53.6
print(denormalize(net.outputs[0].value))








# neuronList = [[[-3.5143580532580674, -3.377054480250476], [-11.288578870314078, -11.261286908579724]], [[52.38430795334441, -62.52342952301519]]]
# net = NeuroNet(neuronList, 0.7, 0.3, False)
# # print(net.getSinapsesWeightMap())
# net.improveWithInputAndOutput([1,0], [1])
# print(net.outputs[0].value)
# print(net.MSE) 





# IDEAL IDEAL
# [[[1.8590141917364174, -1.9734587966575976, -0.8468825805668709], [11.564897035242248, -9.038225204717946, 1.9382703628784843]], [[20.70058792121787, -14.1097803663027, 0.7805902318449678]]]