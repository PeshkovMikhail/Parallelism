import torch
import torch.nn as nn
import numpy as np

import os
class Net(nn.Module):

    def __init__(self):

         super(Net, self).__init__()

         self.fc1 = nn.Linear(32**2, 16**2) # входной слой

         self.fc2 = nn.Linear(16**2, 4**2) # скрытый слой

         self.fc3 = nn.Linear(4 ** 2,1) # скрытый слой

# прямое распространение информации

    def forward(self, x):

        sigmoid = nn.Sigmoid()

        x = sigmoid(self.fc1(x))

        x = sigmoid(self.fc2(x))

        x = sigmoid(self.fc3(x))

        return x





inputs = torch.rand(32**2) # входные данные нейронной сети

net = Net() # создание объекта "нейронная сеть"
print(net.fc1.weight.detach().numpy().dtype)

result = net(inputs) # запуск прямого распространения информации

print(result)

if not os.path.exists("weights/"):
    os.mkdir("weights")

np.save("weights/input.npy", inputs.detach().numpy())

np.save("weights/fc1.npy", net.fc1.weight.detach().numpy())
np.save("weights/fc1_bias.npy", net.fc1.bias.detach().numpy())

np.save("weights/fc2.npy", net.fc2.weight.detach().numpy())
np.save("weights/fc2_bias.npy", net.fc2.bias.detach().numpy())

np.save("weights/fc3.npy", net.fc3.weight.detach().numpy())
np.save("weights/fc3_bias.npy", net.fc3.bias.detach().numpy())
