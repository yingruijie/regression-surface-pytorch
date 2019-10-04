import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# set current path
os.chdir(os.path.split(os.path.realpath(__file__))[0])
# clear the file
path = 'image_output/'
for filename in os.listdir(path):
    os.remove(path + filename)

# create dataset
class Dataset(data.Dataset):
    def __init__(self, start_x1, end_x1, num_x1, start_x2, end_x2, num_x2, f):
        super().__init__()
        self.num_x1 = num_x1
        self.num_x2 = num_x2
        self.x1 = torch.linspace(start_x1, end_x1, num_x1)
        self.x2 = torch.linspace(start_x2, end_x2, num_x2)
        # 创建网格
        self.x1, self.x2 = torch.meshgrid(self.x1, self.x2) 
        self.z = f(self.x1, self.x2)
        self.x1 = self.x1.reshape(num_x1 * num_x2)
        self.x2 = self.x2.reshape(num_x1 * num_x2)
        self.z = self.z.reshape(num_x1 * num_x2)
        ax = plt.figure().add_subplot(111, projection = '3d')
        ax.scatter(torch.unsqueeze(self.x1, dim=1).cpu().numpy(), torch.unsqueeze(self.x2, dim=1).cpu().numpy(), torch.unsqueeze(self.z, dim=1).cpu().numpy(), c = 'b')
        plt.show()

    def __getitem__(self, item):
        return self.x1[item], self.x2[item], self.z[item]
    
    def __len__(self):
        return (self.num_x1 * self.num_x2)

# a linear net
class Net(nn.Module):
    def __init__(self, n_input, n_output):
        super(Net, self).__init__()
        self.linear = nn.Linear(n_input, 20)
        self.relu = F.relu
        self.output = nn.Linear(20, n_output)
    
    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = self.output(x)
        return x

start_x1 = -1
end_x1 = 1
num_x1 = 11
start_x2 = 0
end_x2 = 1
num_x2 = 11
# function
f = lambda x1, x2: 2 * x1.pow(2) + x2.pow(0.5)  + 0.1 * torch.rand(x1.shape) + 0.1 * torch.rand(x2.shape)
# get dataset
dataset = Dataset(start_x1, end_x1, num_x1, start_x2, end_x2, num_x2, f)
data_loader = data.DataLoader(dataset=dataset, batch_size=num_x1*num_x2, shuffle=False)
# net: input = 2, output = 1
net = Net(n_input = 2, n_output = 1).cuda()
# use SGD optimizier
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
# loss function
loss_function = nn.MSELoss()
losses = []
# training
for epoch in range(100):
    for _, train_data in enumerate(data_loader):
        x1 = torch.unsqueeze(train_data[0], dim=1).cuda()
        x2 = torch.unsqueeze(train_data[1], dim=1).cuda()
        z  = torch.unsqueeze(train_data[2], dim=1).cuda()
        
        x = torch.cat((x1, x2), 1).cuda()
        pred = net(x)
        loss = loss_function(pred, z)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss)

    if epoch % 10 == 0:
        # xi_test  : tensor(num_x1, num_x2)
        x1_test = torch.linspace(start_x1, end_x1, num_x1)
        x2_test = torch.linspace(start_x2, end_x2, num_x2)
        x1_test, x2_test = torch.meshgrid(x1_test, x2_test)
        # get z_pred by net(x) and z_read by f(x1, x2)
        z_pred = net(torch.cat(( x1_test.reshape(num_x1*num_x2, 1), x2_test.reshape(num_x1*num_x2, 1)), 1).cuda()).reshape(num_x1, num_x2)
        z_real = f(x1_test, x2_test)
        # plot and output
        ax = plt.figure().add_subplot(111, projection = '3d')
        scatt = ax.scatter(torch.unsqueeze(x1_test, dim=1).cpu().numpy(), torch.unsqueeze(x2_test, dim=1).cpu().numpy(), torch.unsqueeze(z_real, dim=1).cpu().numpy(), c = 'b')
        plot_sur = ax.plot_surface(x1_test.cpu().numpy(), x2_test.cpu().numpy(), z_pred.cpu().detach().numpy(), rstride=1, cstride=1, cmap='rainbow')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title("the " + str(epoch) + "th training")
        plt.savefig(path + str(epoch) + "_plt.jpg")
        plt.show()

batch_nums = range(1, len(losses) + 1)
plt.plot(batch_nums, losses)
plt.title("loss - Batch")
plt.xlabel("batch")
plt.ylabel("loss")
plt.savefig(path + "loss.jpg")
plt.show()
        
    
    