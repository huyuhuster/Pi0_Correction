import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
 
torch.manual_seed(1)
 
EPOCH = 1
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = True
 
# train dataset
training_data = torchvision.datasets.MNIST(
             root='./mnist/', # dataset storage path
             train=True, # True mean train dataset，False mean test dataset
             transform=torchvision.transforms.ToTensor(), # Normalization the data to（0,1
             download=DOWNLOAD_MNIST,
             )
 
# print the size of the train sample and test dataset
print(training_data.train_data.size())
print(training_data.train_labels.size())
# torch.Size([60000, 28, 28])
# torch.Size([60000])
 
plt.imshow(training_data.train_data[0].numpy(), cmap='gray')
plt.title('%i' % training_data.train_labels[0])
plt.show()
 
# The format of the dataset get via torchvision.datasets cab directly put in DataLoader
train_loader = Data.DataLoader(dataset=training_data, batch_size=BATCH_SIZE,
                               shuffle=True)
 
# get the dataset
test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)
# get the firt 2000 test dataset
test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1),
                  volatile=True).type(torch.FloatTensor)[:2000]/255
# (2000, 28, 28) to (2000, 1, 28, 28), in range(0,1)
test_y = test_data.test_labels[:2000]
 
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential( # (1,28,28)
                     nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5,
                               stride=1, padding=2), # (16,28,28)
        # want to don't change the size of picture convoluted by con2d, padding=(kernel_size-1)/2
                     nn.ReLU(),
                     nn.MaxPool2d(kernel_size=2) # (16,14,14)
                     )
        self.conv2 = nn.Sequential( # (16,14,14)
                     nn.Conv2d(16, 32, 5, 1, 2), # (32,14,14)
                     nn.ReLU(),
                     nn.MaxPool2d(2) # (32,7,7)
                     )
        self.out = nn.Linear(32*7*7, 10)
 
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1) # flat（batch，32,7,7）to（batch，32*7*7）
        output = self.out(x)
        return output
 
cnn = CNN()
print(cnn)
'''
CNN (
  (conv1): Sequential (
    (0): Conv2d(1, 16, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (1): ReLU ()
    (2): MaxPool2d (size=(2, 2), stride=(2, 2), dilation=(1, 1))
  )
  (conv2): Sequential (
    (0): Conv2d(16, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (1): ReLU ()
    (2): MaxPool2d (size=(2, 2), stride=(2, 2), dilation=(1, 1))
  )
  (out): Linear (1568 -> 10)
)
'''
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_function = nn.CrossEntropyLoss()
 
for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):
        b_x = Variable(x)
        b_y = Variable(y)
 
        output = cnn(b_x)
        loss = loss_function(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
 
        if step % 100 == 0:
            test_output = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = sum(pred_y == test_y) / test_y.size(0)
            print('Epoch:', epoch, '|Step:', step,
                  '|train loss:%.4f'%loss.data[0], '|test accuracy:%.4f'%accuracy)
 
test_output = cnn(test_x[:10])
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print(pred_y, 'prediction number')
print(test_y[:10].numpy(), 'real number')
'''
Epoch: 0 |Step: 0 |train loss:2.3145 |test accuracy:0.1040
Epoch: 0 |Step: 100 |train loss:0.5857 |test accuracy:0.8865
Epoch: 0 |Step: 200 |train loss:0.0600 |test accuracy:0.9380
Epoch: 0 |Step: 300 |train loss:0.0996 |test accuracy:0.9345
Epoch: 0 |Step: 400 |train loss:0.0381 |test accuracy:0.9645
Epoch: 0 |Step: 500 |train loss:0.0266 |test accuracy:0.9620
Epoch: 0 |Step: 600 |train loss:0.0973 |test accuracy:0.9685
Epoch: 0 |Step: 700 |train loss:0.0421 |test accuracy:0.9725
Epoch: 0 |Step: 800 |train loss:0.0654 |test accuracy:0.9710
Epoch: 0 |Step: 900 |train loss:0.1333 |test accuracy:0.9740
Epoch: 0 |Step: 1000 |train loss:0.0289 |test accuracy:0.9720
Epoch: 0 |Step: 1100 |train loss:0.0429 |test accuracy:0.9770
[7 2 1 0 4 1 4 9 5 9] prediction number
[7 2 1 0 4 1 4 9 5 9] real number
'''

