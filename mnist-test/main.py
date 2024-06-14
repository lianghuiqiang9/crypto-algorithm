
import torch
import torchvision
from torch.utils.data import DataLoader

n_epochs = 1 # 循环整个训练数据集的次数。
batch_size_train = 64 # 训练数据集批量大小为64
batch_size_test = 1000 # 测试数据集的批量大小为1000，但是终端中显示未10000，多一个0，一批10个吧
learning_rate = 0.01 # 学习率为0.01
momentum = 0.5 # 动量参数，用于优化算法（如 SGD）
log_interval = 10 # 日志记录间隔（每多少个批次记录一次日志）
random_seed = 1 # 随机种子，用于结果的可重复性
torch.manual_seed(random_seed)

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,)) #指定数据转换操作，这里将数据转换为张量并进行归一化。
                               ])),
    batch_size=batch_size_train, shuffle=True)
    # transforms.Compose 是一个将多个数据转换操作串联在一起的工具。
    # torchvision.transforms.ToTensor(): 将 PIL 图像或 NumPy 数组转换为 PyTorch 张量，并且将像素值从 [0, 255] 范围归一化到 [0, 1] 范围。
    # torchvision.transforms.Normalize((0.1307,), (0.3081,)): 归一化张量，将每个通道的像素值减去均值 (0.1307) 并除以标准差 (0.3081)。这些值是根据 MNIST 数据集的全局统计特性计算得到的。
    # batch_size_train 指定每个批次加载的样本数量。这里用变量 batch_size_train（假设其值为 64）来指定批次大小。
    # shuffle=True 指定在每个 epoch 开始前随机打乱数据集。这有助于提高模型的泛化能力，避免模型在数据集的特定顺序上过拟合。



test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size_test, shuffle=True)
    # 除了数据集和批处理大小之外，PyTorch的DataLoader还包含一些有趣的选项。例如，我们可以使用num_workers > 1来使用子进程异步加载数据，或者使用固定RAM(通过pin_memory)来加速RAM到GPU的传输。但是因为这些在我们使用GPU时很重要，我们可以在这里省略它们。

examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)
print("print the data")
print(batch_idx)
print(example_targets)
print("print the data shape")
print(example_data.shape) #有1000个图片，目标也是1000个。
print("end the print the data")
# 其中 batch_idx 是批次的索引，example_data 是图像数据，example_targets 是标签。

'''
import matplotlib.pyplot as plt
fig = plt.figure()
for i in range(6):
  plt.subplot(2,3,i+1) # 2 行 3 列的布局
  plt.tight_layout()   # 这行代码调整子图的布局，以确保它们之间的间距合适。
  plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
  plt.title("Ground Truth: {}".format(example_targets[i]))
  plt.xticks([])
  plt.yticks([])
plt.show()
# 这段代码会在一个图形窗口中绘制出前 6 个样本的图像，每个图像下方显示对应的标签。
# 有1000个单通道的28*28像素的灰度(没有rgb通道)
'''

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    # 定义了一个名为 Net 的类，它继承自 nn.Module，这是构建神经网络模型的基类。
    def __init__(self):
        super(Net, self).__init__()
        # 在 __init__ 方法中，调用了父类 nn.Module 的构造函数，确保正确地初始化模型。
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        # 输入 28*28 输出 10*24*24
        # 最大池化2 输出 10*12*12
        # 定义了第一个卷积层 conv1，使用 nn.Conv2d 创建。这个卷积层的输入通道数是 1（因为输入是单通道的灰度图像），输出通道数是 10，卷积核大小是 5x5。
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # 输入10*12*12 输出20*8*8
        # 最大池化2 输出 20*4*4 = 320
        # 定义了第二个卷积层 conv2，输入通道数是第一个卷积层的输出通道数（即 10），输出通道数是 20，卷积核大小是 5x5。
        self.conv2_drop = nn.Dropout2d()
        # 定义了一个二维 Dropout 层 conv2_drop，用于防止过拟合。Dropout 是一种正则化技术，随机将输入单元的一部分置为零，可以有效降低模型复杂度。
        self.fc1 = nn.Linear(320, 50)
        # 定义了第一个全连接层 fc1，输入大小为 320（经过两个卷积层和池化层后的特征图大小），输出大小为 50。
        self.fc2 = nn.Linear(50, 10)
        # 定义了第二个全连接层 fc2，输入大小为 50（上一层的输出），输出大小为 10（类别数，即 0-9 的十个数字）。
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2)) # 最大池化核为2*2, pool2d是二维池化操作
        # 每一个都是relu函数进行。
        # 执行第一层卷积操作，然后应用 ReLU 激活函数，并通过最大池化层进行下采样。这一层的输出经过池化后尺寸缩小了一半。
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # 执行第二层卷积操作，然后应用 ReLU 激活函数，并通过 Dropout 层进行正则化，最后通过最大池化层进行下采样。
        x = x.view(-1, 320)
        # 将特征图展平为一维向量，以便送入全连接层。
        x = F.relu(self.fc1(x))
        # 执行第一个全连接层的操作，然后应用 ReLU 激活函数。
        x = F.dropout(x, training=self.training)
        # 应用 Dropout 层进行正则化，只在训练时应用。
        x = self.fc2(x)
        # 最后输出了10个数。
        # 执行第二个全连接层的操作，得到模型的输出。
        return F.log_softmax(x) # log( e^x_i / \sum e^x_j )
        # 对模型的输出进行 softmax 操作，并取对数，得到最终的预测结果。softmax 将输出转换为概率分布，对数操作有助于数值稳定性和训练速度。

network = Net()
# 这行代码创建了一个 Net 类的实例，即一个神经网络模型。Net 类是前面定义的包含了卷积层、池化层和全连接层的模型。
optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                      momentum=momentum)
# 这行代码创建了一个 SGD（随机梯度下降）优化器 optimizer。
# network.parameters() 返回模型中所有可学习参数的迭代器。这些参数包括卷积核的权重和偏置项以及全连接层的权重和偏置项。
# lr=learning_rate 设置了学习率，即优化算法在更新参数时的步长。学习率决定了参数更新的速度。
# momentum=momentum 设置了动量参数，用于加速 SGD 收敛。动量算法会在更新过程中考虑之前的梯度方向，并加速收敛。
# 综上，这段代码将创建一个具有随机初始化参数的神经网络模型，并使用 SGD 优化器来更新模型参数，使其逐渐拟合训练数据

train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

def train(epoch):
  network.train()
  # 这行代码将神经网络模型设为训练模式，这对于包含了 Dropout 和 BatchNormalization 等操作的网络来说很重要。
  for batch_idx, (data, target) in enumerate(train_loader):
    '''
    print("print the data")
    print(batch_idx)
    print(target)
    print("print the data shape")
    print(data.shape) #有1000个图片, 目标也是1000个。
    print("end the print the data")
    '''
    optimizer.zero_grad() # 手动将梯度设置为零，因为PyTorch的默认情况下会累积梯度。
    # 这行代码将优化器中的梯度清零，以准备计算新一轮的梯度。
    # data为64个28*28的单通道图像
    # target是64个0-9的数组成的tensor
    # print("data: ", data.shape)
    # print("target: ", target)
    output = network(data)
    # print("output: ", output)
    # 10*64
    # 这行代码将输入数据 data 喂给神经网络模型 network，并得到模型的输出 output。
    loss = F.nll_loss(output, target)
    # output也是64但是是对每一个(10个)的负对数似然损失。
    # print("loss : ", loss)
    # 这行代码计算了模型输出 output 和实际标签 target 之间的负对数似然损失（Negative Log Likelihood Loss）。
    loss.backward()
    # 这行代码执行反向传播，计算模型参数的梯度。
    optimizer.step()
    # 这行代码执行一步梯度下降，更新模型参数。
    if batch_idx % log_interval == 0:
      # 这行代码检查是否达到了日志记录的间隔。log_interval 定义了多少个批次记录一次日志。
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))
      # 这行代码打印训练过程中的日志信息，包括当前 epoch、当前批次的进度、损失值等。
      train_losses.append(loss.item())
      # 这行代码将当前批次的损失值添加到 train_losses 列表中，用于后续的可视化。
      train_counter.append(
        (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
      # 这行代码将当前批次的步数添加到 train_counter 列表中，用于后续的可视化。
      torch.save(network.state_dict(), './model.pth')
      torch.save(optimizer.state_dict(), './optimizer.pth')
      # 这两行代码分别保存了模型参数和优化器状态字典到文件中，以便后续的模型保存和加载。

# train(1)

def test():
  network.eval()
  # 这行代码将神经网络模型设为评估模式。在评估模式下，Dropout 和 BatchNormalization 等操作会被关闭，确保评估时模型的行为和训练时一致。
  test_loss = 0
  correct = 0
  # 这两行代码定义了两个变量 test_loss 和 correct，用于存储测试过程中的损失值和正确预测的样本数量。
  with torch.no_grad():
    # 这行代码包装了测试过程中的代码块，禁用了梯度计算，以节省内存和提高运行速度。
    for data, target in test_loader:
      # 这是一个循环，用于遍历测试数据集中的每个批次。每个批次由 test_loader 生成器提供。
      output = network(data)
      # 这行代码将输入数据 data 喂给神经网络模型 network，并得到模型的输出 output。
      test_loss += F.nll_loss(output, target, size_average=False).item()
      # 这行代码计算了模型输出 output 和实际标签 target 之间的负对数似然损失，并将其累加到 test_loss 变量中。
      pred = output.data.max(1, keepdim=True)[1]
      print("pred[0] : ", pred[0]) # 循环10次，每次1000个
      # 这行代码计算了模型的预测结果 pred。output.data.max(1, keepdim=True) 返回每个样本预测的最大值和相应的索引，而 [1] 表示只获取索引。
      correct += pred.eq(target.data.view_as(pred)).sum()
      # 这行代码统计了模型预测正确的样本数量，并将其累加到 correct 变量中。
      
  test_loss /= len(test_loader.dataset)
  # 这行代码计算了平均测试损失，即所有测试样本的损失的平均值。

  test_losses.append(test_loss)
  # 这行代码将平均测试损失添加到 test_losses 列表中，用于后续的可视化。

  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))
  # 这行代码打印了测试过程的日志信息，包括平均测试损失和测试准确率。
   
# test()

# test()  # 不加这个，后面画图就会报错：x and y must be the same size

for epoch in range(1, n_epochs + 1):
    train(epoch)
    test()
