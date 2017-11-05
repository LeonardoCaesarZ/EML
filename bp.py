import numpy as np
import matplotlib.pyplot as plt

class BP:
    def __init__(self, learn_rate=0.01, iter_num=1000):
        self.model = []                 # 三维数组，仅含隐含层、输出层。1: 层数; 2: 当前层神经元序号; 3: 上一层神经元序号
        self.funcs = []                 # 一维数组
        self.learn_rate = learn_rate    # 训练速度
        self.iter_num = iter_num        # 迭代次数

        plt.ion()

    def add_layer(self, in_num, out_num, init_weight=0.01, init_bias=0.01, activation_func=None):

        # 初始化神经网络模型，即weight、bias组成的二维数组
        layer = np.array([[init_weight for _ in range(in_num+1)] for _ in range(out_num)])
        for cur in layer:
            cur[-1] = init_bias

        self.model.append(layer)
        self.funcs.append(activation_func)
    
    def train(self, x, y):
        model = self.model
        funcs = self.funcs

        # 创建二维数组，记录每层每个神经元的输出，重写于正向传播，使用于反向传播
        n_layers = len(model)
        out = []                        # 每层每个神经元的输出
        out.append(np.copy(x))          # 输入层的输出
        for i in range(n_layers):
            n_units = len(model[i])     # 该层神经元数
            out.append(np.array([0.0 for _ in range(n_units)]))

        iter_num = self.iter_num
        progress = -1
        for i in range(1, iter_num+1): # 训练iter_num次

            self.forward(x, out)    # 正向传播
            self.back(y, out)       # 反向传播

            tmp_progress = int(i * 100 / iter_num)
            if tmp_progress > progress:             # 进度进行每至少1%打印一次当前cost
                progress = tmp_progress
                print("progress: %d %%, cost: %f" % (progress, self.get_cost(x, y, out)))

                plt.cla()
                plt.plot(y)
                plt.plot(out[-1])
                plt.pause(0.001)

    # 正向传播，得到预测序列
    def forward(self, x, out):
        model = self.model
        funcs = self.funcs

        x_ = np.copy(x)
        for i, layer in enumerate(model):   # 遍历神经网络的每一层
            x_ = np.concatenate((x_, np.array([1.0])), axis=0) # because of bias

            for j, wbs in enumerate(layer): # 遍历每层的每个神经元，与每个神经元相关的是分别在前后连接的权重
                tmp = np.dot(x_, wbs)       # 临时变量，每个神经元的输出，下一层的输入。矩阵点乘法则恰好符合运算需求
                if funcs[i] != None:        # 激活函数
                    tmp = funcs[i](tmp)
                out[i+1][j] = tmp
            
            x_ = out[i+1] # +1原因: model数组不包括输入层，out数组包括输入层

    # 计算损失，通过正向传播
    def get_cost(self, x, y, out):
        self.forward(x, out)
        cost = np.sum(np.square(out[-1] - y) / 2)
        return cost

    # 反向传播，更新权重
    def back(self, y, out):

        model = self.model
        funcs = self.funcs
        lr = self.learn_rate

        isOutputLayer = True

        for i in range(len(model)-1, -1, -1):   # 倒序遍历所有神经元层，“反向”为此意
            t = i+1     # 第i层的输出，编号t；第i层的输入，编号i

            y_ = out[t] # 该层的输出
            x_ = out[i] # 该层的输入

            x_ = np.concatenate((x_, np.array([1.0])), axis=0)  # 加上因为bias的特殊处理

            for j, _ in enumerate(model[i]):    # 遍历该层所有神经元

                if isOutputLayer:
                    e = [y - out[-1]]       # 输出层的alter值e部分的计算不同于隐含层，e为随意命名
                else:
                    e = [model[i][j] * g]

                g = funcs[i](y_[j], True) * sum(e)[j]
                alter = lr * (g * x_)       # weight、bias的改变值。(g * x_)为负梯度，利用梯度下降的思想对权重进行更新。
                                            # 默认使用最小二乘为损失函数，若更换损失函数，将相应逻辑更替(g * x_)即可

                model[i][j] += alter
            
            isOutputLayer = False

    # 激活函数sigmoid，反向传播时传入输出序列，正向传播时传入输入序列
    def sigmoid(self, x, isBack=False):
        if isBack:
            return x * (1 - x)
        else:
            return 1 / (1 + np.exp(-x))


if __name__ == "__main__":
    x = np.linspace(1, 10, 30)
    y = (np.sin(x) + 1) / 2
    
    # 训练
    nn = BP(learn_rate=0.01, iter_num=1000)
    nn.add_layer(30, 30, activation_func=nn.sigmoid)
    nn.add_layer(30, 30, activation_func=nn.sigmoid)
    nn.train(x, y)

