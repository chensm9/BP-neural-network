# BP神经网络
---
实验参考博客：
* https://blog.csdn.net/xuanwolanxue/article/details/71565934
* https://blog.csdn.net/u014303046/article/details/78200010

---
实验数据来源：
* http://yann.lecun.com/exdb/mnist/

---
## 神经网络设计
网络层数：3，各层的神经元节点个数设计如下：
* 输入层：784个神经元节点（因为MNIST的图像为28*28大小，即784个像素点）
* 隐含层：只有一层，该层设置为100个神经元节点
* 输出层：10个神经元节点（手写数字0-9）


正向传播方向：
* 输入输出函数：
  对每一层
    遍历每一个神经细胞，做如下操作：
    1. 获取第n个神经细胞的输入权重数组 
    2. 遍历输入权重数组每一个输入权重，累加该权重和相应输入的乘积
    3. 将累加后的值通过激活函数，得到当前神经细胞的最终输出
    4. 该输出作为下一层的输入，对下一层重复上述操作，直到输出层输出为止
  

* 激活函数：
  sigmoid（S型）函数，公式为： **f(x) = 1 / (1 + exp(x))**

反向训练方向：
* 训练函数：
  1. 首先输入期望输出，同输出层的输出进行计算得到输出误差数组
  2. 然后对包括输出层的每一层：
    遍历当前层的神经细胞，得到该神经细胞的输出，同时利用反向传播激活函数计算反向传播回来的误差, 进行调整权重矩阵。

* 激活函数：
  sigmoid（S型）函数的导函数，公式为： **f(x) = x * (1 - x)**
  
  
---
## 实验结果
使用mnist数据的60000条训练数据，使用其10000条测试数据进行训练和测试。
测试成功率：93%左右