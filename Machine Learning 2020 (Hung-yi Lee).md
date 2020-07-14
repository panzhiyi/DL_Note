# Machine Learning 2020 (Hung-yi Lee)

## catalog

[TOC]

## Introduction

#### Learning map

<img src=".\images\image-20200528183646825.png" alt="image-20200528183646825" style="zoom:67%;" />



#### 机器学习就是自动找函式（函数表达式）

各种形式的输入数据-->函式-->各种形式的预测结果，函式是一个任务的解决方法

##### 函式的形式

* regression(回归)，输出形式为一个可以**连续变化**的值
* classification(分类)，输出形式为**有限集合**中的一个值（二分类问题，多分类问题）
* generation(合成)，输出形式为有结构的复杂值（如图片，文本）

##### 怎样找到函式

给定函式的搜寻范围：线性模型，CNN模型，RNN模型，GAN模型等等

根据loss function评价函式

* supervised learning(监督学习)，通过带标签的数据，设计loss function(损失函数)来评价函式对于这个任务的适合程度
* semi-supervised(半监督学习)，只有部分数据带标签
* unsupervised(非监督学习)，所有的数据都不带标签（如生成任务和最经典的聚类任务）
* reinforcement learning(强化学习)，在未标记数据上，只告诉函式是否趋近于目标，而不告诉你如何趋近目标。例如下围棋，只告诉你最终你是输是赢，但不告诉你在棋盘某种状态下如何才能下赢。

函式寻找的方法：Gradient Descent(梯度下降)

## Regression(回归)

##### 什么是regression

regression任务的函式输出是一个可以**连续变化**的值（如预测房价，降雨量）

##### 怎样解决regression任务

举例：预测宝可梦升级后的战斗能力值（cp值）

> step1 定义函式的集合——model

假设我们找的function set的形式为$y=b+w*x$，也可写成**线性模型**$y=b+\sum w_ix_i$的形式

其中我们称$b$为bias，$w_i$为weight，$x_i$为feature

> step2 确定评价函式好坏的函数——loss function

training data: $(x^i,\hat y^i)$

定义一个loss function $L$，它的输入为一个函式$f$，用来评价这个函式$f$。具体来说，
$$
L(f)=L(b,w)=\sum_{n=1}^{N}(\hat y^n-(b+w*x^n))^2
$$

> step3 根据$L$挑选最好的$f$——gradient descent

$$
f^*=arg\underset {f}{\operatorname {min} }\,L(f)
$$

$$
w^*,b^*=arg\underset {w,b}{\operatorname {min} }\,L(w,b)
$$

Gradient Descent:（b同理）

* 随机选取w的初始值$w^0$
* 计算w在$w^0$位置对于损失函数L的梯度$\frac{dL}{dw}|_{w=w^0}$，并根据梯度更新w的值$w^1=w^0-\eta\frac{dL}{dw}|_{w=w^0}$。
  我们向梯度方向的负方向增加w，所以能够使得L减小。$\eta$为learning rate，控制梯度下降的速度。
* 重复操作，使得L降到local optimal

##### 选择复杂的函式——overfiting

当我们选择复杂函式(model)的时候，train data上的error通常会减小，test data上的error通常会先减小后增大。

<img src=".\images\image-20200528205350250.png" alt="image-20200528205350250" style="zoom: 50%;" />

overfiting:在训练集上表现比测试集上的表现好很多的现象

##### 解决overfiting——regularization

重新设计损失函数
$$
L(f)=\sum_{n=1}(\hat y-(b+\sum w_ix_i))^2+\lambda\sum(w_i)^2
$$
$\lambda\sum(w_i)^2$作用是使模型参数越小越好，即模型变化平滑。w越小，对输入x的改变越不明显，模型f越平滑。这样有噪声的时候，对模型的影响会变小。

不需要考虑bias，因为bias的大小并不影响模型的平滑程度。

## Basic Concept——Where does the error come from?

##### Bias

由于挑选的model过于简单，导致简单的model无法完美的解决任务，所以导致了误差error。

##### variance

由于挑选的model过于复杂，导致数据量不足以挑选出最好的模型，挑选的都是model集合中次优的model，所以导致了误差error。

##### validation data

根据test data挑选的模型，通常在train&test以外的其他数据上表现不如test data。这是因为你的模型是根据在test data上的表现挑选出来的。所以为了真实的反映模型的水平，会从train data中随机挑选一部分作为validation data，以模型在validation data上的表现挑选模型，然后在test data上观察模型真实的表现。

## Gradent Descent

#### Tuning your learning rate

太大太小都会出现问题：太大-->难以收敛，太小-->收敛速度慢

<img src=".\images\image-20200528231757191.png" alt="image-20200528231757191" style="zoom:50%;" />

##### 动态调整学习率

随epoch的增加，learning rate减小

* 一开始远离目标，可以快一些；当接近目标时，降低learning rate的值
* decay: $\eta^t=\eta/\sqrt{t+1}$

给不同的参数不同的学习率——Adagrad

##### Adagrad

$$
w^{t+1}=w^t-\frac{\eta^t}{\sigma^t} g^t
$$

$\sigma^t$为之前所有算出梯度的均方根

#### SGD(Stochastic Gradient Descent)

- [ ] Vanilla Gradient descent：计算所有样本在当前模型上的平均梯度，进行一次梯度更新（准确但缓慢）

- [x] Stochastic Gradient Descent：计算一个样本在当前模型上的梯度，进行一次梯度更新（不准确但快速）

原因：由于目前限制模型精度的仍然还是算力，所以提高速度比准确更为重要。【个人理解】

#### Feature Scaling

将feature的所有维度都归一化为均值为0，方差为1.

<img src=".\images\image-20200528234147365.png" alt="image-20200528234147365" style="zoom:50%;" />

原因：减少feature中某一维度因为均值方差太大导致梯度太大对结果的影响，使得梯度下降的方向能够从一开始就指向local optimal的方向，少走弯路。

#### Theory

如何在f中的某一位置的邻域中找到使得L(f)最小的点？

Taylor Series(泰勒展开)：$h(x)\approx h(x_0)+h'(x_0)(x-x_0)$，当x足够接近$x_0$时。

同理，MultiVariable Taylor Series： $h(x,y)\approx h(x_0,y_0)+\frac{\partial h(x_0,y_0)}{\partial x}(x-x_0)+\frac{\partial h(x_0,y_0)}{\partial y}(y-y_0)$，当x,y足够接近$x_0$,$y_0$时。

如果邻域范围足够小，我们会发现，领域中最小的点就位于$h(x_0,y_0)$的**梯度方向的负方向的邻域边界**上，这就和Gradient Descent的公式和思路一致。

#### Limitation

<img src=".\images\image-20200529000352182.png" alt="image-20200529000352182" style="zoom:50%;" />

梯度下降可能停在高原，鞍点和局部最小点。【后面会讲到在深度学习中，这些点很难出现，因为要满足所有参数都同时梯度接近0】

### Classification(分类)

##### 什么是classification

classification任务的函式输出是一个**有限集合里的某一个值**（如是否贷款给某人，医疗诊断，手写文字辨识，人脸辨识）

##### 用regression的方法做classification

预测的值接近1-->class 1;预测的值接近2-->class 2……这样来做classification可以吗？×

* 会惩罚那些**过于正确**的点
* 对将class的序号和序号的数值强行学习一个对应关系

<img src=".\images\image-20200529220538952.png" alt="image-20200529220538952" style="zoom:50%;" />

##### 利用朴素贝叶斯分类器来做分类

$x$属于类别$c_1$的几率$P(c_1|x)$为
$$
P(c_1|x)=\frac{P(c_1)P(x|c_1)}{P(c_1)P(x|c_1)+P(c_2)P(x|x_2)}
$$


当$P(c_1|x)>0$时，$x$最有可能属于$c_1$。

> 那如何求得$P(c_1|x)$呢？

通过training data，我们可以估测出$P(c_1)$,$P(c_2)$,$P(x|c_1)$,$P(x|c_2)$这四个值。

其中$P(c_1)$,$P(c_2)$为先验概率(prior)可以直接计算。

> 那怎样计算类别$c_1$出现$x$的概率$P(x|c_1)$呢？

我们可以通过$c_1$已有的数据估测$c_1$的概率分布，然后知道分布后将x带入概率分布就可以计算$P(x|c_1)$了。

我们假设$c_1$符合高斯概率分布，那么要确定的就是高斯概率分布的均值$\mu$和方差矩阵$\sum$。

> 那如何利用已有数据来预测$\mu$和$\sum$呢？

**最大似然估计**：每一组$\mu$和$\sum$决定的高斯分布都有可能采样出training data中的$c_1$分布的数据，但是这些高斯分布采样出这些数据的概率有大有小，我们希望从所有的高斯分布中找到**最有可能**采样出training data中的$c_1$分布的数据的那一个高斯分布，由于在概率学中学过，其结论为，当$\mu$等于数据的平均值，$\sum$等于数据的方差的高斯分布是最有可能产生这样数据的高斯分布。

如果数据x的所有维度都是独立的，那么这种分类方法称为朴素贝叶斯分类器。

##### 对$P(c_1|x)$的分析

<img src=".\images\image-20200529225058088.png" alt="image-20200529225058088" style="zoom:50%;" />

$P(c_1|x)$可以写成$\sigma(z)$的形式，其中$\sigma$为sigmoid函数，z最终可以化成$z=w\cdot x+b$的形式。在朴素贝叶斯分类器中，能够直接计算出w和b（生成式模型）。

最后我们发现，如果可以通过某种方法直接预测w和b（判别式模型），类似与regression中做的那样，再对z进行sigmoid，就能得到分类的结果，这样的方法叫做**逻辑回归**。

##### 生成式&判别式分类器

生成式分类器：通过已有数据预测分布，可以根据预测到的分布自我生成新的数据

判别式分类器：只对已有数据进行判别，未知数据都只根据已有数据来判断其类别

### Logisitic Regression(逻辑回归)

#### 求解逻辑回归

##### Function Set

现在我们选择的函数集合为$f_{w,b}(x)=\sigma(w\cdot x+b)$来解决分类任务(x和w都是向量)，

二分类时，当$\sigma(w\cdot x+b)>0.5$时，$x$属于类别$c_1$，否则属于类别$c_2$。

用图表示$\sigma(w\cdot x+b)$：

<img src=".\images\image-20200529232839851.png" alt="image-20200529232839851" style="zoom:50%;" />

##### Loss Function

还是利用极大似然估计的思路设计loss function：
$$
L(w,b)=f_{w,b}(x^1)f_{w,b}(x^2)(1-f_{w,b}(x^3))\cdots f_{w,b}(x^N)
$$
希望找一组w和b使得L(w,b)最大：
$$
w^*,b^*=arg\underset {w,b}{\operatorname {max} }L(w,b)=arg\underset {w,b}{\operatorname {min} }-ln(L(w,b))
$$
其中，$-ln(L(w,b))$可以表示为
$$
-ln(L(w,b))=-(ln(f_{w,b}(x^1))+ln(f_{w,b}(x^2))+ln(1-f_{w,b}(x^3))\cdots +ln(f_{w,b}(x^N)))
$$

$$
=-\sum_n\,[\hat y^nlnf_{w,b}(x^n)+(1-\hat y^n)(1-lnf_{w,b}(x^n))]
$$

这里就是cross-entropy loss，它衡量了$\hat y$和$f_{w,b}(x^n)$的**分布相似程度**。

##### Find the best function

利用梯度下降，每一步下降的步长为：
$$
\frac {-ln(L(w,b))}{\partial w_i}=\sum_n-(\hat y^n-f_{w,b}(x^n))x^n**
$$

#### 对比

<img src=".\images\image-20200530010621779.png" alt="image-20200530010621779" style="zoom:50%;" />

为什么不使用logistic regression + mean square error?

因为经过梯度计算，发现这样设计loss function，在远离目标的地方梯度接近于0。

#### 怎样做多分类

<img src=".\images\image-20200530011634408.png" alt="image-20200530011634408" style="zoom:50%;" />

对所有分类的输出做softmax将其转化成概率。
$$
softmax:y_i=e^{z_i}/\sum_{j}^ne^{z_j}
$$

#### 逻辑回归的限制

看似强大的逻辑回归却无法解决简单的异或问题

class 1:(0, 0), (1, 1)	class 2:(0, 1), (1, 0)

<img src=".\images\image-20200531014252237.png" alt="image-20200531014252237" style="zoom: 50%;" />

因为逻辑回归的分界还是一个线性的，所以无法区分异或问题这样的数据。

但是我们可以找一个feature transformation将输入特征转换成另一组特征，比如我们设定转换规则为$x'_1$为点到(0,0)的距离，$x'_2$为点到(1,1)的距离。这样我们可以获得新的一组特征为：

<img src=".\images\image-20200531015107382.png" alt="image-20200531015107382" style="zoom:67%;" />

这样我们就可以轻松的分开异或的数据，然而很多时候我们不知道如何寻找transformation，我们同样可以将transformation表达为逻辑回归，通过梯度下降的方法来学习transformation

<img src=".\images\image-20200531015313549.png" alt="image-20200531015313549" style="zoom:50%;" />

当我们将一个又一个的逻辑回归连接组合之后，将逻辑回归换一个叫法Neuron，我们就获得了一个**神经网络**。所以我们可以看到神经网络的本质就是将输入特征（输入层）经过层层的特征转换（隐藏层），最后得到一个能被逻辑回归（输出层）处理的特征。

## Brief Intorduciton of Deep Learning

#### 怎样进行深度学习呢？

和机器学习的步骤一样：定义模型，loss函数，梯度下降

#### 定义模型

通过对神经元不同的连接方式，我们可以定义多种多样的神经网络模型

##### Full Connected Feedforward Network (FCN，全连接前馈网络)

全连接：每一层每一个神经元的输出都作为后一层每一个神经元的输入

<img src=".\images\image-20200601160125532.png" alt="image-20200601160125532" style="zoom:50%;" />

为什么可以使用**GPU加速**？GPU是针对矩阵运算做的优化，而神经网络的计算正好是矩阵运算。

<img src=".\images\image-20200601160416549.png" alt="image-20200601160416549" style="zoom:50%;" />

#### 定义loss function

不同的任务定义不同的loss function，比如分类任务时同逻辑回归的loss function，cross-entropy loss

#### 梯度下降——（back propagation）反向传播算法

反向传播算法是有效的计算网络中每一个参数的梯度的方法（所以可以不用掌握的太明白，只需要知道是一种适用于深度网络的梯度下降方法）

链式法则：

* $y=g(x)$，$z=h(y)$
  $$
  \frac{dz}{dx}=\frac{dz}{dy}\frac{dy}{dx}
  $$
  
* $x=g(s)$，$y=h(s)$，$z=k(x,y)$
  $$
  \frac{dz}{ds}=\frac{\partial z}{\partial x}\frac{dx}{ds}+\frac{\partial z}{\partial y}\frac{dy}{ds}
  $$

对于loss function C，某个神经元的w对于C的偏导为（z是该神经元的输出）
$$
\frac{\partial C}{\partial w}=\frac{\partial z}{\partial w}\frac{\partial C}{\partial z}
$$
其中计算所有神经元w对z的偏导$\frac{\partial z}{\partial w}$的过程称为前向传播，计算所有$\frac{\partial C}{\partial z}$的过程叫反向传播

* 前向传播：$\frac{\partial z}{\partial w}=x$，其值等于该神经元的输入值x

* 反向传播：我们如果假设该神经元后连接的下一层所有神经元的$\frac{\partial C}{\partial z}$已知，那么有

  <img src=".\images\image-20200601164813950.png" alt="image-20200601164813950" style="zoom:50%;" />

  我们将网络的结果输入输出反向，然后将$\frac{\partial C}{\partial z'}$和$\frac{\partial C}{\partial z''}$作为反向网络的输入进行计算就可以算出$\frac{\partial C}{\partial z}$的值了，这就是反向传播了

  <img src=".\images\image-20200601164918933.png" alt="image-20200601164918933" style="zoom:50%;" />

## Why Deep?

有理论证明任何连续的函式，都可用一层的隐藏层的网络来表示，那么为什么网络要变深而不是变胖？

为了公平的比较，我们使用同样参数量的深瘦网络和浅胖网络作比较，发现深瘦网络性能优于后者。

简单的理解原因：深层的网络能够将学习任务模块化，可以更高效的学习特征。

**end-to-end learning(端到端的学习)**: 只要给输入和输出，函式怎样去做是自动学习出来的而不需要人为指定

## Tips for Deep Learning

当你发现深度学习的结果不好的时候，可能的原因和对应的解决方法：

#### Gradient Vanish(梯度消失)

靠近输入层的参数梯度很小，靠近输出层的参数梯度正常，导致前层的参数并没有被充分训练到。

因为如果使用sigmoid激活函数，那么sigmoid的导数范围是0-1，由于网络很深，经过层层的乘法，前层的导数会越来越接近0.

解决办法：使用ReLU或者Maxout激活函数来代替sigmoid激活函数

##### ReLU

<img src=".\images\image-20200601170628770.png" alt="image-20200601170628770" style="zoom:67%;" />

优点：计算快，符合生物观察，解决梯度消失问题

##### Maxout

learnable activation function

<img src=".\images\image-20200601171104373.png" alt="image-20200601171104373" style="zoom:50%;" />

#### 局部最小值

梯度下降会让结果落入局部最小值

解决办法：调整梯度下降算法的方案

##### Momentum(动量)

考虑上一次梯度的影响，然后影响这一次的参数更新

#### Overfitting(过拟合)

过度拟合训练数据，而导致在测试数据上的性能降低

##### Early Stopping(早停)

在训练过程中刚要出现过拟合的情况时，即使停止训练。

##### Regularization(正则化)

> 加入l2正则项的loss function为：

$$
L'(\theta)=L(\theta)+\lambda\frac{1}{2}||\theta||_2
$$

其梯度为：
$$
\frac{\partial L'}{\partial w}=\frac{\partial L}{\partial w}+\lambda w
$$
那么参数更新公式变为：


$$
w^{t+1}=w^t-\eta(\frac{\partial L}{\partial w}+\lambda w^t)=(1-\eta\lambda)w^t-\eta\frac{\partial L}{\partial w}
$$
所以$1-\eta\lambda$是一个接近于1的数字（一般为0.99），为weight decay。

> 加入l1正则项的loss function为：

$$
w^{t+1}=w^t-\eta(\frac{\partial L}{\partial w}+\lambda sgn(w^t))=w^t-\eta\frac{\partial L}{\partial w}-\eta\lambda \, sgn(w^t))
$$

##### Dropout

训练过程中，每次训练都随机丢弃p%的神经元，使得网络变瘦变简单。虽然某些参数在本次训练中由于丢弃而未被更新，但是在整个训练过程中仍然会被训练。在测试过程中，用整个网络来做测试而不丢弃神经元，同时将参数都乘上（1-p%）。

可以用集成学习的思想来理解dropout为什么有效。

## Convolutional Neural Network(CNN)

卷积神经网络(CNN)是专门针对图像数据设计的网络结构。

如果我们采用FCN来处理图像，由于图像分别率很高的原因导致输入数据的维度特别高，所以FCN中的参数数量会太大。

CNN实质上就是从FCN中删掉某些多余的神经元连接，使网络在处理图像数据时更加高效。(训练层面的提高)

为什么CNN能在删掉神经元连接的情况下依然有效呢？(结果层面的提高)

* 底层的网络如果说学习到简单的pattern的话，通常对于图片简单的pattern覆盖的区域只是图片的一小部分（可以做**局部**卷积）
* 在图像中pattern通常在图像的各个位置会重复出现，所以可以用同一种学习的pattern来检测图像中所有的位置（可以做**参数共享**）

#### CNN的网络架构

<img src=".\images\image-20200602100415006.png" alt="image-20200602100415006" style="zoom:67%;" />

##### Convolution

考虑这样的操作，用某一大小的filter与图像上相同大小的局部做内积，在固定的步长下，我们可以得到一张新的feature map。

对于$m_1*m_2$大小的图像，c个$n*n$大小的filter，固定步长s下，得到的新的feature map的大小为$((m_1-n)/s)*((m_2-n)/s)*n$.

**padding**：如果想要保持输入feature map和输出feature map的大小一致，需要在周围补0，叫做padding。

这样的卷积操作可以完美的用网络来构造出来，通过网络来学习filter的参数。由于局部卷积和参数共享，使得网络的参数量大大减少。

##### Pooling(池化)

将n*n大小里的值选一个最大的作为输出，叫做max pooling；平均后作为输出，叫做average pooling.

##### Flatten

将feature map展成一维向量，作为后面全连接层的输入。

## Recurrent Neural Network(RNN)

循环神经网络是针对时序数据设计的。

假设我们去理解一句话的含义，如果每次输入数据只是一个词，那么网络就无法理解句子中词与词之间的含义。如果我们将一句话作为输入，那么长度不定以及词与词之间的顺序网络难以理解。

但是如果我们使用网络中的值用来表示网络读取前一次数据时的状态，当前的输入包括输入值+状态值，那么我们就能够合理的处理时序数据了。

#### simple version

<img src=".\images\image-20200602110355658.png" alt="image-20200602110355658" style="zoom:67%;" />

#### LSTM(Long Short-term Memory)

<img src=".\images\image-20200602111312857.png" alt="image-20200602111312857" style="zoom:67%;" />

输入控制阀门，忘记控制阀门，输出控制阀门。我们用上述的结构来代替原来网络中的神经元。

## Semi-supervised Learning

训练数据中，只有一小部分数据有label，绝大部分只有数据却没有label。

为什么semi-supervised learning是对任务有用的？

unlabeled data 的分布可以告诉我们有用的信息。

#### Semi-supervised Generative Model

类似于EM&k-mean算法的思想，先利用labeled data估计出分布，然后根据分布来估计每个unlabeled data属于不同类别的概率，然后再根据算出的概率重新和labeled data一起更新分布，重复操作。这样不再是最大似然labeled data估计，而是最大似然labeled data + unlabeled data估计。

#### Low-density Separation

我们假设所有数据在分界处的数据密度分布最低

##### self-training

使用labeled data训练出model f，然后用f在unlabeled data上获得pseudo-label，然后将一小部分unlabeled data加入到labeled data set中重新训练model f.

与Semi-supervised Generative Model相比，self-training使用了hard label，而Generative Model使用了soft label(概率)。

对于逻辑回归来说，我们只能采用hard label的形式，因为soft label对model的更新没有影响。

#### Entropy-based Regularization

我们假设分类输出的结果是越集中越好

那么我们可以计算输出结果的entropy(衡量数据分布的集中程度)作为训练的loss function的一项。
$$
L=\sum_{x^r}C(y^r,\hat y^r)+\lambda\sum_{x^u}E(y^u)
$$
前一项为对labeled data的cross entropy loss，后一项为对unlabeled data的entropy loss.

#### Smoothness Assumption

我们假设越相邻的数据他们是同一种label的可能性越大

更精确的：

* x的分布是不平均的（某些地方集中，某些地方分散）
* x1和x2在密集区域靠近，那么$\hat y^1$和$\hat y^2$是一样的

其实考虑的是x1和x2的**测地距离**的远近。

#### Graph-based Approach

建图，如果x1和x2足够接近，那么在x1和x2点之间连上一条边，相似度为这条边的权重。

那么在同一个连通图的数据是同一种label，即label通过图结构传播。

![image-20200603135622856](.\images\image-20200603135622856.png)

##### Smoothness of labels

$$
S=\frac{1}{2}\sum_{i,j}w_{i,j}(y^i-y^j)^2
$$

这个值越小，越smoothness。

##### 拉普拉斯矩阵

这个很有意思的一点，我之前做的显著性区域检测，就是先找种子点，再对种子点进行扩散，用的就是拉普拉斯矩阵。和半监督学习中很相似，种子点就是带label的数据，对种子点进行扩散就是使用拉普拉斯学习未标记数据的label.

![image-20200603140131324](.\images\image-20200603140131324.png)

##### loss function for Graph-based Approach

$$
L=\sum_{x^r}C(y^r,\hat y^r)+\lambda S
$$

## Explainable Mechine Learning

模型要对给出的结果做解释（有现实意义）

* local explaination: 模型为什么判定该图片是猫？针对某一例子
* global explaination: 模型认为猫应该长什么样子？针对所有例子

#### Local explaination

将object x分为N个components{x1,x2,…,xN}。对于不同object，components不同，对于图像是像素超像素，对于语音可以是一个单词。

我们将object的某个component拿掉或改动后再让模型判断，如果对结果影响大，那么就认为该component是模型判断的关键。

component粒度也是要动态选择的，太小对结果影响就都很小，太大对结果影响就都很大，不具有分辨性。

**利用梯度**：可以对每个输入xi计算对yk的梯度，表示输入对输出的影响大小，也反映了判断出yk的在输入上的位置。（CAMs）

#### Global explaination

##### Activation Maximization

$$
x^*=arg\underset {x}{\operatorname {max} }\,y_i
$$

我们将网络的参数固定而为了使yi变大优化输入x，那么我们就得到该网络输出最大yi对应的输入，我们就知道网络判断出yi时认为的输入长的是什么样子。

然而如果只用第一个式子，输出的内容通常人类无法理解。

* R(x)是人为定义的衡量x多像有意义的图像的值来做正则化

$$
x^*=arg\underset {x}{\operatorname {max} }\,y_i\,+R(x)
$$

* 使用生成器做“正则化”

#### Local Interpretale Model-Agnostic Explanations(LIME)

用某一个可解释的model来解释不可解释的model（用可解释model对不可解释的model做**局部拟合**）

<img src=".\images\image-20200603194649099.png" alt="image-20200603194649099" style="zoom:67%;" />

## Attack and Defense

#### Attack DNN

我们对网络能够分类正确的数据加入一些细小的噪声，使得网络将新的数据分类错误。

##### Loss function for attack

Trainning:							$L_{train}(\theta)=C(y^0,y^{true})$								希望输出label和真实label越近越好

Non-targeted Attack:		$L(x')=-C(y',y^{true})$									希望输出label和真实label越远越好

Targeted Attack:				$L(x')=-C(y',y^{true})+C(y^0,y^{false})$		希望输出label和真实label越远越好同时和假label越近越好

Constraint:						  $d(x^0,x')\leq\varepsilon$												   希望噪声输入和原输入越像越好

##### How to Attack

$$
x^*=arg\underset {d(x^0,x')\leq\varepsilon}{\operatorname {min} }\,L(x')
$$

如果没有限制，那么我们通过梯度下降，固定网络的参数，对loss function来修改输入x；由于有限制，每次梯度下降之后我们判断一下当前的输入x是否满足限制$d(x^0,x')\leq\varepsilon$，如果不满足，那么需要对当前的输入x进行修正，使得x满足限制。

##### Why Attack happened?

<img src=".\images\image-20200603203706493.png" alt="image-20200603203706493" style="zoom:67%;" />

在某种通过梯度下降寻找的方向，在这个方向上非常不鲁棒。

White Box v.s. Black Box

前面提及的攻击都是已知网络参数$\theta$的前提下，称为White Box Attack。那如何进行Black Box Attack呢？

##### Black Box

使用同样的网络结构（甚至不同网络结构）和同样的训练数据自己训练一个proxy network，然后拿proxy network代替black network进行攻击同样有效。

#### Defense

##### Passive defense

不修改网络的参数，是一种Anomaly Detection(是否能检测未知数据)

* 在输入时加入filter，对输入进行平滑处理噪声。
* 对输入进行缩放或padding

实际上如果attack时已知上述操作，那么可以针对特定处理再进行attack

##### Proactive defense

训练时就考虑到要对攻击图片鲁棒

多次训练：得到模型后，自我攻击，再重新训练。

## Network Compression

#### network pruning(网络剪枝)

网络通常时过参数化的(over-parameterized)，所以可以进行prune

<img src=".\images\image-20200603211412339.png" alt="image-20200603211412339" style="zoom: 50%;" />

##### why pruning?

为什么不直接训练一个小的网路，而是先训练大的再剪枝呢？因为大的网络更容易被优化。

##### prune weight or neuron?

prune neuron优于weight，因为在实际操作的时候，为了补齐，prune的weight都固定为0，并没有减少网络参数。

#### Knowledge Distillation

训练大的网络作为teacher network，然后用小的网络作为student network来模拟teacher network。让student network学习teacher network的输出。

因为teacher network相比原来的label给了更多的信息。比如原来手写数字辨识输入图片1的例子，原来label为1，但是teacher network给的label为1的可能性为0.7，7的可能性为0.2，9的可能性为0.1，意味着透露出1和7，9是比较相似的。

#### Parameter Quantization

对网络的参数进行量化：

使用更少的位来表示参数

对参数进行聚类，用同类的参数的平均值来代替参数的值

##### binary weight

将网络的参数只是用+1，-1来离散表示，梯度更新就从一组binary weight到另一组binary weight.

发现会使网络更好，原因是类似于正则化的效果。

#### Architecture Design

被认为是实际操作中最有效的方式。

##### Low rank approximation

低秩近似，在层与层之间加入一层来减少参数量，拿全连接举例，原来的需要W=M\*N个参数，现在需要U+V=M\*K+N\*K=(M+N)*K个参数，如果k<<M,N，那么就能达到降低网络参数的目的。

<img src=".\images\image-20200604101152628.png" alt="image-20200604101152628" style="zoom: 67%;" />

#### Dynamic Computation

网络是动态计算的，可以视情况选择复杂还是简单的网络。

* 训练多个不同的网络，缺点是要存储多个网络
* 对不同层的特征都直接接一个分类器做分类，选择不同深度的层能做到不同结构的网络（这种可能会损害到原来网络的性能）

## Unsupervised Learning

非监督学习可以分成两大类：

* Clustering&Dimension Reduction(化繁为简)
* Generation(无中生有)

### Clustering(聚类)

K-means

Hierarchical Agglomerative Clustering(HAC，聚合层次聚类)：按相邻程度建树

### Dimension Reduction(降维)

数据在某些维度上是冗余的。

#### Feature selection

直接拿掉某些维度，保留剩余维度

#### Principle component analysis(PCA，主成分分析)

线性降维$z=Wx$，我们有两种不同的解释，但都对应着PCA

* **最大投影方差：**PCA希望能找到一个投影矩阵W，将高维数据x投影到低维空间中得到z，希望投影后的z的方差越大越好（希望z分散，便于分类）。

* **最小重构代价：**PCA希望找到低维的空间，使得在低维空间的x的投影z，与高维的x的误差越小越好（投影后对离原数据的改变最小）

PCA其实可以看成一层的auto-encoder的最优解。

#### Neighbor Embedding

非线性降维，可以保持点与点的测地距离关系

##### Locally Linear Embedding(LLE)

在高维空间中的$x_i$可以用其相邻的几个点$x_j$近似线性表示，其线性表示的参数为$w_{ij}$；LLE希望在降维后的$z_i$也可以用$z_j$表示，其参数和高维空间的参数$w_{ij}$相同。

##### Laplacian Eigenmaps

类似于半监督学习的图方法，根据点在高维空间的距离建图，希望降维后的z在空间中越smooth越好，而且降维后的z要span满整个低维空间，不然只有前面smooth的限制，z就全部等于0了。

##### T-distributed Stochastic Neighbor Eembedding(t-SNE)

前面的方法值要求相近的点接近，但未要求远离的点远离。

计算所有高维点对的相似性$S(x^i,x^j)$，然后做normalization
$$
P(x^j|x^i)=\frac{S(x^i,x^j)}{\sum_{k\ne i}S(x^i,x^k)}
$$
同理我们可以得到降维后的normalization后的相似性$Q(z^j|z^i)$，我们希望两者相似性的分布尽可能相同，即
$$
L=\sum_i KL(P(*|x^i)Q(*|z^i))
$$
越小越好。

> SNE&t-SNE

高维相似性：$S(x^i,x^j)=exp(-||x^j-x^i||_2)$

低维相似性：

* SNE:$S'(z^i,z^j)=exp(-||z^j-z^i||_2)$
* t-SNE:$S'(z^i,z^j)=1/(1+||z^j-z^i||_2)$

#### Auto-encoder

<img src=".\images\image-20200604141849443.png" alt="image-20200604141849443" style="zoom:67%;" />

可以做pre-training DNN，不仅可以做压缩而且可以做升维。

##### Auto-encoder for CNN

<img src=".\images\image-20200604143814438.png" alt="image-20200604143814438" style="zoom: 67%;" />

> Unpooling

unpooling需要在之前pooling时额外记得每一个最大值是从哪里获得的。然后unpooling时把值放在纪录的位置，然后其他位置补0。

或者直接将unpooling的值复制给全部区域（keras的做法）

> Deconvolution

实际上，deconvolution就是convolution。

<img src=".\images\image-20200604144615233.png" alt="image-20200604144615233" style="zoom:67%;" />

padding后做convolution就是deconvolution了。

#### More for Auto-encoder

##### More than minimizing reconstruction error

###### Using Discriminator

Discriminator: 输入图像和encoder学到的对应code，输出两者是否匹配。

Discriminator的loss function为$L_D$，参数是$\phi$，其训练结果为$L^*_D=\underset{\phi}{min}\,L_D$；Encoder的参数为$\theta$，那么训练Encoder的公式为$\theta^*=\underset{\theta}{min}\,L^*_D$.

那么经典的auto-encoder其实其实就是使用Discriminator的一个特例。

###### Sequential Data

如果是序列数据，那么输出的形式就会多种多样，导致auto-encoder变化多样。

##### More interpretable embedding

###### Feature Disentangle(特征分离)

###### Discrete Representation(离散表示)

将code表示为one-hot或者binary的形式，解读就更加容易。

VQVAE: 使用codebook，让encoder学习的code与codebook中相似的vector来进行decoder，这样更容易让网络学习到输入的共性而不是输入的特性。

### Generation

#### PixelRNN

初始一张图像的第一个像素，输入RNN来预测下一个像素，然后将前面的所有像素重新作为输入预测下一个像素。

这样操作如果指定了第一个像素的值，输出图像就确定了，所以每次预测下一个像素时都让模型有概率选择不是最有可能出现的那个像素，这样保证随机性。

#### Variational Auto-Encoder(VAE)

<img src=".\images\image-20200604212413368.png" alt="image-20200604212413368" style="zoom:67%;" />

使用VAE相比PixelRNN来说，能够控制生成数据，我们固定code的值，只改变某几维的值，能够通过生成数据的改变理解着几维的含义。

##### Why VAE?

为什么使用VAE呢，是因为通常使用普通的AE的decoder生成数据的时候，结果不好。我们希望说在code和code之间的code对应的图像是介于图像和图像之间的，VAE通过对code增加噪声使code周围的一部分code都生成类似的图像。

<img src=".\images\image-20200604213144701.png" alt="image-20200604213144701" style="zoom:67%;" />

如下图，$\sigma$表示噪声的方差，exp保证噪声是正数影响，而$m$是我们原来的code。如果只用重构误差限制，那么网络将不学习噪声部分，所以需要添加loss function控制$\sigma$的值不能太小。

<img src=".\images\image-20200604213624362.png" alt="image-20200604213624362" style="zoom:67%;" />

#### Generative Adversarial Network

两个网络对抗学习，一个生成网络G任务是创造以假乱真的图像，一个描述网络D任务是判断一张图像是不是人造的。

一开始G生成的数据很差，D很容易分辨，直到G生成的数据使得D无法分辨为止，那么G就被训练好了。由于G从来没有看过真实的数据，所以产生的数据都是真实数据中不存在的。

G的输入是一个随机的向量，输出是一张图像；D的输入是G生成的图像或者真实存在的图像，输出是0/1表示是生成/真实的图像。

<img src=".\images\image-20200605002027292.png" alt="image-20200605002027292" style="zoom:67%;" />

## Anomaly Dectection

training data$\{x^1,x^2,…,x^N\}$，我们需要找到一个函式检测输入的x时候与training data相似。

所以异常数据只是相对于训练数据来讲的。

直接获得一组正常数据和一组异常数据去训练一个二分类器？

异常检测无法视为二分类的问题，因为异常数据太多无法穷举，所以无法将异常数据视为一种类别；有些情况异常数据难以获得。

#### with labels

但是没有一种label为“unknown”类。

##### 算法

根据labels创建一个分类器，而且希望分类器同时输出判断该分类的信心分数c，然后根据c的大小来判断输入是否是异常数据。

confidence: the maximum scores / negative Entropy

##### 如何判断Anomaly Dectection的好坏？

evaluation Set: Images x and Label of x is from training data labels or not.

异常数据和正常数据通常数量悬殊，所以简单的使用正确率来评价Anomaly Dectection的好坏。

我们可以设置cost来对正常数据判断到异常和异常判断到正常的错误平衡。

可能的问题：

可能存在比原来数据更具有特征的数据（如分类猫和狗的任务，猫和狗是正常数据，但是老虎和狼相对于猫和狗来讲更具有区分性，所以这样的数据放入分类器会产生更强的信心分数）

#### without labels

##### polluted:

假设多数训练数据都是正常数据，只有一小部分训练数据是异常数据。

算法：学习一个概率分布$P(x)=f_\theta(x)$来拟合所有的训练数据，当$P(x)>\lambda$时x是正常数据，反之是异常数据。

根据极大似然估计算出$\theta^*$，然后就能算出所有数据产生的概率，那些产生概率小的数据就是异常数据。

##### clean

训练数据都是正常数据，没有异常数据。

利用auto-encoder，输入为训练数据，当新的数据输入该auto-encoder时，正常数据会很容易被还原，而异常数据难以被还原。

## Meta Learning(元学习)

learn to learn，从其他学习任务中学习到如何学习的技巧。

##### 与machine learning的对比

<img src=".\images\image-20200605161337561.png" alt="image-20200605161337561" style="zoom:67%;" />

怎样找到F呢，和找f的步骤是一样的（只不过F是找一个f的函数，而f是找一个model的函数）。

<img src=".\images\image-20200605161543491.png" alt="image-20200605161543491" style="zoom:67%;" />

##### Define a set of learning algorithm

<img src=".\images\image-20200605162112896.png" alt="image-20200605162112896" style="zoom:67%;" />

灰框中所有的步骤都是都是一个learning algorithm F，实际上之前的深度学习的步骤都是人为设计的（比如红框中的网络架构，初始参数，梯度更新方法）。如果我们采用不同的网络架构，不同的初始化的参数值，不同梯度更新方法就会有一组learning algorithms。

##### Define the goodness of a function F

我们使用一个特定训练任务通过训练集的表现评价f，我们用多个训练任务的多个数据集的表现来评价F。
$$
L(F)=\sum^N_{n=1}l^n
$$
其中N表示有N个任务，$l^n$表示在第n个任务上测试集的loss。

F的**训练资料(Support set)**是多个任务，每个任务里面有训练数据和测试数据，**测试资料(Query set)**是不同于训练任务的其他任务；f的训练资料是某个特定任务的训练集，测试资料是该任务的测试集。

few-shot learning: 少样本学习。由于多个任务训练很慢，所以meta learning训练资料选择的任务都是few-shot learning来确保速度。

##### Find the best function $F^*$

$F^*=arg\, \underset{F}{min}\,L(F)$

### MAML

Model-Agnostic Meta-Learning
$$
L(\phi)=\sum^N_{n=1}l^n(\hat\theta^n)
$$
其中$\phi$决定了网络初始化参数$\hat\theta^n$的值，我们通过梯度下降就可以minimize $L(\phi)$，其公式为
$$
\phi=\phi-\eta\nabla_\phi L(\phi)
$$
实际操作中只在任务中下降$\theta$一次，原因就是梯度下降太慢了。

那么$\nabla_\phi L(\phi)$怎么计算呢？实际上可以使用$\theta$下降一次后的梯度方向做近似。

### Reptile

$\nabla_\phi L(\phi)$在Reptile中怎么计算呢？使用$\theta$下降多次后的与初始的$\theta$的方向。

![image-20200605171945306](.\images\image-20200605171945306.png)

前面举例都是使用$\phi$决定了网络初始化参数$\hat\theta^n$的值，实际上$\phi$也可以决定网络的结构和梯度更新等等。

## Life Long Learning

Continuous Learning, Never Ending Learning, Incremental Learning

同一个网络结构经过学习不同的任务使得对于学习过的每个任务都能有效解决。

### Knowledge Retention(知识保留)

but Not intransigence(但是不能够顽固，有能力学习新的东西)

实际上现在的网络模型在连续训练不同任务时，会出现Catastrophic Forgetting(灾难性遗忘)的问题，即学一个任务就忘掉前面所有的任务。

为什么不能简单的将所有任务的输入数据混合一起训练呢？新任务来到时需要将以前所有任务的数据都拿来，而且训练时间变长。

#### Elastic Weight Consolidation(可塑权重巩固)

在模型中的一些权重对于前面的任务是重要的，所以我们训练这次任务时只改变不重要的那些权重。
$$
L'(\theta)=L(\theta)+\lambda\sum_ib_i(\theta_i-\theta_i^b)^2
$$
其中，之前任务的每一个参数为$\theta_i^b$有一个权重$b_i$表示该参数有多重要。$b_i=0$时，表示本次训练对$\theta_i^b$无限制；$b_i=\infty$时，表示本次训练无法改变$\theta_i^b$的值。

那么权重$b_i$怎么计算呢？计算$\theta_i^b$的二次微分作为权重$b_i$，若二次微分小，说明改变$\theta_i^b$使输出的结果变化小，反之，说明对输出结果的变化大。

#### Generating Data

使用生成器来生成之前任务的数据而不用存储之前任务的数据了。

### Knowledge Transfer(知识转移)

我们希望同一个模型在不同任务上的知识互通有无，不同的任务互相帮助。

### Model Expansion(扩展模型)

任务学到一定程度，模型无法很好的处理所有的任务，所以想要自动的对已有模型进行扩展，但是扩展后的参数需要足够有效，即不能无限度的扩展模型。

### Curriculum Learning(课程式学习)

学习 任务的顺序

## Deep Reinforcement Learning

有一个Agent和一个Environment，Agent根据Environment的State做出一个行为可以改变Environment的状态，然后Environment会改变状态并给Agent一个Reward衡量刚刚Agent的行为是好还是不好。

##### example

alpha-go：根据棋盘的局势下子，然后棋盘的局势改变，然而reward是稀疏的，只有termial state时的输赢决定了reward，之前的落子基本上reward=0.

play video game：根据观察游戏的显示画面，做出反应，然后游戏中得分作为反应的reward，直到最后游戏结束。整个过程成为一个episode，学习使得在整个episode过程中reward和最大。

<img src=".\images\image-20200606153654238.png" alt="image-20200606153654238" style="zoom:67%;" />

##### Diffculties of Reinforcement Learning

* Reward delay：某些操作可以使得在未来操作中获得更大的reward，甚至需要某些在当前获得负reward的操作。（炉石手牌术的操作）
* Agent的行为会对环境有影响，所以Agent需要学会探索（否则他不会知道那些操作会获得reward还是负的reward）。

### Policy-based: Learning a Actor

$Action=\pi(Observation)$，就是利用深度学习找一个函式。

#### Defining a Function Set

输入为图像，输出为action的概率，网络由于输入是图像，所以可以采用卷积神经网络。

<img src=".\images\image-20200606161708066.png" alt="image-20200606161708066" style="zoom:67%;" />

#### Defining a Loss Function

我们拿Actor $\pi_\theta(s)$实际的去玩这个游戏，游戏过程中state序列为{$s_1,s_2,…,s_T$}，action用a表示，reward用r表示，那么

total reward: $R_\theta=\sum_{t=1}^Tr_t$

即使是同一个actor，total reward也是不一样的，因为actor根据输出概率随机选取的ation(为了保证探索)

所以我们定义$\bar R_\theta$来表示$R_\theta$的期望值，我们希望最大化$\bar R_\theta$，而不是具有随机性的$R_\theta$

##### 如何计算$\bar R_\theta$

我们定义一个episode为一个$\tau=\{s_1,a_1,r_1,s_2,a_2,r_2,…,s_T,a_T,r_T\}$

那么$\pi_\theta$下某个$\tau$出现的概率记为$P(\tau|\theta)$
$$
\bar R_\theta=\sum_\tau R(\tau)P(\tau|\theta)
$$
我们让$\pi_\theta$玩这个游戏N次，会获得{$\tau^1,\tau^2,…,\tau^N$}，这个过程就像是从$\tau$的分布中采样N次，那么
$$
\bar R_\theta=\sum_\tau R(\tau)P(\tau|\theta)\approx\frac{1}{N}\sum_{n=1}^NR(\tau^n)
$$

#### Find best Function（policy gradient）

Gradient Ascent梯度上升，因为我们要让Reward最大化。
$$
\nabla\bar R_\theta=\sum_\tau R(\tau)\nabla P(\tau|\theta)\\
=\sum_\tau R(\tau)P(\tau|\theta)\frac{\nabla P(\tau|\theta)}{P(\tau|\theta)}\\
=\sum_\tau R(\tau)P(\tau|\theta)\nabla logP(\tau|\theta)\\
\approx\frac{1}{N}\sum_{n=1}^NR(\tau^n)\nabla logP(\tau|\theta)
$$
其中，$\nabla logP(\tau|\theta)=\sum_{t=1}^T\nabla log\,p(a_t|s_t,\theta)$，所以最后
$$
\nabla\bar R_\theta\approx\frac{1}{N}\sum_{n=1}^NR(\tau^n)\nabla logP(\tau|\theta)\\
=\frac{1}{N}\sum_{n=1}^N\sum_{t=1}^{T_n} R(\tau^n)\nabla log\,p(a_t|s_t,\theta)
$$
表示为在某一次游戏中，最后$R(\tau^n)$是正的，那么就增加第t次行动面对s选择为a的概率来指导$\theta$更新的方向，反之亦然。

注意这里我们考虑是最后total reward的大小，从全局的结果来评价这次游戏的每一个操作。

##### baseline

如果reward在游戏中全是正数会存在一个问题，就是只要sample到的$\tau$就会增加其出现的概率，而抛弃了一些没有sample到的$\tau$但是获得的$R(\tau)$更大的情况。所以一般会在每一个reward前减掉一个bias，使得reward有正有负。

### Value-based: Learning a Critic

给定一个actor $\pi$，critic能够评估这个actor有多好

state value function $V^\pi(s)$，它评估了在s的状态下，一直到游戏结束actor $\pi$能够得到的reward的期望值是多大。

##### 如何学习$V^\pi(s)$

* Monte-Carlo based approach(蒙特卡洛方法)，通过蒙特卡洛方法计算出某一状态下的reward的期望，然后让$V^\pi(s)$与算出来的期望越接近越好
* Temporal-difference approach(时序差分算法)，利用公式$V^\pi(s_t)=V^\pi(s_{t+1})+r_t$，这样不需要跑完这个游戏再训练critic

<img src=".\images\image-20200606191221291.png" alt="image-20200606191221291" style="zoom:67%;" />

#### Q-Learning

state value function $Q^\pi(s,a)$，它评估了在s的状态下，actor $\pi$采取了a后，一直到游戏结束能够得到的reward的期望值是多大

在实际操作中我们不采用下图左边的形式而是右边的网络形式，这样只需要输入s可以得到所有action对应的Q

<img src=".\images\image-20200606193315803.png" alt="image-20200606193315803" style="zoom:67%;" />

当我们有了Q之后我们就可以根据Q找到一个$\pi'$比原来的$\pi$要表现得更好

<img src=".\images\image-20200606193244848.png" alt="image-20200606193244848" style="zoom:67%;" />

### Actor+Critic

Asynchronous Advantage Actor-Critic (A3C)

Advantage Actor-Critic: actor不再根据$R(\tau)$，而是根据critic的值来学习。

Asynchronous: 对当前的模型开很多的分身，让分身在数据中训练，得到的梯度再取平均更新当前模型的参数。
