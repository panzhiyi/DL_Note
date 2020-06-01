# Machine Learning 2020 (Hung-yi Lee)

## Foreword

写这个笔记的目的，首先是为我申博面试复习做准备，其次是让小朋友@sevati_song入门看这个课程的时候不孤单。

李宏毅老师的课程讲的非常面向新人，清晰且全面，我的笔记只是为了让我更加能够理解这些知识，更加能够构建体系而记录的，只能做到查漏补缺的目的。

希望小朋友读到的时候能够开心，培养出学习的快乐。

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