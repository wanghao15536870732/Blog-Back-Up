---
title: 机器学习方法(三)：Logistic Regression 对数几率回归
date: 2019-07-31 20:11:49
tags:
- Machine Learning
- 笔记
toc: true
reward: true
---

## 广义线性模型

<div align=center>
<img src="/LinearRegression.png" alt="对数线性回归示意图" style="zoom:40%;"/>
</div>


对于线性回归而言，模型可以简写为：
$$
\begin{equation}
y=w^Tx+b \ \  or \ \ y=θ^Tx(更常用，下面统一用该式子表示) \tag{1}
\end{equation}
$$
假设我们的模型的输出是在指数尺度上的变化，我们可以简单的将输出的对数作为线性模型逼近的目标，将线性模型映射到指数变化上，即：
$$
\begin{equation}
ln\, y=w^Tx+b \ \ or \ \ ln\, y=θ^Tx\tag{2}
\end{equation}
$$
这就是“对数线性回归”`(log-linear regression)`，如上图中的红线映射到黑色的指数线，这就是广义线性模型的思想。更一般的考虑单调可微函数`g(·)`，令：
$$
\begin{equation}
y=g(w^Tx+b) \ \ or \ \ y=g(θ^Tx) \tag{3}
\end{equation}
$$
这就得到了“广义线性模型”，虽然在形式上还是线性回归，但实质上是对输入空间的非线性变化，使得函数具有非线性的属性。

## 一、对数几率回归

虽然被称为“回归”，但是对数几率回归`(Logistic Regression)`不是用来解决回归问题的，而是用来解决分类问题的。对于简单的二分类问题，实际上是样本点到一个值域y∈{0,1}y∈{0,1}的函数，表示这个点分在`正类(postive)`或者`反类(negtive)`的概率，若该样本非常可能是`正类`，那么输出的概率值越接近1；反之，若该样本非常可能是`负类`，则输出的概率值越接近0。

而线性回归模型产生的预测值$y=wT+b$是实数值，于是需要一个理想的函数来实现输出实数值z到0/1值的转化。最理想的是单位阶跃函数`(uint-step function)`：

而线性回归模型产生的预测值y=wT+by=wT+b是实数值，于是需要一个理想的函数来实现输出实数值zz到0/10/1值的转化。最理想的是单位阶跃函数`(uint-step function)`：
$$
\begin{equation}
y=\left\{\begin{array}{ll}
0, & z<0 \\
0.5, & z=0 \\
1, & z>0
\end{array}\right. \tag{4}
\end{equation}
$$
然而该函数不连续，于是我们希望能够找到一个近似单位越界函数，并且单调可微的函数来代替。

对数几率函数`(sigmoid function)`正是一个常用的替代函数:
$$
\begin{equation}
g(z)=\frac{1}{1+e^{-z}} \tag{5}
\end{equation}
$$
它将`z`的值转化为一个接近`0`或`1`的`y`值，将对数几率函数作为式$(3)$中的$g(·)$，可得：
$$
\begin{equation}
h_{\theta}(x)=\frac{1}{1+\mathrm{e}^{-\theta^{T_{x}}}} \tag{6}
\end{equation}
$$
二者的图像如下图所示：

<div align=center>
<img src="/SigmoidFunction.png" alt="Sigmoid函数图像"/>
</div>

从上图可以看出，它将$z$值转换为一个接近$0$或$1$的$y$值，我们可以将其视为类$1$的后验概率估计$hθ=P(y=1|x;θ)hθ=P(y=1|x;θ)$，即输入一个测试数据$x$，通过$Sigmoid$函数计算出来的结果即为该点$x$输入类别$1$的概率大小。

通常我们将$g(z)≥0.5$ 的归为类别$1$，即预测结果$y=1$；$g(z)<0.5$ 的归为类$0$， 即预测结果$y=0$：
$$
\begin{equation}
\hat{y}=\left\{\begin{array}{ll}
1, & h_{\theta}(x) \geq 0.5 \\
0, & h_{\theta}(x)<0.5
\end{array} \quad h_{\theta}(x)=g\left(\theta^{T} x\right), g(z)=\frac{1}{1+e^{-z}}\right. \tag{7}
\end{equation}
$$
对于式 $(1),$ 式 $(6),$ 可变换为 $\ln \frac{y}{1-y}=\theta^{T} x,$ 若将 $y$ 视为样本 $x$ 作为正例的可能性, 则 $1-y$ 是其反例的可能性, 两者的比值 $\frac{y}{1-y}$ 称为几率 $(\mathrm{odds}),$ 反映了 $x$ 作为正例的相对可能性。对几率取对数则得到了对数几率 `(log odds，亦称logit)`，将 $y$ 视为后验概率估计，重写公式有:
$$
\begin{equation}
\ln \frac{y}{1-y}=\ln \frac{P(y=1 \mid x)}{P(y=0 \mid x)}=\ln \frac{P(y=1 \mid x)}{1-P(y=1 \mid x)}=\ln \left(\frac{1}{1-P(y=1 \mid x)}-1\right)=\theta^{T} x \tag{8}
\end{equation}
$$
由式$(8)$可得：
$$
\begin{equation}
\left\{\begin{array}{l}
p(y=1 \mid x)=1-\frac{1}{1+e^{\theta T_{x}}}=\frac{e^{\theta^{T} x}}{1+e^{\theta T_{x}}}=\frac{1}{1+e^{-\theta} T_{x}} \\
p(y=0 \mid x)=\frac{1}{1+e^{\theta^{T} x}}
\end{array}\right. \tag{9}
\end{equation}
$$

## 二、决策边界（Decision Boundary）

|             决策边界              |                   非线性决策边界                   |
| :-------------------------------: | :------------------------------------------------: |
| ![决策边界](DecisionBoundary.png) | ![非线性决策边界](NonLinearDecisionBoundaries.png) |

## 三、代价函数（Cost Function）

确定逻辑回归的数学形式后, 接下来需要做的就是给定训练集, 通过最小化代价函数, 找出模型的最优参数 $\theta_{\circ}$ 线性回归中的代价函数:
$$
\begin{equation}
J(\theta)=\frac{1}{2 m} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right)^{2} \tag{10}
\end{equation}
$$
我们可以类比线性回归, 使用平均误差平方来作为代价函数。

但是对于逻辑回归而言, $h_{\theta}(x)$ 函数 $\left(h_{\theta}(x)=\frac{1}{1+e^{-\theta T_{x}}}\right)$ 是一个非常复杂的非线性救, 如果将 $\operatorname{Sigmoid}$ 函数带入式 $(9),$ 并画出 $J(\theta)$ 的图像, 你会发现 $J(\theta)$ 是个非凸函数, 这就意味代价函数有着许多的局部最小值, 如下图`non-convex`所示, 如果将梯度下降法应用到该函数上, 无法保证可以收敘到最小值, 这很不利于我们的求解。

<div align=center>
<img src="/non_convex_convex.png" alt="凸函数和非凸函数"/>
</div>

因此，通常使用[极大似然估计](https://www.youtube.com/watch?v=C6a-SMY0H50)来求解，即找到一组参数，使得在这组参数下，我们数据的似然度（概率）最大。
$$
\begin{equation}
\left\{\begin{array}{l}
p(y=1 \mid x)=h_{\theta}(x) \\
p(y=0 \mid x)=1-h_{\theta}(x)
\end{array}\right. \tag{11}
\end{equation}
$$
式$(11)$可以写成一般形式:
$$
\begin{equation}
p(y \mid x ; \theta)=h_{\theta}(x)^{y}\left(1-h_{\theta}(x)\right)^{1-y} \tag{12}
\end{equation}
$$
使用极大似然估计得到似然函数：
$$
\begin{equation}
L(\theta)=\prod_{i=1}^{m} p\left(y^{(i)} \mid x^{(i)} ; \theta\right)=\prod_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)^{y^{(i)}}\left(1-h_{\theta}\left(x^{(i)}\right)\right)^{1-y^{(i)}}\right) \tag{13}
\end{equation}
$$
为了方便求解，对式$(13)$两边同时取对数，得：
$$
\begin{equation}
\begin{aligned}
l(\theta) &=\ln L(\theta) \\
&=\sum_{i=1}^{m}\left[y^{(i)} \ln \left(h_{\theta}\left(x^{(i)}\right)\right)+\left(1-y^{(i)}\right) \ln \left(1-h_{\theta}\left(x^{(i)}\right)\right)\right] \\
&=\sum_{i=1}^{m}\left[y^{(i)} \ln \frac{h_{\theta}\left(x^{(i)}\right)}{1-h_{\theta}\left(x^{(i)}\right)}+\ln \left(1-h_{\theta}\left(x^{(i)}\right)\right)\right] \\
&=\sum_{i=1}^{m}\left[y^{(i)}\left(\theta^{T} x\right)-\ln \left(1+e^{\theta^{T} x}\right)\right] 
\end{aligned} \tag{14}
\end{equation}
$$
现在的目标是使得$l(\theta)$最大，则在其基础上使用梯度下降法求解$-l(\theta)$的最小值即可：
$$
\begin{equation}
J(\theta)=-l(\theta)=-\frac{1}{m} \sum_{i=1}^{m}[y^{(i)} \ln (h_{\theta}(x^{(i)}))+(1-y^{(i)}) \ln (1-h_{\theta}(x^{(i)}))] \tag{15}
\end{equation}
$$
为了更好的理解推导出的代价函数，可以写出代价$Cost(h_\theta(x),y)$与$h_θ$的关系函数：
$$
\begin{equation}
\operatorname{Cost}\left(h_{\theta}(x), y\right)=-\operatorname{yln}\left(h_{\theta}(x)\right)+(1-y) \ln \left(1-h_{\theta}(x)\right) \tag{16}
\end{equation}
$$
即：
$$
\begin{equation}
\operatorname{Cost}\left(h_{\theta}(x), y\right)=\left\{\begin{array}{ll}
-\ln \left(h_{\theta}(x)\right) & \text { if } y=1 \\
-\ln \left(1-h_{\theta}(x)\right) & \text { if } y=0
\end{array}\right. \tag{17}
\end{equation}
$$
绘制该函数图像：

|       $−ln(h_\theta(x))$        |       $−ln(1−h_\theta(x))$        |
| :-----------------------------: | :-------------------------------: |
| ![损失函数1](cost_function.png) | ![损失函数2](cost_function_1.png) |

从上图可以看出，如果样本值$y$为$1$的话，预测值$h_\theta(x)$越接近$1$损失越小，反之越大；同样，如果样本值$y$为$0$的话，预测值$h_\theta(x)$越接近0损失越小，反之越大。

## 四、使用梯度下降法求解

$$
J(\theta)=-\frac{1}{m} \sum_{i=1}^{m}\left[y^{(i)} \ln \left(h_{\theta}\left(x^{(i)}\right)\right)+\left(1-y^{(i)}\right) \ln \left(1-h_{\theta}\left(x^{(i)}\right)\right)\right]
$$

对式$(5)求导不难发现，$Sigmoid$函数有一个特性，即：
$$
\begin{equation}
g(z)^{\prime}=\frac{-\left(-e^{-z}\right)}{\left(1+e^{-z}\right)^{2}}=\frac{1}{1+e^{-z}} \cdot \frac{e^{-z}}{1+e^{-z}}=\frac{1}{1+e^{-z}} \cdot=(1-\frac{1}{1+e^{-z}}=)=g(x)(1-g(x)) \tag{18}
\end{equation}
$$
使用梯度下降法需要先求解梯度：
$$
\begin{aligned}
\frac{\partial J(\theta)}{\theta_{j}} &=-\frac{1}{m} \sum_{i=1}^{m}\left(y^{(i)} \frac{1}{h_{\theta}\left(x^{(i)}\right)}-\left(1-y^{(i)}\right) \frac{1}{1-h_{\theta}\left(x^{(i)}\right)}\right) \frac{\partial\left(h_{\theta}\left(x^{(i)}\right)\right)}{\partial \theta_{j}} \\
&=-\frac{1}{m} \sum_{i=1}^{m}\left(y^{(i)} \frac{1}{h_{\theta}\left(x^{(i)}\right)}-\left(1-y^{(i)}\right) \frac{1}{1-h_{\theta}\left(x^{(i)}\right)}\right) h_{\theta}\left(x^{(i)}\right)\left(1-h_{\theta}\left(x^{(i)}\right)\right) \frac{\partial\left(\theta_{j}^{T} x^{(i)}\right)}{\partial \theta_{j}} \\
&=-\frac{1}{m} \sum_{i=1}^{m}\left(y^{(i)}\left(1-h_{\theta}\left(x^{(i)}\right)-\left(1-y^{(i)}\right) h_{\theta}\left(x^{(i)}\right)\right) x_{j}^{(i)}\right.\\
&=-\frac{1}{m} \sum_{i=1}^{m}\left(y^{(i)}-h_{\theta}(x^{(i)})\right) x_{j}^{(i)})
\end{aligned}
$$
公式最终简化如下如下：
$$
\begin{equation}
\frac{\partial J(\theta)}{\theta_{j}}=-\frac{1}{m} \sum_{i=1}^{m}(y^{(i)}-h_{\theta}(x^{(i)}) x_{j}^{(i)}=\frac{1}{m} \sum_{i=1}^{m}(h_{\theta}(x^{(i)})-y^{(i)}) x_{j}^{(i)} \tag{19}
\end{equation}
$$
梯度下降法更新权重：
$$
\begin{equation}
\theta_{j}:=\theta_{j}+\eta \sum_{i=1}^{m}(y^{(i)}-h_{\theta}(x^{(i)})) x_{j}^{(i)} \tag{20}
\end{equation}
$$
或：
$$
\begin{equation}
\theta_{j}:=\theta_{j}-\eta \sum_{i=1}^{m}(h_{\theta}(x^{(i)})-y^{(i)}) x_{j}^{(i)} \tag{20}
\end{equation}
$$

```python
# calculation J
z = X * theta.T
t = -y.T * log(sigmoid(z)) - (1 - y.T) * log(1 - sigmoid(z))
J = t / m
 
# calculation grad
grad_t = (sigmoid(z) - y)
grad = (X.T * grad_t ) / m
```

## 五、解决多分类问题

我们已经知道，逻辑回归`（对数几率回归, Logistic Regression）`只能用于解决二分类问题`(Binary Classification)`，想要解决多分类问题，就需要对逻辑回归进行改进。

有两种方法可以使得逻辑回归解决多分类任务：

- 将多分类问题拆分为多个二分类问题，利用逻辑回归分类器进行投票求解。
- 改进逻辑回归的损失函数，使其不再只考虑二分类非1就0的损失，而是具体考虑每个样本的损失，这种方法被称为`SoftMax`回归。

<div align=center>
<img src="/OneVsOneOneVsRest.png" alt="OneVsOne与OneVsRest" style="zoom:60%;"/>
</div>

### 5.1 One vs Rest

<div align=center>
<img src="/ovr.png" alt="one_vs_rest" style="zoom:70%;"/>
</div>

**思想**：假设对n种类型的样本进行分类，依次将其中某一类当作`正类(positive)`，其他剩余的样本归为`负类(negtive)`，训练n个二元分类器，将待测样本传入这n个分类器中，得概率最高的那个分类器对应的样本类型即认为是该预测样本的类型。

```python
def one_vs_rest(self, X, y, num_labels, learning_rate=1e-3, num_iters=100, batch_size=200, verbose=True):
    """使用 OVR 的思想，分类标准：所得概率最高的那个模型对应的样本类型"""
    X_b = np.hstack([np.ones((len(X), 1)), X])
    all_theta = np.zeros([X_b.shape[1], num_labels])
    for it in range(num_labels):
        # 多分类化为二分类的
        y_train_t = []
        # 将当前类别标签设为 1，其他类标签设为 0
        for jt in y:
            if jt == it:
                y_train_t.append(1)
            else:
                y_train_t.append(0)
        self._theta = None
        self.train(X, np.array(y_train_it), learning_rate, num_iters, batch_size, verbose)
        # 保存每次训练得到的 theta 参数
        self.theta[:, it] = self._theta
    
def on_vs_all_predict(self, X):
    labels = self.sigmoid(X.dot(self.theta))  # 结果是保存着归为每个数字的概率
    y_pred = np.argmax(labels,axis=1)  # 返回每行的最大位置的下标
    return y_pred
```

### 5.2 One vs One

<div align=center>
<img src="/ovo.png" alt="one_vs_one" style="zoom:70%;"/>
</div>

**思想**：对n种类型的样本进行分类，每次选出22种类型，两两结合，一共有$C_2^n$种二分类情况，使用 $C_2^n$种模型预测样本类型，对预测结果进行投票，出现次数最多的样本类型，即为该样本最终的预测类型。

<div align=center>
<img src="/iris.png" alt="鸢尾花数据分类" style="zoom:80%;"/>
</div>

## 六、手写数字识别

<div align=center>
<img src="/mnist.png" alt="手写数字识别数据集""/>
</div>