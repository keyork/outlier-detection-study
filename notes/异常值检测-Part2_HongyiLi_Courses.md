# Anomaly Detection

## 问题定义

给出一组训练数据$\{x^1,x^2,\cdots ,x^N\}$

将一个$x$输入到Detector，输出$normal$或者$anomaly$

也可以称作outlier, novelty, exceptions

“像不像训练资料”就是要做的事情

- 一个很直觉的想法就是：

  - 收集正常和异常的资料

  - 做二分类，normal->Class 1, anormal->Class 2

  - 训练一个二分类的模型

但事情并不是这么简单

因为不太能看成二分类的问题

异常检测本质上是一个$x\ \&\ !x$的问题，而不是$A\ or\ B$的问题

异常的数据不太好收集

## 异常检测的分类

三类：带label、不带label的clean数据、不带label的polluted数据

### 第一类：带有label

可以训练一个Classifier

input一个$x$，output的是class为$y$和confidence score为$c$

设置一个$\lambda$，比较$c(x)$和$\lambda$的大小，这个是Classifier要做的事情

做anomaly detection则要看模型输出的所有分数，如果是normal，则会有一个很高很高的confidence，如果是anomaly，则confidence会比较平均

于是一个很直观的方式就是**取最大值**

或者**计算熵**

有一个工作，是直接让模型output一个confidence

《Learning Confidence for Out-of-Distribution Detection in Neural Networks》

如何评价一个异常检测系统的好坏

在二分类问题中，都是用正确率来评价系统的好坏

异常检测并不是这样，因为normal和anomaly的数据极度不均衡，所以不能够用acc来衡量系统的好坏

引入一个cost table的东西，根据不同的任务来选择不同的cost table

| Cost     | Anomaly | Normal |
| -------- | ------- | ------ |
| Detected | 0       | $B$    |
| Not Det  | $A$     | 0      |

选择不同的$A$和$B$

还有一个AUC(Area under ROC curve)

### 第二类：clean的数据

所有训练数据都是normal

“网络上的人一起玩宝可梦”

如何检测出一个“网络小白”

需要一些训练资料：$\{x^1,x^2,\cdots,x^N\}$

需要把每一个$x$表示成一个向量$[x_1\ x_2\ \cdots]$

我们有大量的$x$，但没有$y$，于是我们可以建立一个模型，输出$P(x)$，某一种行为发生的几率，设置一个阈值$\lambda$

如何量化$P(x)$？Likelihood

假设有一个probability density function$f_\theta(x)$，$\theta$是一个参数，决定了probability density function的形状，需要从训练资料中得到$\theta$

然后对于每一个训练资料，都有一个$f_\theta(x^i)$，于是得到一个似然函数：$L(\theta)=f_\theta(x^1)f_\theta(x^2)\cdots f_\theta(x^N)$

求$\theta^*=arg\max\limits_{\theta}L(\theta)$

假设$f_\theta(x)$是一个network，$\theta$是一个network的参数

input一个$\{x_1,x_2,x_3,x_4,x_5\}$，很多很多的feature，训练出一个model，让其在训练集上likelihood尽可能大，然后在测试数据上输出的likelihood就是一个指标，一般也会取log

---

还有auto-encoder可以做这件事情

Train的过程，把所有训练资料拿到，过一个Encoder，将数据encode成一个code，然后再过一个Decoder，还原数据，尽可能让这两个像

Test，如果能够还原，则没问题，如果不能还原，则为异常值

---

One-class SVM

Isolated Forest

### 第三类：polluted的数据

手上的数据含有一部分anomaly

