# K-Fashion调参过程
**环境：**

pip install pyecharts

pip install tqdm

pip install numpy

pip install pandas

**第一步（调整C）：** 打开B0\_training.py，调整C

 !(图片错误)

没有报错就达到基本要求，具体定价方式合理与否尚需讨论。

**第五步：** 随意打开render.html的各个分支，查看各个参数，思考当前策略合理性，目前想到的想到的问题如下，提供参考：

- 随着层次加深，C所在的多项式呈现的比重逐渐增大，是否需要考虑各层使用动态的C进行处理，正在思考是否需要将 **每一层的C** 的计算公式为：

其中为每层的C值，表示初始设置的C值

- 当前每个决策 **节点价值** 为其所有子节点价值的加权均值，其中子节点的权值为在其母节点被访问的情况下，该节点被访问的概率，而所有子节点的权值和不一定为1，因为存在母节点被访问而不再访问子节点的情况。
- 当前 **回归人口** 处理方式是否合理：针对第12周的节点，分别路径上每层计算回归人口回归的价格期望的期望，将这些人重新加到第12周大部队中，回归用户的价格期望计算方式如下：

- 当前 **增量方式** 是随机选取两个季度的状态模拟，且新定价会不高于原状态定价，如第二周定价为699，出现了未出现的状态，则该状态之后的第三周的定价不会超过699，若状态为无产品卖出，则定价小于上一周定价。
