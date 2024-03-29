# K-Fashion调参过程
**环境：**

pip install pyecharts

pip install tqdm

pip install numpy

pip install pandas

**第一步（调整C）：** 打开B0\_training.py，调整C

**第二步（训练模型）：** 运行B0\_training.py，运行后得到如下结果说明程序顺利执行完毕。此时文件夹model中新增了myModel20190930.pkl文件【模型文件】，根目录下新增了render.html文件【树可视化网页】

**第三步（查看C状态）：** 打开C0\_predictItem.py，运行，出现如下图案时可输入状态，借助原始数据进行测试【根据3.1-3.4提示进行测试】


  **第3.1步：** 打开data/Simulated\_Data1.csv文件，任选一个季度第一周，以下图为例，该周收益应为599\*11=6589（11个用户期望大于599，599是如上图所示预测的第一周定价），再取定价499带入数据得定价499时的收入499\*6=5988\&lt;6589，再取699\*8=5592，可看出599应是此刻最高价【计算时候还需注意库存问题，因为有可能定价499下愿意购买的客户有8个（3992），而定价599下愿意购买的客户有5个（2995），但由于剩余库存只有5，因此定价599会优于499】，单调递增和单调递减的定义域之间的收益为最大收益，因此无需计算更多价位。

A: 
5\&gt;599

B: 
3\&gt;599

C:

3\&gt;599

**第3.2步：** 输入599的 **已有状态（预测）** ，上图中的状态为5,3,3，输入结果如下图所示【注意逗号要英文逗号】（不会进行新的训练，若出现，说明有错，立即戳我）：

**第3.3步：** 重复3.1、3.2操作，记录与最优收益的差值（记录最优收益就好，实际总收益最终会输出），若差值无法接受【emmm这部分挺主观的，主管判断就好啦】，调整c值后重复第一、二、三步，直至得到可接受的C值。

**第3.4步：** 输入&quot;exit&quot;【一定要记得输入exit退出，否则之前测试结果不会存入模型（的人如果觉得测试结果不能反映模型状态问题似乎也不那么大，但是只有exit后才能输出总收益呀，虽然自己也能算。。。）】，退出，等待模型存储和打印完毕（如下图所示）


**第3.5步：** 将可接受的收益差值和C值发到群上

**第四步（寻找关于增量训练的建议）：** 打开C0\_predictItem.py，运行，出现如下图案时可输入状态，借助新数据（自创数据）进行测试

**        第4.1步：** 打开render.html，打开599决策节点，查看其下的状态节点【注意后面的节点可能会很乱，可以在对应节点上悬停鼠标或坚持&quot;节点之上就是节点描述原则&quot;获取节点各方面参数】

**第4.1步：** 只需要关注status的前三列状态（A,B,C卖出量），尽量去寻找 **不存在** 的状态，同时需要根据库存（status最后三列分别表示A、B、C的库存）来制定新的状态，即新定的状态不存在，且分别不可多余最后三列数字。将 **新状态（卑微预测）** 输入结果如下图所示（会出现红框处的基于该节点重新训练之后节点，如果没有，说明有错，立即戳我）：


没有报错就达到基本要求，具体定价方式合理与否尚需讨论。

**第五步：** 随意打开render.html的各个分支，查看各个参数，思考当前策略合理性，目前想到的想到的问题如下，提供参考：

- 随着层次加深，C所在的多项式呈现的比重逐渐增大，是否需要考虑各层使用动态的C进行处理，正在思考是否需要将 **每一层的C** 的计算公式为：

其中为每层的C值，表示初始设置的C值

- 当前每个决策 **节点价值** 为其所有子节点价值的加权均值，其中子节点的权值为在其母节点被访问的情况下，该节点被访问的概率，而所有子节点的权值和不一定为1，因为存在母节点被访问而不再访问子节点的情况。
- 当前 **回归人口** 处理方式是否合理：针对第12周的节点，分别路径上每层计算回归人口回归的价格期望的期望，将这些人重新加到第12周大部队中，回归用户的价格期望计算方式如下：

- 当前 **增量方式** 是随机选取两个季度的状态模拟，且新定价会不高于原状态定价，如第二周定价为699，出现了未出现的状态，则该状态之后的第三周的定价不会超过699，若状态为无产品卖出，则定价小于上一周定价。
