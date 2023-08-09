# Object Detection 3D
## 3D目标检测介绍与基础
### 3D目标检测的预测目标
1. 位置(x,y,z)，通常是中心点在真实世界的坐标，以米为单位
2. 尺寸(w,l,h)，物体的三维尺寸，以米为单位
3. 朝向$\theta$，通常为俯视图上物体前进方向和x轴的夹角
4. 物体类别（汽车，行人）或属性（静止、运动）
5. 有些场景需要预测速度

更一般的情况下，物体可能在yaw,pitch,roll三个方向上都有旋转，在加上上下、左右
前后三个方向的移动，总共有6个运动自由度，称为6-DoF(Degree of Freedom)

在自动驾驶场景里，默认pitch,roll为0，因此只需要yaw，称为4-DoF

### 3D检测的问题和思路
3D和2D不同的地方就是要多出一个对深度信息的估计，图像上没有深度信息，但是3D检测要求给出深度的信息，所以必须要有所补充

1. 方法一：数据，使用3D传感器来获取周围环境的物理信息，让数据本身就包含深度信息。
   这样的做法通常更准确，但是需要增加额外的设备和成本，数据形态和图像不同，需要特定的模型和算法。
   **即基于点云的算法**
2. 方法二：从算法入手，基于图像，让算法推理出深度。
   这样的做法设备简单、成本低，算法层面可以借鉴很多的2D的模型，但是问题本身存在一定的病态性
   **即基于单目相机或多目相机的纯视觉算法**
3. 方法三：融合图像和点云数据进行预测
   实践中需要考虑多传感器的标定和对齐
   **即多模态方法**

### 3D传感器与数据
#### LiDAR
><https://en.wikipedia.org/wiki/Lidar>

基本原理：发射激光脉冲，物体反射后被接收器接收，测量发射接收的时间差即可计算距离，这个时间差叫做TOF（time of flight）。

![](https://upload.wikimedia.org/wikipedia/commons/f/f1/20200501_Time_of_flight.svg)

让激光头旋转起来就能实现水平360度的测量

这一系列得到的点的集合称为点云，是对物体表面的空间采样

为了覆盖不同的俯仰角度，可以将激光头在垂直方向排成一列，这些激光头同步旋转，可以探测到不同俯仰角度上的物体，由于每个激光头的俯仰角在制造过程中是确定且已知的，所以可以计算出相应的三维坐标

#### 点云数据的特点
1. 点云是点的集合，是非结构化的数据，点与点之间没有顺序或位置上的关系，将点云矩阵按行打乱后表示的是同样的点云。相比之下RGB图像是结构化的，像素之间会有上下左右的位置关系
2. 点云提供三维结构的信息，但是不包含逻辑和纹理
3. 点云在近处密度大，远处密度小，室外场景更明显，一些远处的小物体可能只有几个点，但是图像会有很高的分辨率

#### 3D传感器

**激光雷达**：使用激光测量物体的距离，常用于室外场景，精度较高，但激光容易受天气等因素影响
**毫米波雷达**：Radar,使用毫米波（1-10mm的电磁波）测量物体到雷达的距离，常用于室外，微波的穿透性强于光波，因此不易受天气和环境的影响，精度比LiDAR低，此外可借助多普勒效应测量物体速度
**RGB-D相机**：通过红外线测量物体到相机的距离，可以在拍摄RGB彩图的同时给出单通道深度图，红外线在室外受光影响较大，通常用于室内

### 坐标系
图像坐标系：左上角原点，XY轴向右向下，单位为像素，（u,v）
相机坐标系：相机光轴为Z轴，X轴水平向右，Y轴竖直向下，单位为米，(x',y',z')
世界坐标系：关注的物体所在的坐标系，是已知的，单位为米，(x,y,z)

我们要做的是将世界坐标系变换到图像坐标系

世界坐标系到相机坐标系的变换是刚体变换，平移+旋转，用外参矩阵表示

从相机坐标系到图像坐标系，使用的是小孔成像模型，用内参矩阵表示

><https://blog.csdn.net/chentravelling/article/details/53558096>
## 点云的特征和提取算法
### 点云网络的设计
总体的思路：对点云使用主干网络提取特征，通过任务头得到结果。

#### 点云处理方法
点云是不能直接使用卷积处理的，有以下的方法

1. 给点云赋予空间结构：空间栅格化，基于空间形成3D栅格或者地面形成2D栅格，将点云以某种方式嵌入栅格中，形成栅格化的数据，之后可以基于卷积网络或Transformer来提取特征
2. 直接处理点云，**PointNet和PointNet++模型**

#### PointNet
><https://arxiv.org/abs/1612.00593>

![](http://stanford.edu/~rqi/pointnet/images/pointnet.jpg)

理论：点云的性质：无序、要考虑点与点之间的联系、变换不变性。因为点云的性质，因此网络的函数应该是对称的，PointNet使用MLP计算每个点的特征，再用max-pooling得到全局的特征。因此网络不仅能够给出对整个输入预测的label也可以给出某个点或者某一部分的label。并且由于每个点是单独处理的，所以可以继续延展，在输入PointNet前用各种网络对点进行处理均可。

PointNet的理论表明，MLP+Pooling+MLP的形式，是满足变换不变性的。MLP提取特征时也会进行升维，使得Pooling在降维时不会丢失太多信息。

作者也给出了PointNet网络能够拟合任意的连续集合函数的证明。

一个简单的实现方法：对点云数据进行预处理，通过仿射变换将坐标转换到一个常规坐标系下，比如将物体转换到一个比较好的角度再处理和计算特征。依照这个思路，PointNet网络中加入了刚性变化的模块，以实现对刚性变换的不变性。对于分类问题，网络参考蓝色的部分;对于分割的问题，网络将global feature拼接到原来的向量上，得到的整合后的向量输入进入网络处理。

PointNet使用一个子网络T-Net预测刚性变换的参数，再将变换矩阵应用到所有空间点或特征点上，以实现整个网络对刚性变换的不变性

PointNet具有稳定性。其结构中的MLP应用于单点，池化基于所有点的特征，因此只能得到单点的特征和全局的特征，没有介于中间尺度的特征，PointNet是没有逐级抽象的能力的

#### PointNet++
><https://arxiv.org/abs/1706.02413>
引入了Set Abstraction来解决层次化的问题。

过程是先进行下采样得到中心点，每个中心点取邻域进行聚合，基于每个领域中的点计算PointNet特征，再与原始的点云坐标拼接

下采样用的是FPS（Farthest Point Sampling），FPS基于点云的空间结构，可以更好地撑起点云

Set Abstraction相当于一个对点云进行一次抽象的操作，可以堆叠这个操作，实现层次化处理

![](http://stanford.edu/~rqi/pointnet2/images/pnpp.jpg)

同时，根据点云的密度不均匀，PointNet++还引入了Multi-scale grouping（MSG），具体来说是在Set Abstraction的聚合步骤中使用不同的邻域半径进行PointNet的特征计算，再将不同的尺度的特征拼接起来，作为中心点的特征，输出到下一层

## 基于点云数据的3D检测算法
### 基本思路
* 方法一：基于体素(Voxel)
    在三维空间中划分格子，将点云以某种方式嵌入到格子中，得到一个三维的特征体，之后可以借鉴2D检测的流程，使用主干网络和特征头组成的模型产生预测的结构，主干网络通常基于3D卷积网络
    **代表作：VoxelNet,SECOND...**
* 方法二：基于点云的鸟瞰图
    与体素的方法类似，但只在地面上划分格子，高度维不划分，可以将点云直接转化为2D特征图，之后基于2D检测算法产生预测框，由于预测目标有所区别，需要更改回归分支
    **代表作：Pixor,Complex-YOLO,PointPillars,CenterPoint等**
* 方法三：基于点
    使用PointNet++，直接基于非结构化的点云产生特征，再基于点的特征产生预测框
    **代表作：Point RCNN**
### 基于空间栅格（体素）的方法
基本思路：
1. 将空间在水平竖直深度三个方向上划分格子，每个格子称为体素（类比像素），或仅在地面划分格子，形成一个个柱体
2. 在体素内使用类PointNet模型提取点云特征，将不规则的点云转化为规则的2D特征图或3D特征体
3. 基于卷积网络和图像检测算法产生检测框

#### VoxelNet
><https://arxiv.org/abs/1711.06396>

![](https://miro.medium.com/v2/resize:fit:720/format:webp/1*ge0k7Nb4-6Zhz7R4a-2ZBA.png)
整体流程：
1. 将空间划分为体素，在体素内使用VFE提取局部点云特征，得到三维的特征体
2. 将特征送入3D卷积网络，进一步提高表达能力，将最终输出在竖直方向压缩，得到2D特征图
3. 将2D特征图送入PRN网络产生3D框的预测

![](https://miro.medium.com/v2/resize:fit:720/format:webp/1*wqOvy8tYQ6TqAb-Ncgkjow.png)
VFE结构用于提取体素内的点云特征：
1. 结构类似PointNet：用一层全连接层对每个点的原始坐标进行变化，再池化得到全局特征，再将全局特征拼接到逐点特征上输出
2. 可以堆叠，VFE的输出可以作为下一个VFE模块的输入，多层堆叠来提高特征的表达能力

VoxelNet中间层：
1. VFE输出特征体尺寸为128x10x400x352（特征维x竖直x水平x深度）
2. 输入若干3D卷积网络，竖直方向通过步长降采样，输出特征尺寸为64x2x400x352
3. 将竖直方向的两个体素的特征拼接起来，得到2D特征图，尺寸为128x400x352(特征维x水平x深度)

VoxelNet的问题：
1. 因为有较多的3D卷积层，速度比较慢
2. 角度回归精度不够（早期的很多工作都有这个问题）

#### SECOND
><https://www.mdpi.com/1424-8220/18/10/3337>

SECOND针对问题的改进：
1. 提出了针对稀疏Tensor的高效算法spconv，相比VoxelNet提速4倍
2. 角度回归的目标值设置为了角度的正弦值，考虑了角度的周期性，提高了角度回归的精度

### 基于点云的鸟瞰图
#### PointPillars
><https://arxiv.org/abs/1812.05784>

![](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcuqDaX%2FbtrVe4mU7TR%2FlKDa3uMDvpMuS6RIBdOBs0%2Fimg.png)

1. 只在地面上划分二维网格，不在高度方向划分格子，形成一系列柱体Pillars
2. 在每个柱体内使用简化版的PointNet编码点云特征，得到2D特征图
3. 送入2D检测模型（类似SSD）产生边界框预测结果

**具体实现步骤如下：**

* 按照点云数据所在的X，Y轴（不考虑Z轴）将点云数据划分为一个个的网格，凡是落入到一个网格的点云数据被视为其处在一个pillar里，或者理解为它们构成了一个Pillar。

* 每个点云用一个$D=9$维的向量表示，分别为$(x,y,z,e,x_c,y_c,z_c,x_p,y_p)$。其中$x,y,z,r$为该点云的真实坐标信息（三维）和反射强度；$x_c,y_c,z_c$为该点云所处Pillar中所有点的几何中心；$x_p,y_p$为$x-x_c,y-y_c$,反映了点与几何中心的相对位置。

* 假设每个样本中有$P$个非空的pillars，每个pillar中有$N$个点云数据，那么这个样本就可以用一个$(D,P,N)$张量表示。

* 怎么保证每个pillar中有$N$个点云数据呢？如果每个pillar中的点云数据数据超过$N$个，那么我们就随机采样至$N$个；如果每个pillar中的点云数据数据少于$N$个，少于的部分我们就填充为0。

* 这样就实现了点云数据的张量化。实现张量化后，作者利用简化版本的PointNet对张量化的点云数据进行处理和特征提取。特征提取可以理解为对点云的维度进行处理，原来的点云维度为$D=9$,处理后的维度为$C$,那么我们就获得了一个$(C,P,N)$的张量。接着，我们按照Pillar所在维度进行Max Pooling操作，即获得了$(C,P)$维度的特征图。

* 为了获得伪图片特征，作者将$P$转化为$(H,W)$。那么我们就获得了形如$(C,H,W)$的伪图片了。

* 最终转化成了一张伪图片，能够告诉网络，在坐标为$(H,W)$的地方的pillar有$C$的信息。
#### CenterNet
><https://arxiv.org/abs/1904.08189>

![](https://pic1.zhimg.com/80/v2-8307252a4c79c69303161e7998907364_720w.webp)

Centernet提出，针对2D检测的算法要将传统检测算法中的“以框表示物体”变成“以中心点表示物体”，将2D检测建模为物体中心的检测和额外的回归任务，一个框架就可以覆盖2D，3D，姿态估计等一系列任务
#### CenterPoint
><https://arxiv.org/abs/2006.11275>

使用了CenterNet的检测头的思路，自底向上地构建检测框架
#### Single-stride Sparse Transformer
><https://arxiv.org/abs/2112.06375>

![](https://media.arxiv-vanity.com/render-output/7188013/x4.png)

核心思路：利用Transformer的attention机制，能获取足够的感受野且能够适用于稀疏的点云体素

结构：可以看成是PointPillars+Swin Transformer+LiDAR

通过实验作者发现，尝试去除降采样算子的过程中网络性能总是提升的，但是完全不使用降采样算子的话，网络的有效感受野会变非常小，完全使用单步卷积的速度会非常慢。Transformer可以一定程度上缓解这个问题，作者通过Regional Grouping和Shifted Regional Grouping的操作扩充了感受野，并解决了单一划分可能切断一个物体的不合理性，同时省去了很多没有必要的卷积操作，在保证了感受野的同时避免了下采样导致的信息损失。

![](https://media.arxiv-vanity.com/render-output/7188013/x3.png)
#### 小结
* VoxelNet开创了先河，但是3D卷积速度很慢
* SECOND提出了3D稀疏卷积spconv，提高了VoxelNet的速度，同时提出了正弦值作为损失，提高了角度回归的精度
* PointPillars使用Pillars创造伪图片直接输出2D特征图，可以使用成熟的2D检测网络产生预测结果，易于落地，现在使用广泛
* CentorPoint结合了CenterNet的检测头，也应用很广泛
* SST使用了改版的Swin-Transformer作为主干网络，Transformer相比卷积更加适合稀疏的点云。

### 基于点的方法
#### PointRCNN
><https://arxiv.org/abs/1812.04244>

![](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FkOzEr%2FbtrVcLU6uCj%2F3aKsiLUyfsWNyXIhgxk71K%2Fimg.png)

**两个阶段**
* 第一阶段使用PointNet++对点云进行前景/背景点分割，基于前景点的特征产生3D候选框 

    1.使用PointNet++计算逐点特征
    2.分割前景/背景点，将标注框内的点当作前景点，将标注框外的点当作背景点，检测框就是一个免费的标注，
    3.基于每个前景点的特征产生3D提议框
    4.使用NMS移除重叠框

* 第二阶段使用PointNet++基于候选框内的点的特征对候选框进行修正

    1.将框内点的坐标旋转平移到框的局部坐标系下，在框内计算每个点的局部特征。每个点的局部特征=（局部坐标，距相机距离，反射率），送入MLP进一步编码，为每个点计算一个局部特征
    2.局部特征与第一阶段计算的语义特征拼接，送入PointNet++网络，得到整个框的特征。
    3.预测物体类别，回归边界框残差
    4.结合候选框和偏移量得到最终预测框
    
#### Part-A2
><https://arxiv.org/abs/1907.03670>

在Point-RCNN基础上进一步发掘标注框提供的额外信息，引入了part-aware和part-aggregation两个模块，使用3D稀疏卷积替代了PointNet

#### PV-RCNN
><https://arxiv.org/abs/1912.13192>

结合了3D稀疏卷积的高效和Set Abstraction的灵活，第一阶段使用3D稀疏卷积产生特征和提议框，第二阶段基于Set Abstraction聚合点的特征并最终完成边界框的预测，在一二阶段之间，卷积特征到点特征的转换是通过Voxel-to-keypoint Scene Encoding步骤完成的。

## 点云融合视觉的方法
点云包含物体的三维信息，但没有纹理和语义信息。图像包含纹理和语义信息，但是不包含三维信息。这两种形式恰好互补，多模态方法想融合不同模态的信息进行预测。通常需要同时提取点云和图像的特征，对两种模态的特征进行融合再产生预测结果。

#### MV3D
><https://arxiv.org/pdf/1611.07759.pdf>

![](https://img-blog.csdnimg.cn/2021072216044872.png)

基于点云和图像的二阶段方法，整体流程：
1. 将点云鸟瞰图通过手工设计特征转换为2D流程图，送入卷积网络产生特征
2. 将点云前视图和图像送入卷积网络分别提取特征
3. 使用RPN基于鸟瞰图产生3D的提议框
4. 将3D提议框按照几何关系分别投影到点云鸟瞰图、点云前视图和图像上
5. 对于每个提议框，使用RoI Pooling从3种特征图中提取框的特征
6. 对三种特征进行融合，最终预测该提议框的类别和边界框的回归值

由于这里的提议框是由鸟瞰图生成的，点云在处理小物体的时候，recall值很不理想，该模型在使用时在小物体上有很大的局限性。

#### Frustum PointNets
><https://arxiv.org/abs/1711.08488>

![](http://stanford.edu/~rqi/frustum-pointnets/images/teaser.jpg)

基于点云和图像的两阶段方法，流程：
1. 使用2D检测的方法，从图像产生2D的检测框，同时预测物体的类别。因为是基于图像的，图像的分辨率高于点云，而且可以直接给出类别，对远处的物体也很友好。
2. 图像上的检测框在空间中对应形成一个锥形
3. 使用PointNet从点云中预测物体的3D位置和大小

![](http://stanford.edu/~rqi/frustum-pointnets/images/pipeline.jpg)

先使用2D检测网络进行特征的提取，然后要用两个PointNet，第一个PointNet进行前景和背景的分割，第二个PointNet对于前景点进行框选预测。

#### MVX-Net
><https://arxiv.org/abs/1904.01649>

![](https://img-blog.csdnimg.cn/93ceb69eeca24c638676abf5f198107b.png)

基本思路：模型主题为基于点云的VoxelNet结构，将图像特征按照集合关系映射到点云上，和点云特征进行融合，作为信息的补充

方案一：点融合方法，将点投影到图像上，索引对应位置的图像特征，拼接到点特征上

方案二：RoI融合方法，将Voxel投影到图像上，将对应位置的特征池化，拼接到对应的Voxel特征上

#### 小结
1. MV3D将3D提议框投影到不同的试图，融合不同试视图的特征，进行RoI的类别预测和边界框回归
2. Frustum-PointNets使用2D图像网络产生提议框并分类，提议框对应一个锥形区域，再使用两个PointNet进行预测，但是两个模型无法联合训练
3. MVX-Net基于VoxelNet，使用2D检测模型提取图像特征，再根据几何关系融合到对应的点或者体素上，补充点云的信息

## 纯视觉的3D检测算法
高精度的雷达成本高昂，对各种复杂的天气情况敏感，视觉传感器成本低廉小巧，有很大的经济价值

视觉算法依托于图像，图像的分辨率高，对小物体的识别率更高，同时基于图像的2D检测方法更成熟，更容易落地。但是图像不包含深度信息，需要借助图像的语义信息和多目图像之间的关系推测深度

#### 算法设计思路
1. 伪点云：基于单目或者多目图像预测每个位置的深度，生成“伪点云”，再利用成熟的点云检测算法。

><https://arxiv.org/abs/1812.07179>

2. 单目视觉算法：基于2D检测的算法框架，增加额外的回归分支以预测3D框的全部7个属性

><https://arxiv.org/abs/2002.10111>
><https://arxiv.org/abs/2104.02323>
><https://arxiv.org/abs/2104.10956>

3. 多视角融合：借助多目相机获得对场景更好的理解，不同视角之间的关联可以基于Attention机制学习

><https://arxiv.org/abs/2110.06922>
><https://arxiv.org/abs/2203.17270>
### 伪点云
#### Pseudo-LiDAR
><https://arxiv.org/abs/1812.07179>

![](https://www.cs.cornell.edu/~yanwang/project/plidar/cvpr2018-pipeline.png)

核心思路：将同一时刻的左右两张相机图片经过单目或立体视觉深度估计生成模拟雷达的3D点云，再通过基于雷达信息的3D检测模块预测出3D检测框。

#### Pseudo-LiDAR++
><https://arxiv.org/abs/1906.06310>

核心思路：在Pseudo-LiDAR的基础上改进双目摄像机的深度估计网络以解决Pseudo-LiDAR对于远处物体的检测效果不好的问题，并进一步利用4线激光雷达作为辅助来微调检测结果

#### DD3D
><https://arxiv.org/abs/2108.06417>

资料链接
><https://zhuanlan.zhihu.com/p/508794328>

<img decoding="async" src="https://pic1.zhimg.com/80/v2-7e35e0d83c298d59a43bebd0f92a7ee0_720w.jpg" width="50%"><img decoding="async" src="https://owen-liuyuxuan.github.io/papers_reading_sharing.github.io/3dDetection/res/dd3d_arch.png" width="50%">

核心提问：Is Pseudo-Lidar needed for Monocular 3D Object detection?

伪点云是两阶段的，先预测深度，再预测框。DD3D提出了用同一个网络完成深度预测和3D物体检测。
### 单目视觉算法
主要为魔改2D的算法使其适配3D的任务
#### SMOKE
><https://arxiv.org/abs/2002.10111>

基于Centernet的框架，进行热力图分支的预测和深度回归分支的预测，自底向上进行预测任务。

![](https://pic3.zhimg.com/80/v2-eb67c5bdb473df63ba302798f3add41a_720w.jpg)

对于热力图分支：将3D框的中心点 $(x,y,z)$ 通过内参矩阵 $K$ 投影到图像上，作为模型需要预测的2D中心点 $(x_c,y_c)$

对于深度回归分支：使用预定义的均值 $\mu_z$ 和标准差 $\sigma_z$ 对深度进行标准化，即 $z=\mu_z+\delta_z\sigma_z$ ，让网络预测 $\delta_z$

对于长宽高和转向角本作也提出了自己的预测方法

#### FCOS 3D
><https://arxiv.org/abs/2104.10956>

![](https://gaussian37.github.io/assets/img/vision/detection/fcos3d/0.png)

思路
* 基于2D检测中的FCOS模型
* 主干网络和FPN不变
* 对3D框的7个预测值进行编码，使其更加适应2D检测模型
* 增加额外预测分支预测所有的3D参数

#### 小结
SMOKE基于2D的CenterNet，设计了针对3D框的热力图分支和回归分支

FCOS 3D基于2D检测的FCOS，针对3D框设计了回归的目标

### 基于Transformer的多视角融合方法
### DETR3D
><https://arxiv.org/abs/2110.06922>

![](https://tsinghua-mars-lab.github.io/detr3d/images/detr3d_model.png)

核心思路：借鉴2D检测中的RETR，使用Transformer基于多视角图像特征预测3D检测框，其中的Transformer模块是针对3D任务特别设计的

编码器使用ResNet+FPN提取多视角、多层次的图像特征。

解码器针对3D检测设计的检测头，逐步聚合图像特征，最终预测出检测框

loss是基于set prediction loss训练

### DEVFormer
><https://arxiv.org/abs/2203.17270>

核心思路：在先有的多相机3D检测方法（DETR3D）中引入时序信息，在Transformer中融合时空信息

![](https://pic2.zhimg.com/80/v2-387928f029237beecf3cd5ff48692251_720w.webp)
