# 计算机图形学入门

[toc]

## 计算机图形学概述 Overview of Computer Graphics

### 计算机图形学是什么

计算机图形学是利用计算机来合成操作虚拟信息的技术。

The use of computers to synthesize and manipulate visual information.

### 为什么要学习计算机图形学

#### 计算机图形学的应用

+ 游戏 Video Game
+ 电影 Movies
+ 动画 Animations
+ 设计 Design
+ 可视化 Visualization
+ 虚拟现实 Virtual Reality
+ 增强现实 Augmented Reality
+ 数字画作 Digital IIIustration
+ 仿真 Simulation
+ 图形用户接口 Graphical User Interfaces
+ 拓扑学 Typography

#### 计算机图形学的基础挑战

+ 创造一个真实的虚拟世界，并和它交互
+ 需要对理解物理世界的法则
+ 需要新的计算方法、显示方式

#### 计算机图形学的技术挑战

+ 用数学语言来描述现实世界，比如曲线、曲面等
+ 光照和阴影的物理法则
+ 将图形在3D中重现并变换操作
+ 动画，仿真

### 课程内容

#### 光栅化 Rasterization

+ 将几何基元投影到屏幕上
+ 把投影基元转换为像素
+ 实时渲染，即至少30fps

#### 曲线和曲面 Curves and Meshes

+ 如何在计算机中表示几何图形

#### 光线追踪 Ray Tracing

+ 把光线从相机打到每个像素
  + 计算其相交和阴影
  + 持续计算直到光线接触到光源

#### 动画仿真 Animation/Simulation

+ 动画关键帧
+ 质量-弹簧系统

## 变换 Transformation

### 二维变换

#### 缩放 Scale

![image-20230701192827818](C:\Users\ZYX\Desktop\GAMES101计算机图形学入门\LearningNote.assets\image-20230701192827818.png)

缩放变换可以用如下方程表示
$$
{x}' = sx
\\
y' = sy
$$

可用矩阵表示为

$$
\begin{bmatrix}  
  x' \\
  y'
\end{bmatrix} =  
\begin{bmatrix}  
  s & 0 \\
  0 & s
\end{bmatrix}
\begin{bmatrix}  
  x \\
  y
\end{bmatrix}
$$

![image-20230701193321184](C:\Users\ZYX\Desktop\GAMES101计算机图形学入门\LearningNote.assets\image-20230701193321184.png)

上图所示的缩放变换可以用如下方式表示
$$
\begin{bmatrix}  
  x' \\
  y'
\end{bmatrix} =  
\begin{bmatrix}  
  s_x & 0 \\
  0 & s_y
\end{bmatrix}
\begin{bmatrix}  
  x \\
  y
\end{bmatrix}
$$

#### 翻转 Reflection

![image-20230701193443783](C:\Users\ZYX\Desktop\GAMES101计算机图形学入门\LearningNote.assets\image-20230701193443783.png)
$$
\begin{bmatrix}  
  x' \\
  y'
\end{bmatrix} =  
\begin{bmatrix}  
  -1 & 0 \\
  0 & 1
\end{bmatrix}
\begin{bmatrix}  
  x \\
  y
\end{bmatrix}
$$

#### 错切 Shear

![image-20230701193542288](C:\Users\ZYX\Desktop\GAMES101计算机图形学入门\LearningNote.assets\image-20230701193542288.png)

在竖直方向上，变换前后的y值没有变化。

在水平方向上，在最底下边的x值没有变化，沿y轴越往上x值就会增加ay。
$$
\begin{bmatrix}  
  x' \\
  y'
\end{bmatrix} =  
\begin{bmatrix}  
  1 & a \\
  0 & 1
\end{bmatrix}
\begin{bmatrix}  
  x \\
  y
\end{bmatrix}
$$

#### 旋转 Rotate

![image-20230701194028958](C:\Users\ZYX\Desktop\GAMES101计算机图形学入门\LearningNote.assets\image-20230701194028958.png)

正向旋转矩阵如下所示。
$$
R_{\theta} =
\begin{bmatrix}
\cos{\theta} & -\sin{\theta} \\
\sin{\theta} & \cos{\theta}
\end{bmatrix}
$$

逆向旋转矩阵如下所示。

$$
R_{-\theta} =
\begin{bmatrix}
\cos{\theta} & \sin{\theta} \\
-\sin{\theta} & \cos{\theta}
\end{bmatrix}
$$

观察上面两个矩阵可以知道
$$
R_{-\theta} = {R_{\theta}}^T
$$
又根据定义可知
$$
R_{-\theta} = {R_{\theta}}^{-1}
$$
综上
$$
R_{-\theta} = {R_{\theta}}^{-1} = {R_{\theta}}^T
$$

#### 线性变换 Linear Transforms

$$
x' = ax + by \\
y' = cx + dy
$$

$$
\begin{bmatrix}  
  x' \\
  y'
\end{bmatrix} =  
\begin{bmatrix}  
  a & b \\
  c & d
\end{bmatrix}
\begin{bmatrix}  
  x \\
  y
\end{bmatrix}
$$

$$
\mathbf{x}' = \mathbf{M}\  \mathbf{x}
$$

### 齐次坐标 Homogeneous Coordinate

#### 为什么要引入齐次坐标

![image-20230701200008157](C:\Users\ZYX\Desktop\GAMES101计算机图形学入门\LearningNote.assets\image-20230701200008157.png)
$$
x' = x + t_x \\
y' = y + t_y
$$

$$
\begin{bmatrix}  
  x' \\
  y'
\end{bmatrix} =  
\begin{bmatrix}  
  a & b \\
  c & d
\end{bmatrix}
\begin{bmatrix}  
  x \\
  y
\end{bmatrix}
+
\begin{bmatrix}  
  t_x \\
  t_y
\end{bmatrix}
$$

可以发现，在平移变换中，无法将其用一个矩阵表示。这是因为平移变化并不是一种线性变换。

为了将所有变化统一使用一个矩阵表示，由此引入齐次坐标。

#### 齐次坐标系

在xy轴的基础上，再引入第三个维度（w维度），就得到齐次坐标系。

+ 2D点	$(x,y,1)^T$
+ 2D向量$(x,y,0)^T$

为什么要在最后增加一个0或者1，它们有什么意义？

+ 向量 + 向量 = 向量 （0+0 = 0）
+ 点 - 点 = 向量 （1-1 = 0）
+ 点 + 向量 = 点 （1+0 = 1）
+ 点 + 点 = ？

在齐次坐标系中
$$
\begin{pmatrix} x \\ y \\ w \end{pmatrix} = \begin{pmatrix} x/w \\ y/w \\ 1 \end{pmatrix} , w \ne 0
$$

#### 仿射变换 Affine Transformations

仿射变换 = 线性变换 + 平移变换
$$
\begin{pmatrix} 
x' \\
y'
\end{pmatrix}
=
\begin{pmatrix} 
a & b \\
c & d
\end{pmatrix}
\begin{pmatrix} 
x \\
y
\end{pmatrix}
+
\begin{pmatrix} 
t_x \\
t_y
\end{pmatrix}
$$
使用齐次坐标系后，上式可以写作
$$
\begin{pmatrix} 
x' \\
y' \\
1
\end{pmatrix}
=
\begin{pmatrix} 
a & b & t_x\\
c & d & t_y\\
0 & 0 & 1
\end{pmatrix}
\begin{pmatrix} 
x \\
y \\
1
\end{pmatrix}
$$

#### 缩放变换

$$
\mathbf{S}(s_x,s_y) = 
\begin{pmatrix} 
s_x & 0 & 0\\
0 & s_y & 0\\
0 & 0 & 1
\end{pmatrix}
$$

#### 旋转变换

$$
\mathbf{R}(\alpha) = 
\begin{pmatrix} 
\cos{\alpha} & -\sin{\alpha} & 0\\
\sin{\alpha} & \cos{\alpha} & 0\\
0 & 0 & 1
\end{pmatrix}
$$

#### 平移变换

$$
\mathbf{T}(t_x, t_y) = 
\begin{pmatrix} 
1 & 0 & t_x\\
0 & 1 & t_y\\
0 & 0 & 1
\end{pmatrix}
$$

### 逆变换 Inverse Transform

$\mathbf{M}^{-1}$就是变换$\mathbf{M}$的逆变换。

![image-20230701202203264](C:\Users\ZYX\Desktop\GAMES101计算机图形学入门\LearningNote.assets\image-20230701202203264.png)

### 复合变换 Composite Transform

![image-20230701202345372](C:\Users\ZYX\Desktop\GAMES101计算机图形学入门\LearningNote.assets\image-20230701202345372.png)

复杂的变换可以通过多个简单变换组合而成，复杂变换的矩阵就是多个简单变换矩阵的乘法，但是要注意**矩阵相乘的次序**。

先平移后旋转

![先平移后旋转](C:\Users\ZYX\Desktop\GAMES101计算机图形学入门\LearningNote.assets\image-20230701202352960.png)
$$
R_{45} \cdot T_{(1,0)}
\begin{bmatrix}
x \\
y \\
1
\end{bmatrix}
=
\begin{bmatrix}
1 & 0 & 1 \\
0 & 1 & 0 \\
0 & 0 & 1
\end{bmatrix}
\begin{bmatrix}
\cos{45^{\circ}} & -\sin{45^{\circ}} & 1 \\
\sin{45^{\circ}} & \cos{45^{\circ}} & 0 \\
0 & 0 & 1
\end{bmatrix}
\begin{bmatrix}
x \\
y \\
1
\end{bmatrix}
$$


先旋转后平移

![image-20230701202601440](C:\Users\ZYX\Desktop\GAMES101计算机图形学入门\LearningNote.assets\image-20230701202601440.png)
$$
T_{(1,0)} \cdot R_{45}
\begin{bmatrix}
x \\
y \\
1
\end{bmatrix}
=
\begin{bmatrix}
\cos{45^{\circ}} & -\sin{45^{\circ}} & 1 \\
\sin{45^{\circ}} & \cos{45^{\circ}} & 0 \\
0 & 0 & 1
\end{bmatrix}
\begin{bmatrix}
1 & 0 & 1 \\
0 & 1 & 0 \\
0 & 0 & 1
\end{bmatrix}
\begin{bmatrix}
x \\
y \\
1
\end{bmatrix}
$$
对同一个图像应用多个简单仿射变换，$\mathbf{A}_1,\mathbf{A}_2,...,\mathbf{A}_n$
$$
\mathbf{A}_n \cdots \mathbf{A}_2 \mathbf{A}_1 \cdot \mathbf{x}
$$

### 分解复杂变换 Decomposing Complex Transforms

![image-20230701203701700](C:\Users\ZYX\Desktop\GAMES101计算机图形学入门\LearningNote.assets\image-20230701203701700.png)
$$
\mathbf{T}(c) \cdot \mathbf{R}(\alpha) \cdot \mathbf{T}(-c)
$$

### 三维变换 

#### 三维下的齐次坐标系

类比于二维坐标下的齐次坐标系，在xyz轴的基础上，再引入第三个维度（w维度），就得到齐次坐标系。

+ 3D点	$(x,y,z,1)^T$
+ 3D向量$(x,y,z,0)^T$

$$
\begin{pmatrix}
x \\
y \\ 
z \\ 
w 
\end{pmatrix} = 
\begin{pmatrix} 
x/w \\ 
y/w \\ 
z/w \\
1 
\end{pmatrix} , w \ne 0
$$

#### 仿射变换

$$
\begin{pmatrix}
x' \\
y' \\ 
\end{pmatrix} = 
\begin{pmatrix} 
a & b & c\\ 
d & e & f\\ 
g & h & i\\
\end{pmatrix}
\cdot
\begin{pmatrix}
x \\
y \\ 
z \\ 
\end{pmatrix}
+
\begin{pmatrix}
t_x \\
t_y \\ 
t_z \\ 
\end{pmatrix}
$$


$$
\begin{pmatrix}
x' \\
y' \\ 
z' \\ 
1 
\end{pmatrix} = 
\begin{pmatrix} 
a & b & c & t_x \\ 
d & e & f & t_y \\ 
g & h & i & t_z \\
0 & 0 & 0 & 1 
\end{pmatrix}
\cdot
\begin{pmatrix}
x \\
y \\ 
z \\ 
1
\end{pmatrix}
$$

注意仿射变换中是先应用线性变换，然后应用平移变换。

#### 缩放变换

$$
\mathbf{S}(s_x,s_y,s_z) = 
\begin{pmatrix} 
s_x & 0 & 0 & 0\\
0 & s_y & 0 & 0\\
0 & 0 & s_z & 0 \\
0 & 0 & 0 & 1
\end{pmatrix}
$$

#### 平移变换

$$
\mathbf{T}(t_x, t_y, t_z) = 
\begin{pmatrix} 
1 & 0 & 0 & t_x\\
0 & 1 & 0 & t_y\\
0 & 0 & 1 & t_z\\
0 & 0 & 0 & 1
\end{pmatrix}
$$

#### 旋转变换

绕x轴旋转
$$
\mathbf{R}_x(\alpha) = 
\begin{pmatrix} 
1 & 0 & 0 & 0\\
0 & \cos{\alpha} & -\sin{\alpha} & 0\\
0 & \sin{\alpha} & \cos{\alpha} & 0\\
0 & 0 & 0 & 1
\end{pmatrix}
$$
绕y轴旋转
$$
\mathbf{R}_y(\alpha) = 
\begin{pmatrix} 
\cos{\alpha} & 0 & \sin{\alpha} & 0\\
0 & 1 & 0 & 0\\
-\sin{\alpha} & 0 & \cos{\alpha} & 0\\
0 & 0 & 0 & 1
\end{pmatrix}
$$
注意，在绕y轴旋转中，$\sin{\alpha}$和$-\sin{\alpha}$的位置交换，这是因为$\hat{z} \times \hat{x} = \hat{y}$，y轴方向上的旋转和其他两个轴方向相反。

绕z轴旋转
$$
\mathbf{R}_z(\alpha) = 
\begin{pmatrix} 
\cos{\alpha} & -\sin{\alpha} & 0 & 0\\
\sin{\alpha} & \cos{\alpha} & 0 & 0\\
0 & 0 & 1 & 0\\
0 & 0 & 0 & 1
\end{pmatrix}
$$
把三种旋转矩阵相乘即可得到任意旋转矩阵，如下所示
$$
\mathbf{R}_{xyz}(\alpha, \beta, \gamma)
=
\mathbf{R}_x(\alpha)
\mathbf{R}_y(\beta)
\mathbf{R}_z(\gamma)
$$
上面所示的三个角度也称为欧拉角(Euler Angles)。

罗德里戈旋转公式 Rodrigues' Rotation Formula
$$
\mathbf{R}(\mathbf{n}, \alpha) = 
\cos{\alpha}\  \mathbf{I} +
(1-\cos{\alpha})\mathbf{n}\mathbf{n}^T +
\sin{\alpha} 
\mathbf{R}_z(\alpha) = 
\begin{pmatrix} 
0 & -n_z & n_y\\
n_z & 0 & -n_x\\
-n_y & n_x & 0
\end{pmatrix}
$$

### 观测变换 Viewing transformation

#### 视图变换 View/Camera Transformation

想象以下如何拍一张照片：

1. 把目标物摆好(**model** transformation)
2. 把相机摆好(**view transformation**)
3. 拍照(**projection** transformation)

<img src="C:\Users\ZYX\Desktop\GAMES101计算机图形学入门\LearningNote.assets\image-20230703212011707.png" alt="image-20230703212011707" style="zoom:67%;" />

如何确定相机？

+ 位置 Position $\vec{e}$
+ 朝向 Look-at direction $\hat{g}$
+ 垂直方向 Up direction $\hat{t}$

<img src="C:\Users\ZYX\Desktop\GAMES101计算机图形学入门\LearningNote.assets\image-20230703212425210.png" alt="image-20230703212425210" style="zoom:67%;" />

做如下约定：

+ 相机永远放在坐标原点，y轴作为垂直方向，-z轴作为朝向
+ 随着相机移动变换对象

![image-20230703212607102](C:\Users\ZYX\Desktop\GAMES101计算机图形学入门\LearningNote.assets\image-20230703212607102.png)

如何把相机固定到上述约定的位置：

+ 把$\vec{e}$移到原点
+ 把$\hat{g}$旋转到$-Z$
+ 把$\hat{t}$旋转到$Y$
+ 在完成上述两个旋转的同时，剩下的一个坐标轴$\hat{g} \times \hat{t}$也完成到$X$的旋转

用数学表示上述过程，用$M_{view}$记作变化矩阵。
$$
M_{view} = R_{view} T_{view}
$$

$$
T_{view} =
\begin{bmatrix}
1 & 0 & 0 & -x_e \\
0 & 1 & 0 & -y_e \\
0 & 0 & 1 & -z_e \\
0 & 0 & 0 & 1
\end{bmatrix}
$$

由于两个旋转不好求，可以求$\hat{g} \times \hat{t}$到$X$的旋转矩阵，这个就是旋转矩阵的逆矩阵。

$$
R_{view}^{-1} =
\begin{bmatrix}
x_{\hat{g} \times \hat{t}} & x_{t} & x_{-g} & 0 \\
y_{\hat{g} \times \hat{t}} & y_{t} & y_{-g} & 0 \\
z_{\hat{g} \times \hat{t}} & z_{t} & z_{-g} & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}
$$

$$
R_{view} =
\begin{bmatrix}
x_{\hat{g} \times \hat{t}} & y_{\hat{g} \times \hat{t}} & z_{\hat{g} \times \hat{t}} & 0 \\
x_{t} & y_{t} & z_{t} & 0 \\
x_{-g} & y_{-g} & z_{-g} & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}
$$

#### 投影变换 Projection Transformation

投影变换可以分为两种:

+ 正交投影 Orthographic projection
+ 透视投影 Perspective projection

![image-20230703214326636](C:\Users\ZYX\Desktop\GAMES101计算机图形学入门\LearningNote.assets\image-20230703214326636.png)

透视投影有近大远小的性质,而正交投影没有.

![image-20230703214439745](C:\Users\ZYX\Desktop\GAMES101计算机图形学入门\LearningNote.assets\image-20230703214439745.png)

正交投影认为相机无限远,而透视投影遵循透视法则.

##### 正交投影 Orthographic Projection

![image-20230703221120081](C:\Users\ZYX\Desktop\GAMES101计算机图形学入门\LearningNote.assets\image-20230703221120081.png)

例如把立方体映射到正则立方体$[-1,1]^3$

1. 先做平移
2. 再做缩放

用数学公式描述上述过程可得
$$
M_{ortho} = 
\begin{bmatrix}
\frac{2}{r-l} & 0 & 0 & 0\\
0 & \frac{2}{t-b} & 0 & 0\\
0 & 0 & \frac{2}{n-f} & 0\\
0 & 0 & 0 & 1
\end{bmatrix}
\begin{bmatrix}
1 & 0 & 0 & -\frac{r+l}{2}\\
0 & 1 & 0 & -\frac{t+b}{2}\\
0 & 0 & 1 & -\frac{n+f}{2}\\
0 & 0 & 0 & 1
\end{bmatrix}
$$
注意在本课程中由于使用右手系,所以"近处"n是大于"远处"f的.

##### 透视投影 Perspective Project

![image-20230703222325719](C:\Users\ZYX\Desktop\GAMES101计算机图形学入门\LearningNote.assets\image-20230703222325719.png)

如何做透视投影:

1. 把Frustum变换成Cuboid,使用$M_{persp \to ortho}$
2. 再做一次正交投影,使用$M_{ortho}$

![image-20230703222605326](C:\Users\ZYX\Desktop\GAMES101计算机图形学入门\LearningNote.assets\image-20230703222605326.png)

由相似三角形得
$$
y'=\frac{n}{z}y
$$
同理可得
$$
x' = \frac{n}{z}x
$$
我们的目标是让
$$
\begin{pmatrix}
x\\
y\\
z\\
1
\end{pmatrix}
\Rightarrow
\begin{pmatrix}
nx/z\\
ny/z\\
unknown\\
1
\end{pmatrix}
==
\begin{pmatrix}
nx\\
ny\\
still\ unknown\\
z
\end{pmatrix}
$$
即需要一个矩阵让上式成立
$$
M_{persp \to ortho}^{(4\times 4)}
\begin{pmatrix}
x\\
y\\
z\\
1
\end{pmatrix}
=
\begin{pmatrix}
nx\\
ny\\
unknown\\
z
\end{pmatrix}
$$
可以解得
$$
M_{persp \to ortho}
=
\begin{pmatrix}
n & 0 & 0 & 0\\
0 & n & 0 & 0\\
? & ? & ? & ?\\
0 & 0 & 0 & 1
\end{pmatrix}
$$
又因为在任何一个平行xoy平面的平面的z值不变,可以得
$$
M_{persp \to ortho}
=
\begin{pmatrix}
n & 0 & 0 & 0\\
0 & n & 0 & 0\\
0 & 0 & n+f & -nf\\
0 & 0 & 0 & 1
\end{pmatrix}
\\
其中f为远平面z值,n为近平面z值
$$

$$
\begin{align*}
M_{persp} &= M_{ortho}M_{persp \to ortho}\\
&=\begin{bmatrix}
\frac{2}{r-l} & 0 & 0 & 0\\
0 & \frac{2}{t-b} & 0 & 0\\
0 & 0 & \frac{2}{n-f} & 0\\
0 & 0 & 0 & 1
\end{bmatrix}
\begin{bmatrix}
1 & 0 & 0 & -\frac{r+l}{2}\\
0 & 1 & 0 & -\frac{t+b}{2}\\
0 & 0 & 1 & -\frac{n+f}{2}\\
0 & 0 & 0 & 1
\end{bmatrix}
\begin{bmatrix}
n & 0 & 0 & 0\\
0 & n & 0 & 0\\
0 & 0 & n+f & -nf\\
0 & 0 & 0 & 1
\end{bmatrix}
\end{align*}
$$

