# 基于自适应张量跟踪的L型阵列动态高自由度DOA估计研究方案

## 1. 研究背景与动机

### 1.1 背景
在现代阵列信号处理领域，波达方向（DOA）估计是雷达、声纳和无线通信系统的核心技术。L型阵列因其结构简单且能提供二维角度（方位角和俯仰角）估计而备受关注。然而，传统的基于子空间的方法（如2D-MUSIC、ESPRIT）受到物理阵元数的限制，即$M$个阵元最多只能估计$M-1$个信源。

### 1.2 现状与问题
*   **自由度限制**：为了突破阵元数限制，**Paper_3** 提出了利用L型阵列构建虚拟面阵的方法，通过高阶统计量或互相关构建张量，实现了“超分辨”（Underdetermined）估计，即估计信源数 $K > M$。但该方法通常基于批处理（Batch Processing），计算复杂度高，不适用于实时场景。
*   **动态场景挑战**：在实际应用中，信源往往是移动的（如无人机、移动终端）。**Paper_4** 探讨了自适应张量跟踪算法（如PARAFAC-RLST），能够实时跟踪张量分解的因子矩阵。然而，现有的跟踪算法大多直接应用于接收数据张量，未结合虚拟阵列技术，因此无法在动态场景下同时实现“高自由度”和“低复杂度”。

### 1.3 动机
本研究旨在融合**Paper_3**的虚拟阵列扩展能力与**Paper_4**的自适应跟踪能力。通过构建时变的虚拟阵列张量，并设计递归张量分解算法，实现在信源数多于物理阵元数的“欠定”条件下，对移动信源进行实时、高精度的2D-DOA跟踪。

---

## 2. 核心创新点深入阐述

1.  **动态虚拟张量孔径构建（Dynamic Virtual Tensor Aperture）**：
    不同于传统的静态快拍累积，本方案在每个短时时间窗（Time Block）或滑动窗口内，利用L型阵列两个子阵的互相关特性，构建一个包含了虚拟阵列流形的高阶张量。该张量在数学本质上等效于一个大孔径面阵的接收数据，从而在物理阵元有限的情况下保留了高自由度的信号子空间信息。

2.  **结构化自适应PARAFAC跟踪算法（Structured Adaptive PARAFAC Tracking）**：
    针对构建的五阶或高阶虚拟张量，不再使用交替最小二乘（ALS）进行迭代求解，而是设计一种基于递归最小二乘（RLS）或随机梯度下降（SGD）的张量跟踪器。创新之处在于将导向矢量的范德蒙（Vandermonde）结构约束融入跟踪过程，利用上一时刻的因子矩阵作为先验，仅对新到达的数据进行增量更新，大幅降低计算量。

3.  **时变参数的自动配对与解耦**：
    利用CP分解（CANDECOMP/PARAFAC）在弱条件下本质唯一的特性，直接从跟踪得到的因子矩阵中提取方位角和俯仰角。由于因子矩阵列向量的一一对应关系，该方法在动态跟踪过程中天然避免了角度配对失败的问题，即使在多目标轨迹交叉时也能保持稳健。

---

## 3. 详细技术实现方案

### 3.1 信号模型与阵列几何
考虑一个由 $2M-1$ 个全向阵元组成的L型阵列，位于 $xoz$ 平面。
*   **Z轴子阵**：包含 $M$ 个阵元，位置为 $\{(0,0, (m-1)d) | m=1,\dots,M\}$。
*   **X轴子阵**：包含 $M$ 个阵元，位置为 $\{((m-1)d, 0, 0) | m=1,\dots,M\}$。
原点处阵元共用。假设有 $K$ 个远场窄带移动信源，第 $k$ 个信源在时刻 $t$ 的方位角为 $\phi_k(t)$，俯仰角为 $\theta_k(t)$。

接收信号模型为：
$$
\begin{aligned}
\mathbf{z}(t) &= \mathbf{A}_z(\theta(t))\mathbf{s}(t) + \mathbf{n}_z(t) \\
\mathbf{x}(t) &= \mathbf{A}_x(\theta(t), \phi(t))\mathbf{s}(t) + \mathbf{n}_x(t)
\end{aligned}
$$
其中 $\mathbf{A}_z \in \mathbb{C}^{M \times K}$ 和 $\mathbf{A}_x \in \mathbb{C}^{M \times K}$ 分别是Z轴和X轴子阵的导向矩阵。

### 3.2 步骤一：动态互相关张量构建
为了扩展孔径，我们在滑动窗口 $W$ 内计算互相关矩阵。定义时刻 $t$ 的互相关矩阵 $\mathbf{R}_{zx}(t)$：
$$
\mathbf{R}_{zx}(t) = \frac{1}{W} \sum_{\tau=t-W+1}^{t} \mathbf{z}(\tau)\mathbf{x}^H(\tau) \approx \mathbf{A}_z(t) \mathbf{R}_s(t) \mathbf{A}_x^H(t)
$$
为了应用张量跟踪并利用时间分集，我们将连续 $L$ 个时刻的互相关矩阵堆叠，或者利用四阶累积量构建虚拟张量。此处采用**Paper_3**中的思想，构建一个等效的三阶虚拟张量 $\mathcal{Y}(t) \in \mathbb{C}^{M \times M \times L}$（此处简化为三阶以便推导，实际可扩展至高阶）：
$$
\mathcal{Y}_{ijk}(t) = \text{Cross-Correlation-Element}(i, j, \text{lag}_k)
$$
该张量的CP分解形式为：
$$
\mathcal{Y}(t) \approx \sum_{r=1}^{K} \mathbf{a}_{z,r}(t) \circ \mathbf{a}_{x,r}^*(t) \circ \mathbf{c}_r(t)
$$
其中：
*   $\mathbf{a}_{z,r}(t)$ 是 $\mathbf{A}_z$ 的第 $r$ 列（包含俯仰角信息）。
*   $\mathbf{a}_{x,r}^*(t)$ 是 $\mathbf{A}_x$ 共轭的第 $r$ 列（包含方位角和俯仰角耦合信息）。
*   $\mathbf{c}_r(t)$ 包含信号源的动态功率或时间相关系数。

### 3.3 步骤二：自适应PARAFAC跟踪算法
我们定义时刻 $t$ 的代价函数，引入遗忘因子 $\lambda$ ($0 < \lambda \le 1$) 以适应非平稳环境：
$$
J(t) = \sum_{i=1}^{t} \lambda^{t-i} \left\| \mathcal{Y}(i) - \llbracket \mathbf{A}(i), \mathbf{B}(i), \mathbf{C}(i) \rrbracket \right\|_F^2
$$
其中 $\mathbf{A}(t) = \mathbf{A}_z(t)$, $\mathbf{B}(t) = \mathbf{A}_x^*(t)$。
由于 $\mathbf{A}(t)$ 和 $\mathbf{B}(t)$ 是慢变参数（角度变化连续），而 $\mathbf{C}(t)$ 可能快变。我们采用递归更新策略。

#### 算法流程：
1.  **输入**：当前时刻观测张量切片 $\mathbf{Y}_t$（或新构建的张量块），上一时刻的因子矩阵估计 $\hat{\mathbf{A}}(t-1), \hat{\mathbf{B}}(t-1)$。
2.  **更新 C（信号/时间因子）**：
    假设 $\mathbf{A}, \mathbf{B}$ 暂时不变，求解 $\mathbf{c}(t)$（当前时刻的信号强度向量）：
    $$
    \hat{\mathbf{c}}(t) = \left[ (\hat{\mathbf{B}}(t-1) \odot \hat{\mathbf{A}}(t-1))^H (\hat{\mathbf{B}}(t-1) \odot \hat{\mathbf{A}}(t-1)) \right]^{-1} (\hat{\mathbf{B}}(t-1) \odot \hat{\mathbf{A}}(t-1))^H \text{vec}(\mathbf{Y}_t)
    $$
3.  **更新 A（Z轴导向矩阵）**：
    利用RLS思想更新 $\mathbf{A}$。定义协方差逆矩阵 $\mathbf{P}_A(t)$。
    $$
    \mathbf{V}_A(t) = (\hat{\mathbf{C}}(t) \odot \hat{\mathbf{B}}(t-1))
    $$
    $$
    \mathbf{K}_A(t) = \frac{\mathbf{P}_A(t-1) \mathbf{V}_A(t)}{\lambda + \mathbf{V}_A^H(t) \mathbf{P}_A(t-1) \mathbf{V}_A(t)} \quad (\text{增益矩阵})
    $$
    $$
    \mathbf{P}_A(t) = \lambda^{-1} \left( \mathbf{P}_A(t-1) - \mathbf{K}_A(t) \mathbf{V}_A^H(t) \mathbf{P}_A(t-1) \right)
    $$
    $$
    \hat{\mathbf{A}}(t) = \hat{\mathbf{A}}(t-1) + \left( \mathbf{Y}_{(1)}(t) - \hat{\mathbf{A}}(t-1)\mathbf{V}_A^H(t) \right) \mathbf{K}_A^H(t)
    $$
    *注：$\mathbf{Y}_{(1)}(t)$ 是张量 $\mathcal{Y}(t)$ 的模-1展开形式。*

4.  **更新 B（X轴导向矩阵）**：
    类似于更新 $\mathbf{A}$，使用 $\hat{\mathbf{A}}(t)$ 和 $\hat{\mathbf{C}}(t)$ 来更新 $\hat{\mathbf{B}}(t)$。

### 3.4 步骤三：参数提取
在每个时刻 $t$，从更新后的 $\hat{\mathbf{A}}(t)$ 和 $\hat{\mathbf{B}}(t)$ 中提取角度。
利用 Z 轴导向矢量结构：
$$
\hat{\mathbf{a}}_{z,k}(t) = [1, e^{-j\mu_z}, \dots, e^{-j(M-1)\mu_z}]^T, \quad \mu_z = \frac{2\pi d}{\lambda} \cos(\theta_k(t))
$$
利用 Total Least Squares (TLS) 或相位法估计 $\mu_z$：
$$
\hat{\theta}_k(t) = \arccos\left( \frac{\text{angle}(\hat{\mathbf{a}}_{z,k}(t)) \cdot \lambda}{2\pi d} \right)
$$
同理，从 $\hat{\mathbf{B}}(t)$ 中提取 $\mu_x = \frac{2\pi d}{\lambda} \sin(\theta_k(t))\cos(\phi_k(t))$。结合已知的 $\hat{\theta}_k(t)$，解出 $\hat{\phi}_k(t)$。

---

## 4. 公式推导细节

**关于RLS更新中梯度的展开：**

定义代价函数关于 $\mathbf{A}$ 的局部项：
$$
J_A(t) = \left\| \mathbf{Y}_{(1)}(t) - \mathbf{A}(t) (\mathbf{C}(t) \odot \mathbf{B}(t-1))^T \right\|_F^2
$$
令 $\mathbf{H}(t) = (\mathbf{C}(t) \odot \mathbf{B}(t-1))^T$。
我们需要递归最小化 $\sum \lambda^{t-i} \|\mathbf{Y}_{(1)}(i) - \mathbf{A}\mathbf{H}(i)\|_F^2$。
最优解形式为：
$$
\mathbf{A}(t) = \mathbf{\Phi}_{YH}(t) \mathbf{\Phi}_{HH}^{-1}(t)
$$
其中互相关矩阵和自相关矩阵的递归形式为：
$$
\mathbf{\Phi}_{HH}(t) = \lambda \mathbf{\Phi}_{HH}(t-1) + \mathbf{H}(t)\mathbf{H}^H(t)
$$
$$
\mathbf{\Phi}_{YH}(t) = \lambda \mathbf{\Phi}_{YH}(t-1) + \mathbf{Y}_{(1)}(t)\mathbf{H}^H(t)
$$
根据矩阵求逆引理（Woodbury Matrix Identity），令 $\mathbf{P}(t) = \mathbf{\Phi}_{HH}^{-1}(t)$，则有：
$$
\mathbf{P}(t) = \lambda^{-1} \mathbf{P}(t-1) - \lambda^{-1} \mathbf{k}(t) \mathbf{H}^H(t) \mathbf{P}(t-1)
$$
其中卡尔曼增益向量（或矩阵列）为：
$$
\mathbf{k}(t) = \frac{\lambda^{-1} \mathbf{P}(t-1)\mathbf{H}(t)}{1 + \lambda^{-1} \mathbf{H}^H(t) \mathbf{P}(t-1) \mathbf{H}(t)}
$$
最终得到参数矩阵 $\mathbf{A}$ 的更新公式：
$$
\mathbf{A}(t) = \mathbf{A}(t-1) + \underbrace{(\mathbf{Y}_{(1)}(t) - \mathbf{A}(t-1)\mathbf{H}(t))}_{\text{预测误差}} \mathbf{k}^H(t)
$$
这证明了算法能够以 $O(K^2)$ 或 $O(MK)$ 的复杂度实时更新导向矩阵，而无需进行 $O(M^3)$ 的SVD或特征分解。

---

## 5. 实验设计与验证方案

### 5.1 仿真环境设置
*   **阵列参数**：$M=6$ (Z轴), $M=6$ (X轴)，总阵元 $11$ 个。阵元间距 $d=\lambda/2$。
*   **信源设置**：$K=15$ 个信源（$K > 2M-1$，验证超分辨能力）。
*   **信号类型**：QPSK调制信号，载波频率 2GHz。
*   **运动轨迹**：信源角度随时间按正弦函数或线性函数变化，模拟交叉轨迹和近距离并行轨迹。

### 5.2 实验内容
1.  **静态超分辨验证**：固定角度，对比本方案（基于单次快拍构建张量）与传统2D-MUSIC在 $K > M$ 时的分辨成功率。
2.  **动态跟踪性能**：
    *   **指标**：均方根误差（RMSE） vs. 时间快拍数。
    *   **对比算法**：
        *   Batch PARAFAC (ALS)：作为性能上界，但计算耗时高。
        *   PAST (Projection Approximation Subspace Tracking)：经典的子空间跟踪算法。
        *   Proposed Adaptive Tensor Method。
3.  **鲁棒性测试**：
    *   改变信噪比（SNR）从 -10dB 到 20dB，观察RMSE变化。
    *   改变遗忘因子 $\lambda$，分析跟踪滞后与稳态误差的权衡。
4.  **计算复杂度分析**：记录每一步迭代的平均CPU执行时间，验证实时性。

---

## 6. 预期贡献与影响

*   **理论贡献**：提出了一种在欠定条件下（Underdetermined Case）处理动态DOA估计的新框架，证明了通过自适应张量分解可以实时跟踪虚拟阵列流形。
*   **技术突破**：解决了L型阵列在信源数超过物理阵元数时无法实时跟踪的难题，将张量方法的高精度与自适应滤波的快响应相结合。
*   **应用价值**：为低成本雷达（阵元少）和大规模MIMO系统（需要处理大量多径）提供了高效的信道参数估计算法，特别适用于无人机集群监测或车联网V2X通信场景。

---

## 7. 可能的挑战与解决方案

1.  **挑战：误差传播（Error Propagation）**
    *   *问题*：RLS算法中，如果初始化偏差大或中间出现野值，误差会累积导致跟踪发散。
    *   *解决方案*：引入**周期性重置（Periodic Resetting）**机制，每隔一定时间段运行一次小批量的ALS算法校准因子矩阵；或者采用**变遗忘因子（Variable Forgetting Factor）**策略，当预测误差大时减小 $\lambda$ 以快速纠正。

2.  **挑战：信源相干性（Source Coherence）**
    *   *问题*：多径环境下信源高度相关，导致秩亏，PARAFAC分解唯一性受损。
    *   *解决方案*：在构建虚拟张量 $\mathcal{Y}$ 时，引入**空间平滑（Spatial Smoothing）**技术，即在虚拟面阵上进行子阵列平滑，恢复信号秩。

3.  **挑战：计算复杂度与维数灾难**
    *   *问题*：虽然RLS比ALS快，但如果虚拟阵列维度很大，计算量仍可观。
    *   *解决方案*：利用L型阵列导向矢量的**稀疏性**或**低秩特性**，结合压缩感知（Compressed Sensing）思想，仅更新主要的大系数，或使用基于随机草图（Random Sketching）的张量更新算法。
