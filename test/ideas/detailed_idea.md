# 基于自适应张量跟踪的L型阵列动态高自由度DOA估计研究方案

## 1. 研究背景与动机

### 1.1 研究背景
随着B5G/6G通信、车载雷达及无源定位技术的发展，对多目标二维波达方向（2D-DOA）估计的需求日益增长。L型阵列因其结构简单且能提供二维角度信息而被广泛研究。然而，传统L型阵列处理面临两大瓶颈：
1.  **物理阵元数限制**：传统子空间算法（如MUSIC、ESPRIT）可分辨的信源数受限于物理阵元数（通常为 $2M$），难以应对密集多目标场景。
2.  **动态场景计算负荷**：在信源快速移动的动态场景下，传统方法需要对每个快拍或短时间窗的数据进行独立的奇异值分解（SVD）或特征值分解（EVD），计算复杂度极高，难以满足实时性要求。

### 1.2 研究动机
**Paper_3** 提出了一种基于L型阵列构建虚拟面阵的方法，利用高阶累积量或互相关技术将自由度扩展至 $\mathcal{O}(M^2)$，突破了物理阵元限制。然而，该方法通常基于批处理（Batch Processing），不具备时变跟踪能力。
**Paper_4** 提出了基于PARAFAC分解的自适应跟踪算法（如PARAFAC-RLST），能够在线更新张量因子矩阵，适用于时变信号处理，但其直接应用于原始阵列数据时，自由度未得到提升。

本研究旨在**融合**上述两种思想，提出一种**基于自适应张量跟踪的L型阵列动态高自由度DOA估计算法**。该方案利用虚拟阵列技术扩展自由度，同时引入递归最小二乘（RLS）张量跟踪框架，实现对超过物理阵元数的移动信源进行实时、高精度的2D-DOA跟踪。

---

## 2. 核心创新点

1.  **动态虚拟张量流构建**：不同于传统的静态批处理张量构建，本方案设计了一种增量式的虚拟张量构建机制。在每个时刻 $t$，利用L型阵列接收数据实时生成代表虚拟面阵特性的高阶张量切片，既保留了虚拟阵列的高自由度特性，又适配流式处理。
2.  **双重结构化自适应跟踪**：提出一种改进的自适应PARAFAC跟踪算法。在更新因子矩阵时，不仅利用了张量的低秩特性，还强制约束因子矩阵满足L型阵列特有的范德蒙（Vandermonde）结构，从而在跟踪过程中自动实现角度解耦和配对，显著提高跟踪在低信噪比下的鲁棒性。
3.  **超分辨实时跟踪架构**：实现了在动态场景下对 $K > 2M$ 个信源的实时跟踪，解决了传统算法在“欠定”条件下无法进行动态跟踪的难题。

---

## 3. 详细技术实现方案

### 3.1 信号模型与L型阵列几何

假设L型阵列由两个正交的均匀线阵（ULA）组成，分别位于 $x$ 轴和 $z$ 轴，共用原点阵元。$x$ 轴阵元数为 $M$，$z$ 轴阵元数为 $N$（通常 $M=N$），总阵元数为 $M+N+1$。

在时刻 $t$，假设有 $K$ 个远场窄带非相干信号源入射，其二维角度为 $\{(\theta_k(t), \phi_k(t))\}_{k=1}^K$，其中 $\theta$ 为方位角，$\phi$ 为俯仰角。

接收信号模型为：
$$
\begin{aligned}
\mathbf{x}(t) &= \mathbf{A}_x(t) \mathbf{s}(t) + \mathbf{n}_x(t) \\
\mathbf{z}(t) &= \mathbf{A}_z(t) \mathbf{s}(t) + \mathbf{n}_z(t)
\end{aligned}
$$
其中，$\mathbf{A}_x(t) \in \mathbb{C}^{(M+1) \times K}$ 和 $\mathbf{A}_z(t) \in \mathbb{C}^{(N+1) \times K}$ 分别为 $x$ 轴和 $z$ 轴的导向矢量矩阵。

导向矢量元素定义为：
$$
[\mathbf{a}_x(\theta_k, \phi_k)]_m = e^{-j \frac{2\pi d}{\lambda} (m-1) \sin\theta_k \cos\phi_k}, \quad m=1,\dots,M+1
$$
$$
[\mathbf{a}_z(\theta_k, \phi_k)]_n = e^{-j \frac{2\pi d}{\lambda} (n-1) \cos\theta_k}, \quad n=1,\dots,N+1
$$

### 3.2 步骤一：动态虚拟张量构建

为了获得高自由度，我们利用互相关矩阵构建虚拟面阵张量。定义 $x$ 轴和 $z$ 轴子阵数据的互相关矩阵 $\mathbf{R}_{xz}(t)$。在动态跟踪中，我们使用指数加权移动平均（EWMA）来实时估计协方差：

$$
\hat{\mathbf{R}}_{xz}(t) = \alpha \hat{\mathbf{R}}_{xz}(t-1) + (1-\alpha) \mathbf{x}(t)\mathbf{z}^H(t)
$$
其中 $\alpha$ 为遗忘因子。理论上，$\mathbf{R}_{xz}(t) = \mathbf{A}_x(t) \mathbf{R}_s(t) \mathbf{A}_z^H(t)$。

为了应用PARAFAC模型并获得虚拟孔径，我们将 $\mathbf{R}_{xz}(t)$ 变换为一个三阶张量 $\mathcal{Y}(t)$。
**构造方法**：利用矩阵到张量的重排（Reshaping）或基于平滑技术。
为了最大化自由度（参考Paper_3思路），我们构造一个虚拟差分共阵列张量。这里采用一种简化的**互相关向量化重构法**：

将 $\hat{\mathbf{R}}_{xz}(t)$ 视为一个切片。为了进行PARAFAC跟踪，我们需要引入第三维“多样性”。这里利用子阵列平滑或时间延迟。
**本方案采用空间平滑构建法**：
将 $\hat{\mathbf{R}}_{xz}(t)$ 的元素映射到虚拟矩形阵列张量 $\mathcal{V}(t) \in \mathbb{C}^{M \times N \times K}$（假设秩为K）。
更直接的动态跟踪输入构造：
我们将 $t$ 时刻的观测张量 $\mathcal{X}_{obs}(t)$ 定义为由当前互相关矩阵构成的单快拍张量，但在算法内部维护其分解形式。
实际上，为了利用Paper 4的跟踪算法，我们定义待跟踪的目标张量结构为：
$$
\mathcal{Y}(t) \approx \mathcal{I} \times_1 \mathbf{A}_x(t) \times_2 \mathbf{A}_z^*(t) \times_3 \mathbf{D}(t)
$$
其中 $\mathbf{D}(t)$ 是包含信源功率的对角矩阵（或包含时间相关性的因子）。由于 $\mathbf{R}_{xz} = \mathbf{A}_x \mathbf{P} \mathbf{A}_z^H$，我们可以将其向量化为 $\text{vec}(\mathbf{R}_{xz}) = (\mathbf{A}_z^* \odot \mathbf{A}_x) \mathbf{p}$。
为了构建三阶张量以应用PARAFAC，我们引入人工维度（例如通过子阵列划分）。
**具体操作**：
将 $x$ 轴阵列划分为 $P_1$ 个子阵，z轴划分为 $P_2$ 个子阵。构建空间平滑互相关张量 $\mathcal{Y}(t) \in \mathbb{C}^{m_{sub} \times n_{sub} \times (P_1 P_2)}$。
此时：
$$
\mathcal{Y}(t) \approx \sum_{k=1}^K \mathbf{a}_{x,sub}(t,k) \circ \mathbf{a}_{z,sub}^*(t,k) \circ \mathbf{g}(t,k)
$$
其中 $\mathbf{g}(t,k)$ 包含了空间平滑带来的相位旋转因子。

### 3.3 步骤二：自适应PARAFAC跟踪算法 (Adaptive Tensor Tracking)

目标是最小化以下指数加权代价函数，以实时更新因子矩阵 $\mathbf{A} = \mathbf{A}_{x,sub}$, $\mathbf{B} = \mathbf{A}_{z,sub}^*$, $\mathbf{C} = \mathbf{G}$：

$$
J(t) = \sum_{i=1}^t \lambda^{t-i} \left\| \mathcal{Y}(i) - \llbracket \mathbf{A}(t), \mathbf{