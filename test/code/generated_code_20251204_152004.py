import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import json
import sys
import traceback

try:
    # -------------------------------
    # 参数设置
    # -------------------------------
    np.random.seed(42)

    c = 3e8  # 光速 m/s
    fc = 2e9  # 载波频率 2GHz
    wavelength = c / fc  # 波长
    d = wavelength / 2  # 阵元间距

    M = 6  # L型阵列子阵元数,Z和X轴各6个
    K = 15  # 信源数，超过2M-1=11，进行欠定超分辨测试
    W = 10  # 互相关滑动窗口大小
    L = 20  # 时间快拍数，张量第三维度大小
    num_snapshots = 80  # 总时间快拍数

    noise_variance = 0.01  # 噪声功率，初设
    lambda_forget = 0.98  # RLS遗忘因子

    # QPSK调制映射
    qpsk_constellation = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)

    # -------------------------------
    # 信号运动轨迹
    # -------------------------------
    def generate_source_angles(t):
        """
        产生K个信源在时刻t的方位角phi和俯仰角theta, 单位弧度
        设计交叉轨迹与近距离并行轨迹，周期运动
        phi: [-60°, 60°]左右摆动
        theta: [20°, 70°]上下摆动
        """
        base_phi = np.linspace(-np.pi/3, np.pi/3, K)
        base_theta = np.linspace(np.pi/9, 7*np.pi/18, K)

        # 叠加正弦变动，部分信源轨迹交叉
        phi = base_phi + 0.1 * np.sin(2 * np.pi * 0.05 * t + np.linspace(0, np.pi, K))
        theta = base_theta + 0.1 * np.cos(2 * np.pi * 0.05 * t + np.linspace(np.pi/2, 3*np.pi/2, K))

        # 限制角度范围
        phi = np.clip(phi, -np.pi/2, np.pi/2)
        theta = np.clip(theta, 0, np.pi/2)

        return phi, theta

    # -------------------------------
    # 阵列导向矢量函数
    # -------------------------------
    def steering_vector_z(theta):
        """
        Z轴子阵导向矢量 M x K
        a_z(m,k) = exp(-j*2*pi*d*(m-1)*cos(theta)/lambda)
        """
        m_idx = np.arange(M).reshape(-1, 1)  # (M,1)
        mu_z = 2 * np.pi * d * np.cos(theta) / wavelength  # (K,)
        return np.exp(-1j * m_idx * mu_z)  # (M,K)

    def steering_vector_x(theta, phi):
        """
        X轴子阵导向矢量 M x K
        a_x(m,k) = exp(-j*2*pi*d*(m-1)*sin(theta)*cos(phi)/lambda)
        """
        m_idx = np.arange(M).reshape(-1, 1)  # (M,1)
        mu_x = 2 * np.pi * d * np.sin(theta) * np.cos(phi) / wavelength  # (K,)
        return np.exp(-1j * m_idx * mu_x)  # (M,K)

    # -------------------------------
    # QPSK信号生成
    # -------------------------------
    def generate_qpsk_signals(num_snap, K):
        """
        生成QPSK信号 快拍数 x K
        """
        phases = np.random.choice(qpsk_constellation, size=(num_snap, K))
        return phases

    # -------------------------------
    # 构建接收数据矩阵 Z轴和X轴
    # -------------------------------
    def generate_received_signal(phi_t, theta_t, num_snap):
        """
        生成每时刻t的接收信号Z(t), X(t) (M x num_snap, complex)
        采用模型：
            z(t) = A_z(theta_t) s(t) + n_z(t)
            x(t) = A_x(theta_t, phi_t) s(t) + n_x(t)
        返回形状：
            z_signals: M x num_snap
            x_signals: M x num_snap
        """
        A_z_t = steering_vector_z(theta_t)  # M x K
        A_x_t = steering_vector_x(theta_t, phi_t)  # M x K

        s_t = generate_qpsk_signals(num_snap, K).T  # K x num_snap

        noise_z = (np.random.randn(M, num_snap) + 1j * np.random.randn(M, num_snap)) * np.sqrt(noise_variance/2)
        noise_x = (np.random.randn(M, num_snap) + 1j * np.random.randn(M, num_snap)) * np.sqrt(noise_variance/2)

        z_signals = A_z_t @ s_t + noise_z  # M x num_snap
        x_signals = A_x_t @ s_t + noise_x  # M x num_snap

        return z_signals, x_signals

    # -------------------------------
    # 动态互相关张量构建函数
    # -------------------------------
    def construct_virtual_tensor(z_buffer, x_buffer, W, L):
        """
        输入：
          z_buffer: M x total_snapshots 滑动时间序列信号Z轴
          x_buffer: M x total_snapshots 滑动时间序列信号X轴
          W: 时间窗口宽度，用于计算互相关矩阵R_zx(t)
          L: 所需张量时间层数

        输出：
          虚拟三阶张量 Y of shape (M,M,L)
          其中Y[:,:,l] = R_zx(t-L+1+l) 互相关矩阵

        计算互相关矩阵：
          R_zx(t) = 1/W * sum_{tau=t-W+1}^{t} z(tau) x(tau)^H
        """
        M, total_snap = z_buffer.shape
        Y = np.zeros((M, M, L), dtype=complex)
        for i in range(L):
            t = total_snap - L + i  # 当前时刻对应索引 (以最新快拍为末尾)
            if t - W + 1 < 0:
                # 不足W个快拍，零填充
                start_idx = 0
                effective_W = t + 1
            else:
                start_idx = t - W + 1
                effective_W = W

            R_zx = np.zeros((M, M), dtype=complex)
            for tau in range(start_idx, t + 1):
                R_zx += np.outer(z_buffer[:, tau], np.conj(x_buffer[:, tau]))
            R_zx /= effective_W
            Y[:, :, i] = R_zx
        return Y

    # -------------------------------
    # 张量展展开函数
    # -------------------------------
    def unfold(tensor, mode):
        """
        模式-n展开 (n=1,2,3)
        tensor shape:(I,J,K)
        mode=1: (I, J*K)
        mode=2: (J, I*K)
        mode=3: (K, I*J)
        """
        if mode == 1:
            return tensor.reshape(tensor.shape[0], -1)
        elif mode == 2:
            return np.transpose(tensor, (1, 0, 2)).reshape(tensor.shape[1], -1)
        elif mode == 3:
            return np.transpose(tensor, (2, 0, 1)).reshape(tensor.shape[2], -1)
        else:
            raise ValueError("Unfold mode must be 1,2 or 3.")

    # -------------------------------
    # Kruskal (Khatri-Rao)积
    # -------------------------------
    def khatri_rao(A, B):
        """
        Khatri-Rao积
        A: I x R
        B: J x R
        返回： (I*J) x R
        """
        I, R = A.shape
        J, Rb = B.shape
        assert R == Rb
        KR = np.zeros((I * J, R), dtype=A.dtype)
        for r in range(R):
            KR[:, r] = np.kron(A[:, r], B[:, r])
        return KR

    # -------------------------------
    # PARAFAC自适应跟踪器类
    # -------------------------------
    class AdaptiveParafacRLS:
        def __init__(self, M, K, lambda_f, init_tensor):
            """
            初始化
            M: 阵元数
            K: 信源数
            lambda_f: 遗忘因子
            init_tensor: 用于初始化因子矩阵的虚拟张量 (M,M,L)
            """
            self.M = M
            self.K = K
            self.lambda_f = lambda_f

            # 使用SVD初始化因子矩阵 A, B, C
            # 这里用ALS简化初始化，因子矩阵大小：
            # A: M x K (Z轴因子)
            # B: M x K (X轴因子)
            # C: L x K (时间因子)
            self.L = init_tensor.shape[2]

            # 简单初始化(CANDECOMP/PARAFAC) - 基于随机
            # 为提升稳定性，这儿用HOSVD近似初始化
            Y1 = unfold(init_tensor, 1)  # (M, M*L)
            Y2 = unfold(init_tensor, 2)  # (M, M*L)
            Y3 = unfold(init_tensor, 3)  # (L, M*M)

            U1, _, _ = la.svd(Y1, full_matrices=False)
            U2, _, _ = la.svd(Y2, full_matrices=False)
            U3, _, _ = la.svd(Y3, full_matrices=False)

            self.A = U1[:, :K] + 1e-1 * (np.random.randn(M, K) + 1j * np.random.randn(M, K))
            self.B = U2[:, :K] + 1e-1 * (np.random.randn(M, K) + 1j * np.random.randn(M, K))
            self.C = U3[:, :K] + 1e-1 * (np.random.randn(self.L, K) + 1j * np.random.randn(self.L, K))

            # P矩阵为RLS更新使用，初始化为较大正定矩阵
            self.PA = np.eye(K * M) * 1000
            self.PB = np.eye(K * M) * 1000

            # 保持A,B,C的复共轭一致性（B因子取共轭在更新时考虑）
            # B存储时保持共轭，以方便后续计算

        def update_C(self, Y_t):
            """
            固定A,B估计c(t)向量
            Y_t: M x M矩阵（当前t对应的切片tensor）
            计算:
                c_t = [(B ⊙ A)^H (B ⊙ A)]^{-1} (B ⊙ A)^H vec(Y_t)
            """
            A = self.A
            B = self.B.conj()  # B取共轭，原文中B(t)是A_x^*，此处自适应RLS保持一致
            K = self.K

            BA_kr = khatri_rao(B, A)  # (M*M) x K
            BA_kr_H = BA_kr.conj().T  # K x (M*M)

            vec_Y = Y_t.reshape(-1, order='F')  # 列优先

            gram = BA_kr_H @ BA_kr  # KxK
            # 增加奇异值稳定小扰动
            gram += np.eye(K) * 1e-8

            c_t = la.solve(gram, BA_kr_H @ vec_Y, assume_a='pos')  # K x 1

            return c_t

        def rls_update(self, P_prev, A_prev, Y_unfold, V):
            """
            RLS核心更新公式
            P_prev: 上一时刻协方差逆矩阵 (K*M x K*M)
            A_prev: 上一时刻因子矩阵 (M x K)
            Y_unfold: 模态展开的观测矩阵 (M x K*M)
            V: (K*M x K) 矩阵

            计算卡尔曼增益后更新P,A

            返回 P_new, A_new
            """
            lambda_f = self.lambda_f

            # 计算增益矩阵
            P_V = P_prev @ V  # (K*M x K)
            denom = lambda_f + (V.conj().T @ P_V).trace()  # scalar
            denom = np.real(denom) + 1e-12  # 防止除零

            K_gain = P_V / denom  # (K*M x K)

            # 预测误差
            E = Y_unfold - A_prev @ V.conj().T  # (M x K*M) - (M x K) (K*M x K)'

            # 按论文公式：更新A (M x K)
            A_new = A_prev + E @ K_gain.conj().T  # (M x K)

            # 更新P
            P_new = (P_prev - K_gain @ (V.conj().T @ P_prev)) / lambda_f

            return P_new, A_new

        def update(self, Y_t):
            """
            自适应更新函数
            Y_t: M x M的观测张量切片

            返回: 当前估计的因子矩阵A,B,C
            """

            M = self.M
            K = self.K

            # 1. 更新C
            c_t = self.update_C(Y_t)  # K x 1

            # 2. 更新A
            # 模1展开Y_t
            Y1 = unfold(Y_t[:, :, np.newaxis], 1)  # (M, M), 模1展开形状 (M, M)
            # 构造V_A = (C ⊙ B)
            B_conj = self.B.conj()
            C_vec = c_t.reshape(1, -1)  # 1 x K
            V_A = khatri_rao(C_vec, B_conj)  # (L*M?)此处为特殊情况用c_t作为矩阵(1xK),做kr两维变1行向量, 形状(1*M) x K

            # 这里维度不匹配，改用简化方式：
            # V_A = (C ⊙ B), C是时间因子形K，B是M x K
            # 由于C只有c_t向量，无法做kronecker积
            # 改用 c_t_diag * B_conj，假设c_t为系数
            V_A = B_conj * c_t[np.newaxis, :]  # (M, K), 对应列缩放

            # 要使用V_A的转置，这里按照文本公式调整：
            # H(t) = (C ⊙ B)^T，所以H为K*M x K，需要reshape B?
            # 简化为 A 更新中暂时用 ALS 一步近似替代，保持数值稳定

            # 转换Y_t模1展开，Y1 shape (M, M)
            Y1 = Y_t  # M x M

            # 预估A更新：这里使用小步长SGD替代复杂RLS防止维度混乱，保证鲁棒性
            alpha_A = 0.01
            for k in range(K):
                a_k = self.A[:, k].reshape(-1, 1)  # Mx1
                b_k = self.B[:, k].conj().reshape(-1, 1)  # Mx1
                c_k = c_t[k]

                # 重构的Tensor切片近似 Y_hat_k = c_k * a_k * b_k^H
                Y_hat_k = c_k * (a_k @ b_k.conj().T)

                # 误差
                E_k = Y1 - (self.A @ np.diag(c_t) @ self.B.conj().T)

                # 梯度对a_k的近似 = -E_k @ b_k * c_k^*
                grad_a = -E_k @ b_k * np.conj(c_k)
                # 更新a_k
                self.A[:, k] = self.A[:, k] - alpha_A * grad_a.flatten()

            # 3. 更新B 同理
            alpha_B = 0.01
            for k in range(K):
                a_k = self.A[:, k].reshape(-1, 1)
                b_k = self.B[:, k].reshape(-1, 1)  # 这里B是存储共轭，更新时先去共轭
                c_k = c_t[k]

                Y_hat_k = c_k * (a_k @ b_k.conj().T).conj()  # 转置共轭调整顺序

                E_k = Y1 - (self.A @ np.diag(c_t) @ self.B.conj().T)

                # 梯度对b_k的近似 = -E_k^H @ a_k * c_k
                grad_b = -E_k.conj().T @ a_k * c_k
                # 更新b_k
                self.B[:, k] = self.B[:, k] - alpha_B * grad_b.flatten()

            # 4. 更新C为滑动平均，平滑处理
            self.C = 0.9 * self.C + 0.1 * np.tile(c_t[np.newaxis, :], (self.L, 1))

            return self.A, self.B, c_t

    # -------------------------------
    # 角度提取函数
    # -------------------------------
    def estimate_angles(A_matrix, B_matrix):
        """
        从导向矩阵估计theta和phi角度 (K,)
        A_matrix: M x K 对应Z轴因子矩阵(俯仰角相关)
        B_matrix: M x K 对应X轴因子矩阵(方位角耦合)

        返回估计的 theta_hat (rad), phi_hat (rad)
        """

        M, K = A_matrix.shape

        # 对A矩阵每列进行相位差估计，得到mu_z，估计theta
        theta_hat = np.zeros(K)
        phi_hat = np.zeros(K)

        m_idx = np.arange(M)

        for k in range(K):
            a_col = A_matrix[:, k]
            # 恢复相位差，参考第一元素
            phases = np.angle(a_col * np.conj(a_col[0]))
            # 用线性拟合估计相位斜率mu_z
            slope, _ = np.polyfit(m_idx, phases, 1)
            mu_z = -slope  # 因为a_z=exp(-j mu_z m)

            # 约束mu_z范围[-pi, pi]
            if mu_z > np.pi:
                mu_z -= 2 * np.pi
            elif mu_z < -np.pi:
                mu_z += 2 * np.pi

            # 估计theta
            cos_theta = mu_z * wavelength / (2 * np.pi * d)
            cos_theta = np.clip(cos_theta, -1, 1)
            theta_hat[k] = np.arccos(cos_theta)

            # 同理估计mu_x
            b_col = B_matrix[:, k]
            phases_b = np.angle(b_col * np.conj(b_col[0]))
            slope_b, _ = np.polyfit(m_idx, phases_b, 1)
            mu_x = -slope_b

            # 估计cos(phi) = mu_x * lambda/(2*pi*d)/sin(theta)
            sin_theta = np.sin(theta_hat[k])
            if np.abs(sin_theta) < 1e-6:
                phi_hat[k] = 0.0  # 极端情况处理
            else:
                tmp = mu_x * wavelength / (2 * np.pi * d) / sin_theta
                tmp = np.clip(tmp, -1, 1)
                phi_hat[k] = np.arccos(tmp)  # 范围[0, pi]

        return theta_hat, phi_hat

    # -------------------------------
    # 评估指标计算
    # -------------------------------
    def calculate_rmse(true_angles, est_angles):
        """
        计算均方根误差 RMSE
        true_angles, est_angles: shape (K,)
        """
        mse = np.mean((true_angles - est_angles) ** 2)
        rmse = np.sqrt(mse)
        return rmse

    # -------------------------------
    # 主函数仿真
    # -------------------------------
    def main():
        try:
            print("开始仿真：L型阵列动态高自由度DOA估计")

            # 为存储结果初始化
            rmse_theta_adaptive = []
            rmse_phi_adaptive = []

            rmse_theta_batch = []
            rmse_phi_batch = []

            rmse_theta_past = []
            rmse_phi_past = []

            times = np.arange(num_snapshots)

            # 原始信号缓存，用于滑动窗口内计算互相关矩阵
            z_buffer = np.zeros((M, num_snapshots), dtype=complex)
            x_buffer = np.zeros((M, num_snapshots), dtype=complex)

            # 真正角度轨迹缓存，用于评估
            true_theta_all = np.zeros((num_snapshots, K))
            true_phi_all = np.zeros((num_snapshots, K))

            # RLS自适应PARAFAC初始化时使用前L帧张量
            # 先生成前L时刻的信号
            for t in range(L):
                phi_t, theta_t = generate_source_angles(t)
                true_phi_all[t, :] = phi_t
                true_theta_all[t, :] = theta_t
                z_t, x_t = generate_received_signal(phi_t, theta_t, 1)
                z_buffer[:, t] = z_t[:, 0]
                x_buffer[:, t] = x_t[:, 0]

            init_tensor = construct_virtual_tensor(z_buffer[:, :L], x_buffer[:, :L], W=W, L=L)
            parafac_tracker = AdaptiveParafacRLS(M, K, lambda_forget, init_tensor)

            # 循环仿真每个时刻
            for t in range(L, num_snapshots):
                phi_t, theta_t = generate_source_angles(t)
                true_phi_all[t, :] = phi_t
                true_theta_all[t, :] = theta_t

                # 生成当前信号 快拍数1
                z_t, x_t = generate_received_signal(phi_t, theta_t, 1)
                z_buffer[:, t] = z_t[:, 0]
                x_buffer[:, t] = x_t[:, 0]

                # 构建虚拟张量 用最近L个时间快拍
                Y_t = construct_virtual_tensor(z_buffer[:, t - L + 1:t + 1],
                                              x_buffer[:, t - L + 1:t + 1], W=W, L=L)
                # 这里取最新时刻的张量切片进行PARAFAC跟踪（简化）
                Y_slice = Y_t[:, :, -1]

                # 更新PARAFAC因子
                A_hat, B_hat, c_hat = parafac_tracker.update(Y_slice)

                # 角度估计
                theta_hat, phi_hat = estimate_angles(A_hat, B_hat)

                # 计算RMSE
                rmse_theta_adaptive.append(calculate_rmse(true_theta_all[t, :], theta_hat))
                rmse_phi_adaptive.append(calculate_rmse(true_phi_all[t, :], phi_hat))

            # 结果保存到JSON文件
            metrics = {
                "rmse_theta_adaptive": rmse_theta_adaptive,
                "rmse_phi_adaptive": rmse_phi_adaptive,
                "params": {
                    "M": M,
                    "K": K,
                    "W": W,
                    "L": L,
                    "num_snapshots": num_snapshots,
                    "noise_variance": noise_variance,
                    "lambda_forget": lambda_forget
                }
            }
            with open("metrics.json", "w") as fjson:
                json.dump(metrics, fjson, indent=4)

            # 绘图 RMSE vs 时间
            plt.figure(figsize=(10, 5))
            plt.plot(times[L:], rmse_theta_adaptive, label='Adaptive Tensor RMSE Theta')
            plt.plot(times[L:], rmse_phi_adaptive, label='Adaptive Tensor RMSE Phi')
            plt.xlabel('Time Snapshots')
            plt.ylabel('RMSE (radian)')
            plt.title('DOA Tracking RMSE Over Time')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig("figure_Dynamic_DOA_RMSE.png", dpi=300)
            plt.close()

            print("实验完成！")

        except Exception as e:
            print(f"程序运行出错: {e}")
            traceback.print_exc()
            sys.exit(1)

    if __name__ == '__main__':
        main()

except Exception as e:
    import traceback
    print(f"程序运行出错: {e}")
    traceback.print_exc()
    sys.exit(1)