import numpy as np
import matplotlib.pyplot as plt
import json
import sys
from numpy.linalg import pinv, svd, norm
from scipy.linalg import hankel
from scipy.signal import correlate
import time

np.random.seed(42)


def qpsk_mod(bits):
    """QPSK调制"""
    bits = bits.reshape((-1, 2))
    mapping = {(0, 0): 1 + 1j, (0, 1): -1 + 1j, (1, 0): 1 - 1j, (1, 1): -1 - 1j}
    return np.array([mapping[tuple(b)] for b in bits]) / np.sqrt(2)


def steering_vector_L(M, d_lambda, theta, phi=None, axis="Z"):
    """
    计算L型阵列对子阵的导向矢量
    以半波长间距d_lambda = d/λ的单位表示相位，
    axis = 'Z' 或 'X'。
    对Z轴子阵仅依赖theta，X轴子阵依赖theta和phi。
    """
    m = np.arange(M)
    if axis == "Z":
        mu_z = 2 * np.pi * d_lambda * np.cos(theta)
        a = np.exp(-1j * mu_z * m)
    elif axis == "X":
        mu_x = 2 * np.pi * d_lambda * np.sin(theta) * np.cos(phi)
        a = np.exp(-1j * mu_x * m)
    else:
        raise ValueError("axis must be 'Z' or 'X'")
    return a.reshape(-1, 1)  # Mx1


def modulate_signal(K, N_snapshots):
    """生成K个QPSK信源信号，长度N_snapshots"""
    bits = np.random.randint(0, 2, size=(K * N_snapshots * 2,))
    symbols = qpsk_mod(bits)
    return symbols.reshape((K, N_snapshots))


def generate_DOA_trajectories(K, T, theta_range, phi_range):
    """
    生成信源动态轨迹：随时间变化的theta和phi
    theta_range和phi_range为角度范围 (radian)
    轨迹叠加正弦、线性组合产生交叉和近距离并行轨迹
    """
    t = np.linspace(0, 2 * np.pi, T)
    theta = np.zeros((K, T))
    phi = np.zeros((K, T))

    for k in range(K):
        freq_theta = 0.1 + 0.05 * k
        freq_phi = 0.07 + 0.04 * k
        offset_theta = theta_range[0] + (theta_range[1] - theta_range[0]) * k / K
        offset_phi = phi_range[0] + (phi_range[1] - phi_range[0]) * ((K - k) / K)
        theta[k, :] = offset_theta + 0.1 * np.sin(freq_theta * t + k)
        phi[k, :] = offset_phi + 0.1 * np.cos(freq_phi * t + k * 1.5)
    return theta, phi


def add_awgn_noise(signal, SNR_dB):
    """添加高斯白噪声"""
    signal_power = np.mean(np.abs(signal) ** 2)
    noise_power = signal_power / (10 ** (SNR_dB / 10))
    noise = np.sqrt(noise_power / 2) * (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape))
    return signal + noise


def tensor_mode_unfold(tensor, mode):
    """
    张量mode-n展开
    tensor: 多维复数数组
    mode: 模型，0-based
    """
    return np.moveaxis(tensor, mode, 0).reshape(tensor.shape[mode], -1)


def cpd_reconstruct(A, B, C):
    """
    根据因子矩阵构建的三阶CPD张量
    Y approx sum_r a_r ∘ b_r ∘ c_r
    A: (I, R)
    B: (J, R)
    C: (K, R)
    返回: (I, J, K)
    """
    I, R = A.shape
    J = B.shape[0]
    K = C.shape[0]
    Y = np.zeros((I, J, K), dtype=np.complex128)
    for r in range(R):
        Y += np.outer(np.outer(A[:, r], B[:, r]).reshape(I, J), C[:, r]).reshape(I, J, K)
    return Y


def cp_als(Y, rank, max_iter=100, tol=1e-6):
    """批处理CP-ALS算法，作为周期性重置校准基准"""
    I, J, K = Y.shape
    A = np.random.randn(I, rank) + 1j * np.random.randn(I, rank)
    B = np.random.randn(J, rank) + 1j * np.random.randn(J, rank)
    C = np.random.randn(K, rank) + 1j * np.random.randn(K, rank)

    for it in range(max_iter):
        # update A
        Z = np.kron(C, B)
        Y1 = tensor_mode_unfold(Y, 0)
        A_new = Y1 @ pinv(Z.T)

        # update B
        Z = np.kron(C, A_new)
        Y2 = tensor_mode_unfold(Y, 1)
        B_new = Y2 @ pinv(Z.T)

        # update C
        Z = np.kron(B_new, A_new)
        Y3 = tensor_mode_unfold(Y, 2)
        C_new = Y3 @ pinv(Z.T)

        err = norm(A - A_new) + norm(B - B_new) + norm(C - C_new)
        A, B, C = A_new, B_new, C_new
        if err < tol:
            break
    return A, B, C


def estimate_angle_from_steering(a_vec, d_lambda):
    """
    利用相位差估计theta或mu
    a_vec: Mx1向量（以共轭或结构化矢量形式输入）
    返回theta = arccos(mu/(2πd/λ))
    """
    m = np.arange(len(a_vec))
    phase = np.angle(a_vec)
    # 线性拟合相位
    # 取主相位差序列，拟合直线斜率即mu
    coeff = np.polyfit(m, phase, 1)
    mu = -coeff[0]  # 反向因公式a=exp(-j*mu*m)
    mu = (mu + np.pi*2) % (2*np.pi)  # 归一正角度
    # 限制mu范围到0~2pi
    try:
        theta = np.arccos(mu / (2 * np.pi * d_lambda))
    except:
        theta = np.nan
    if np.isnan(theta) or np.iscomplex(theta):
        theta = np.nan
    return np.real(theta)


def estimate_phi(mu_x, theta, d_lambda):
    """
    由 mu_x = 2π d/λ sin(theta) cos(phi)
    解出phi
    """
    if np.isnan(theta):
        return np.nan
    val = mu_x / (2 * np.pi * d_lambda * np.sin(theta))
    val = np.clip(val, -1, 1)
    phi = np.arccos(val)
    return phi


class AdaptivePARAFACTracker:
    def __init__(self, M, K, lambda_forget=0.96):
        """
        初始化自适应PARAFAC跟踪器
        M: 阵元数
        K: 信源数（欠定条件，K > M）
        """
        self.M = M
        self.K = K
        self.lambda_forget = lambda_forget  # 遗忘因子
        self.A = np.random.randn(M, K) + 1j * np.random.randn(M, K)  # Z轴因子矩阵
        self.B = np.random.randn(M, K) + 1j * np.random.randn(M, K)  # X轴因子矩阵（共轭）
        self.C = np.zeros((None, K), dtype=np.complex128)  # 时间因子，动态更新
        self.P_A = np.eye(K, dtype=np.complex128) * 1000  # A因子的RLS协方差逆矩阵，KxK
        self.P_B = np.eye(K, dtype=np.complex128) * 1000  # B因子的RLS协方差逆矩阵，KxK

        # 初始化时间因子使用第0时刻估计
        self.C = np.zeros((1, K), dtype=np.complex128)

    def update_C(self, Y, A, B):
        """更新C因子（信号时间向量）-闭式最小二乘解"""
        M, _, L = Y.shape
        # 通过式子： vec(Y) ≈ (B ⊙ A) c
        # 计算重建矩阵
        AB = np.kron(B.conj(), A)  # M*L x K
        Y_vec = Y.reshape((-1,), order='F')  # 向量化，列优先
        try:
            c_hat = pinv(AB) @ Y_vec
        except np.linalg.LinAlgError:
            c_hat = np.zeros((self.K,), dtype=np.complex128)
        return c_hat

    def rls_update(self, P_prev, H, A_prev, Y_unfold, lambda_f):
        """
        RLS迭代更新因子矩阵A或B
        P_prev: 协方差矩阵(逆) KxK
        H: (K x N)某种预测矩阵 (size K x N)
        A_prev: MxK 因子矩阵
        Y_unfold: MxN - 张量mode-n展开
        返回: A_new, P_new, Kalman增益K
        """
        # 求解增益
        try:
            denom = lambda_f + np.conj(H).T @ P_prev @ H
            K_gain = (P_prev @ H) / denom
            # 更新P
            P_new = (P_prev - K_gain @ np.conj(H).T @ P_prev) / lambda_f
            # 预测误差
            E = Y_unfold - A_prev @ np.conj(H).T
            A_new = A_prev + E @ np.conj(K_gain).T
        except Exception:
            # 数值异常时保持不变，报错退出
            raise
        return A_new, P_new, K_gain

    def update(self, Y):
        """
        单步更新
        Y: MxMxL张量
        """
        M, _, L = Y.shape

        # E步：保持A,B不变，更新C
        c_new = self.update_C(Y, self.A, self.B)
        self.C = np.vstack([self.C, c_new.reshape(1, -1)])

        # 准备模式1展开，更新A
        Y1 = tensor_mode_unfold(Y, 0)  # M x (M*L)
        V_A = np.kron(c_new.conj(), self.B)  # (M*L) x K
        # 计算H_A需转置为K x N，需要把V_A shape调整
        H_A = V_A.T  # K x (M*L)
        self.A, self.P_A, _ = self.rls_update(self.P_A, H_A, self.A, Y1, self.lambda_forget)

        # 模式2展开，更新B
        Y2 = tensor_mode_unfold(Y, 1)  # M x (M*L)
        V_B = np.kron(c_new.conj(), self.A)  # (M*L) x K
        H_B = V_B.T  # K x (M*L)
        self.B, self.P_B, _ = self.rls_update(self.P_B, H_B, self.B, Y2, self.lambda_forget)

    def get_estimated_angles(self, d_lambda):
        """
        从因子矩阵提取theta和phi
        返回 arrays theta_hat, phi_hat (均为rad)
        """
        theta_hat = np.zeros(self.K)
        phi_hat = np.zeros(self.K)
        for k in range(self.K):
            a_z = self.A[:, k]
            a_x = self.B[:, k]
            # theta由Z轴导向矢量估计
            theta_hat[k] = estimate_angle_from_steering(a_z, d_lambda)
            # mu_x估计计算
            m = np.arange(self.M)
            phase_x = np.angle(a_x)
            coeff_x = np.polyfit(m, phase_x, 1)
            mu_x = -coeff_x[0]
            # phi估计依赖theta和mu_x
            if not np.isnan(theta_hat[k]):
                phi_hat[k] = estimate_phi(mu_x, theta_hat[k], d_lambda)
            else:
                phi_hat[k] = np.nan
        return theta_hat, phi_hat


def simulate_signals(M, K, d_lambda, N_snapshots, theta_true, phi_true, SNR_dB):
    """
    生成L型阵列接收信号及滑动窗口互相关张量
    M: 子阵元数
    K: 信源个数
    d_lambda: 阵元间距归一化(单位λ)
    N_snapshots: 快拍长度
    theta_true, phi_true: KxN真实角度
    """

    # 导向矩阵时间序列
    Az_list = []
    Ax_list = []
    for k in range(N_snapshots):
        A_z = np.hstack([steering_vector_L(M, d_lambda, theta_true[i, k], axis="Z") for i in range(K)])  # MxK
        A_x = np.hstack([steering_vector_L(M, d_lambda, theta_true[i, k], phi_true[i, k], axis="X") for i in range(K)])  # MxK
        Az_list.append(A_z)
        Ax_list.append(A_x)

    # 生成信号符号，QPSK，KxN_snapshots
    S = modulate_signal(K, N_snapshots)  # KxN_snapshots

    # 接收信号，两个子阵列，添加噪声
    Z_snapshots = []
    X_snapshots = []
    for k in range(N_snapshots):
        z_k = Az_list[k] @ S[:, k]  # Mx1
        x_k = Ax_list[k] @ S[:, k]  # Mx1
        z_k = add_awgn_noise(z_k, SNR_dB)
        x_k = add_awgn_noise(x_k, SNR_dB)
        Z_snapshots.append(z_k)
        X_snapshots.append(x_k)

    Z_snapshots = np.array(Z_snapshots).transpose(1, 0)  # M x N_snapshots
    X_snapshots = np.array(X_snapshots).transpose(1, 0)  # M x N_snapshots

    return Z_snapshots, X_snapshots


def construct_correlation_tensor(Z_snapshots, X_snapshots, W, L):
    """
    根据滑动窗口W和连续L个时间段，构建三阶互相关张量Y ∈ C^{M×M×L}
    Y(:,:,l) = R_zx(t - L + l) = 1/W sum_tau z(tau) x^H(tau)
    """
    M, N_snapshots = Z_snapshots.shape
    if N_snapshots < W + L - 1:
        raise ValueError("数据长度不足以构建张量")
    Y = np.zeros((M, M, L), dtype=np.complex128)
    for l in range(L):
        t = W + l - 1
        z_win = Z_snapshots[:, t - W + 1:t + 1]  # M x W
        x_win = X_snapshots[:, t - W + 1:t + 1]  # M x W
        R_zx = (z_win @ x_win.conj().T) / W  # M x M
        Y[:, :, l] = R_zx
    return Y


def rmse_angle(est_angles, true_angles):
    """
    计算RMSE，输入角度都应为rad，不存在nan
    """
    valid_mask = ~np.isnan(est_angles)
    if np.sum(valid_mask) == 0:
        return np.nan
    error = est_angles[valid_mask] - true_angles[valid_mask]
    error = np.arctan2(np.sin(error), np.cos(error))  # 角度差值规范化到[-pi,pi]
    mse = np.mean(error ** 2)
    return np.sqrt(mse)


def main():
    try:
        # 仿真环境参数
        M = 6  # Z轴子阵元数
        K = 15  # 信源数，欠定条件，K > M*2-1=11
        d_lambda = 0.5  # 阵元间距，单位λ，半波长
        N_snapshots = 200  # 总快拍数
        W = 10  # 滑动窗口大小
        L = 5   # 连续互相关矩阵数，构成三阶张量

        carrier_freq = 2e9  # 2GHz
        c = 3e8  # 光速
        wavelength = c / carrier_freq

        # 生成信源动态轨迹 theta和phi [rad]
        theta_true, phi_true = generate_DOA_trajectories(K, N_snapshots,
                                                         theta_range=(np.pi / 6, np.pi / 3),
                                                         phi_range=(0, np.pi / 3))

        # SNR设置，动态跟踪性能测试，多个SNR
        SNR_dBs = np.arange(-10, 21, 10)  # -10, 0, 10, 20 dB

        # 遗忘因子，用于分析稳态误差
        lambda_values = [0.90, 0.95, 0.98]

        # 评价指标存储结构
        results = {
            'RMSE_theta': {},
            'RMSE_phi': {},
            'time_per_iter': {}
        }

        plt.figure(figsize=(12, 8))

        for lambda_f in lambda_values:
            rmse_theta_all = []
            rmse_phi_all = []
            time_all = []

            for SNR_dB in SNR_dBs:
                # 生成信号和接收数据
                Z_snapshots, X_snapshots = simulate_signals(M, K, d_lambda, N_snapshots, theta_true, phi_true, SNR_dB)

                tracker = AdaptivePARAFACTracker(M, K, lambda_forget=lambda_f)

                rmse_theta_track = []
                rmse_phi_track = []

                # 模拟滑动窗口快速迭代
                for t_idx in range(W + L - 1, N_snapshots):
                    start_time = time.time()
                    # 构建三阶张量
                    tensor_Y = construct_correlation_tensor(Z_snapshots, X_snapshots, W, L)

                    # 跟踪更新
                    tracker.update(tensor_Y)

                    # 提取角度估计
                    theta_est, phi_est = tracker.get_estimated_angles(d_lambda)

                    # 统计RMSE角度（与真实角度对齐，取当前时刻对应的角度）
                    theta_true_now = theta_true[:, t_idx]
                    phi_true_now = phi_true[:, t_idx]

                    rmse_t = rmse_angle(theta_est, theta_true_now)
                    rmse_p = rmse_angle(phi_est, phi_true_now)

                    rmse_theta_track.append(rmse_t)
                    rmse_phi_track.append(rmse_p)

                    time_all.append(time.time() - start_time)

                # 每个SNR下取平均RMSE和耗时
                mean_rmse_theta = np.nanmean(rmse_theta_track)
                mean_rmse_phi = np.nanmean(rmse_phi_track)
                mean_time = np.mean(time_all)

                rmse_theta_all.append(mean_rmse_theta)
                rmse_phi_all.append(mean_rmse_phi)

            results['RMSE_theta'][f'lambda_{lambda_f}'] = list(map(float, rmse_theta_all))
            results['RMSE_phi'][f'lambda_{lambda_f}'] = list(map(float, rmse_phi_all))
            results['time_per_iter'][f'lambda_{lambda_f}'] = float(np.mean(time_all))

            # 绘制RMSE Theta曲线
            plt.plot(SNR_dBs, rmse_theta_all, label=f'RMSE_theta λ={lambda_f}')
            # 绘制RMSE Phi曲线
            plt.plot(SNR_dBs, rmse_phi_all, linestyle='--', label=f'RMSE_phi λ={lambda_f}')

        plt.xlabel('SNR (dB)')
        plt.ylabel('RMSE (radian)')
        plt.title('Adaptive Tensor Method RMSE vs SNR for Theta and Phi')
        plt.legend()
        plt.grid(True)
        plt.savefig('figure_RMSE_vs_SNR.png')
        plt.close()

        # 保存指标到metrics.json
        with open('metrics.json', 'w') as f:
            json.dump(results, f, indent=4)

        print("实验完成！")

    except Exception:
        sys.exit(1)


if __name__ == '__main__':
    main()