import numpy as np
import matplotlib.pyplot as plt
import json
import sys
import traceback
from numpy import linalg as LA


def qpsk_modulation(bits):
    """
    QPSK调制：输入二进制(bits大小应为偶数)，输出复数符号
    """
    bits = bits.reshape(-1, 2)
    mapping = {
        (0,0): 1+1j,
        (0,1): -1+1j,
        (1,0): 1-1j,
        (1,1): -1-1j
    }
    symbols = np.array([mapping[tuple(b)] for b in bits])
    symbols /= np.sqrt(2)  # 归一化功率
    return symbols


def steering_vector(M, d, wavelength, angle_rad, axis='z'):
    """
    构建L型阵列的导向矢量
    axis: 'z' or 'x'
    对z轴为 [0, 0, (m-1)d], 对x轴为 [(m-1)d, 0, 0]
    angle_rad: (theta, phi)
    theta: 俯仰角，phi: 方位角
    wavelength: 波长
    返回: M x 1 复数向量
    """
    m_idx = np.arange(M)
    if axis == 'z':
        # 对Z轴子阵，只依赖俯仰角theta
        mu = 2 * np.pi * d / wavelength * np.cos(angle_rad[0])
        sv = np.exp(-1j * mu * m_idx)
    elif axis == 'x':
        # 对X轴子阵，依赖theta和phi
        mu = 2 * np.pi * d / wavelength * np.sin(angle_rad[0]) * np.cos(angle_rad[1])
        sv = np.exp(-1j * mu * m_idx)
    else:
        raise ValueError("axis must be 'x' or 'z'")
    return sv.reshape(-1,1)


def generate_source_angles(K, T):
    """
    生成K个信源在T个时刻的角度变化
    模拟交叉轨迹和近距离并行轨迹
    返回theta (俯仰角) 和 phi(方位角)，均为弧度制，大小为(K,T)
    """

    t = np.linspace(0, 10, T)

    theta = np.zeros((K, T))
    phi = np.zeros((K, T))

    # 前半数信源做正弦变化，部分轨迹交叉
    for k in range(K//2):
        theta[k,:] = np.deg2rad(30 + 10*np.sin(0.5*t + k))
        phi[k,:] = np.deg2rad(20 + 15*np.cos(0.3*t + k*0.5))

    # 后半数信源做线性变化，模拟近距离并行轨迹
    for k in range(K//2, K):
        theta[k,:] = np.deg2rad(40 + 0.5*t + (k-K//2)*2)
        phi[k,:] = np.deg2rad(50 + 0.3*t + (k-K//2)*3)

    return theta, phi


def generate_qpsk_signals(K, T, samples_per_block):
    """
    生成每个时刻每个信源的QPSK信号 (复数),
    signals shape: (K, T*samples_per_block)
    """
    bits_len = 2 * K * T * samples_per_block  # 每符号2比特
    bits = np.random.randint(0,2,bits_len)
    symbols = qpsk_modulation(bits)
    symbols = symbols.reshape(K, T*samples_per_block)
    return symbols


def vec(tensor):
    """张量展平为列向量"""
    return tensor.reshape(-1,1)


def khatri_rao(A,B):
    """Khatri-Rao积，列拼接的Kronecker积，A,B size: MxK, NxK 输出(MN)xK"""
    assert A.shape[1]==B.shape[1]
    return np.vstack([np.kron(A[:,k], B[:,k]) for k in range(A.shape[1])]).T


def unfold(tensor, mode):
    """mode-n展开 张量转矩阵"""
    return np.reshape(np.moveaxis(tensor,mode,0), (tensor.shape[mode], -1))


def parafac_sgd_init(Y, R, max_iter=100):
    """
    简单的ALS初始化 PARAFAC分解, 用以提供初始因子矩阵
    Y: 三阶张量 MxMxL
    R: 秩估计
    返回 A,B,C三个因子矩阵, shape分别为: MxR, MxR, LxR
    """
    M,L = Y.shape[0], Y.shape[2]
    # 初始化随机矩阵
    A = np.random.randn(M, R) + 1j*np.random.randn(M,R)
    B = np.random.randn(M, R) + 1j*np.random.randn(M,R)
    C = np.random.randn(L, R) + 1j*np.random.randn(L,R)

    # 正则化初始矩阵
    def normalize_columns(X):
        norms = LA.norm(X, axis=0)
        return X / norms[np.newaxis,:], norms

    A, _ = normalize_columns(A)
    B, _ = normalize_columns(B)
    C, _ = normalize_columns(C)

    for it in range(max_iter):
        # 更新A
        V = khatri_rao(C, B)  # (L*M)xR
        Y1 = unfold(Y, 0)    # M x (M*L)
        A = np.linalg.lstsq(V.conj().T, Y1.T, rcond=None)[0].T
        A, _ = normalize_columns(A)

        # 更新B
        V = khatri_rao(C, A)  # (L*M)xR
        Y2 = unfold(Y, 1)    # M x (M*L)
        B = np.linalg.lstsq(V.conj().T, Y2.T, rcond=None)[0].T
        B, _ = normalize_columns(B)

        # 更新C
        V = khatri_rao(B, A)  # (M*M)xR
        Y3 = unfold(Y, 2)    # L x (M*M)
        C = np.linalg.lstsq(V.conj().T, Y3.T, rcond=None)[0].T
        C, _ = normalize_columns(C)
    return A,B,C


def estimate_angles_from_steering_vectors(A_hat, B_hat, M, d, wavelength):
    """
    从估计的导向矩阵中提取俯仰角theta和方位角phi
    A_hat shape: M x K (Z轴导向矩阵)
    B_hat shape: M x K (X轴导向矩阵)

    返回 theta_hat, phi_hat shape为(K,)
    """

    K = A_hat.shape[1]
    theta_hat = np.zeros(K)
    phi_hat = np.zeros(K)

    for k in range(K):
        # Z轴导向矢量相位估计
        a_z = A_hat[:,k]
        # 相位差估计，基于第一个和第二个元素
        phase_diff = np.angle(a_z[1]*np.conj(a_z[0]))
        # 对多个阵元取线性拟合相位
        m_idx = np.arange(M)
        phases = np.unwrap(np.angle(a_z))
        # 最小二乘估计斜率代表mu_z
        A_m = np.vstack([m_idx,np.ones(M)]).T
        slope, _ = np.linalg.lstsq(A_m, phases, rcond=None)[0]
        mu_z = -slope  # 因为exp(-j mu_z m)
        # 计算俯仰角
        cos_theta = mu_z * wavelength/(2*np.pi*d)
        cos_theta = np.clip(cos_theta, -1,1)
        theta_hat[k] = np.arccos(cos_theta)

        # X轴导向矢量
        b_x = B_hat[:,k]
        phases_b = np.unwrap(np.angle(b_x))
        slope_b, _ = np.linalg.lstsq(A_m, phases_b, rcond=None)[0]
        mu_x = -slope_b

        # 利用 mu_x = 2pi d / lambda * sin(theta) * cos(phi) 解出phi
        sin_theta = np.sin(theta_hat[k])
        if np.abs(sin_theta) < 1e-6:
            phi_hat[k] = 0.0
        else:
            cos_phi = mu_x * wavelength/(2*np.pi*d)/sin_theta
            cos_phi = np.clip(cos_phi, -1, 1)
            phi_hat[k] = np.arccos(cos_phi)
            # 方向确定问题，无法区分phi和 -phi，这里暂做简化
    return theta_hat, phi_hat


def rmse(true, est):
    """
    均方根误差计算
    true, est维度相同，弧度制，循环角度误差处理
    """
    diff = np.angle(np.exp(1j*(est - true)))  # 处理角度环绕
    return np.sqrt(np.mean(diff**2))


def batch_parafac_als(Y, R, max_iter=50):
    """
    使用ALS批处理PARAFAC作为性能上界
    """
    M,_,L = Y.shape
    A = np.random.randn(M,R) + 1j*np.random.randn(M,R)
    B = np.random.randn(M,R) + 1j*np.random.randn(M,R)
    C = np.random.randn(L,R) + 1j*np.random.randn(L,R)
    def normalize_columns(X):
        norms = LA.norm(X, axis=0)
        return X / norms[np.newaxis,:], norms
    A,_ = normalize_columns(A)
    B,_ = normalize_columns(B)
    C,_ = normalize_columns(C)
    for it in range(max_iter):
        # update A
        V = khatri_rao(C, B)
        Y1 = unfold(Y, 0)
        A = np.linalg.lstsq(V.conj().T, Y1.T, rcond=None)[0].T
        A,_ = normalize_columns(A)
        # update B
        V = khatri_rao(C, A)
        Y2 = unfold(Y, 1)
        B = np.linalg.lstsq(V.conj().T, Y2.T, rcond=None)[0].T
        B,_ = normalize_columns(B)
        # update C
        V = khatri_rao(B, A)
        Y3 = unfold(Y, 2)
        C = np.linalg.lstsq(V.conj().T, Y3.T, rcond=None)[0].T
        C,_ = normalize_columns(C)
    return A, B, C


def past_subspace_tracking(Y, R):
    """
    简易PAST算法示范，用于子空间估计（基准对比）
    Y shape: (M*M, L) 或观测矩阵
    R: 维度
    返回信号子空间估计矩阵
    """
    M_total, L = Y.shape
    Q = np.zeros((M_total, R), dtype=complex)
    P = np.eye(R) * 1e3

    for t in range(L):
        y = Y[:,t].reshape(-1,1)
        w = Q.conj().T @ y
        e = y - Q @ w
        k = P @ w / (1 + w.conj().T @ P @ w)
        Q = Q + e @ k.conj().T
        P = P - k @ w.conj().T @ P
    return Q


def construct_virtual_tensor(z_data, x_data, W, L):
    """
    构建虚拟张量: 三阶张量大小 (M, M, L)
    z_data and x_data shape: (M, T)
    以滑动窗口W和时间延迟L构建互相关矩阵样本
    返回: tensor shape (M, M, L)
    """
    M, T = z_data.shape
    Y = np.zeros((M, M, L), dtype=complex)
    for l in range(L):
        if T - W - l < 0:
            # 不足数据，补0或跳过
            continue
        # 互相关矩阵计算，第 t 处于滑动定时段W内，带延迟l
        R_zx = np.zeros((M,M), dtype=complex)
        for tau in range(T - W - l, T - l):
            R_zx += np.outer(z_data[:,tau + l], x_data[:,tau].conj())
        Y[:,:,l] = R_zx / W
    return Y


def main():
    try:
        # 环境参数配置
        M = 6               # Z轴和X轴阵元数量
        d = 0.5             # 阵元间距(单位波长的倍数, d=lambda/2)
        wavelength = 1.0    # 波长单位归一化，用d=lambda/2所以lambda=1
        K = 15              # 信源数，超分辨(K> 2M -1=11)
        T = 100             # 总时刻数
        W = 10              # 互相关滑动窗口大小
        L = 5               # 延迟维度，构造三阶张量
        
        np.random.seed(2024)

        # 1) 生成信源角度轨迹 theta和phi (俯仰与方位角)
        theta_true, phi_true = generate_source_angles(K, T)  # (K,T) 弧度制

        # 2) 生成QPSK信号，持续采样，假设每时刻采样samples_per_block个
        samples_per_block = 5  # 可调，越大统计稳定性越好
        signals = generate_qpsk_signals(K, T, samples_per_block)  # (K, T*samples_per_block)

        # 3) 初始化阵列导向矩阵与测量数据生成
        # 物理阵元位置不再显式构建
        # 初始化观测数据: z轴和x轴顶点阵列接收数据
        # z_data shape: M x (T*samples_per_block)
        # x_data shape: M x (T*samples_per_block)

        z_data = np.zeros((M, T*samples_per_block), dtype=complex)
        x_data = np.zeros((M, T*samples_per_block), dtype=complex)

        # 添加噪声标准差
        SNR_dB = 10  # 信噪比10dB做演示
        noise_power = 10**(-SNR_dB/10)
        noise_sigma = np.sqrt(noise_power/2)

        # 对每个采样时刻生成测量数据
        for t_idx in range(T*samples_per_block):
            t = t_idx // samples_per_block
            # 构建瞬时导向矩阵A_z和A_x
            A_z = np.hstack([steering_vector(M,d,wavelength,(theta_true[k,t],0), 'z') for k in range(K)])  # M x K
            A_x = np.hstack([steering_vector(M,d,wavelength,(theta_true[k,t],phi_true[k,t]), 'x') for k in range(K)])  # M x K

            s_t = signals[:, t_idx].reshape(-1,1)  # (K,1)

            # 接收信号
            z_sample = A_z @ s_t + noise_sigma*(np.random.randn(M,1) + 1j*np.random.randn(M,1))
            x_sample = A_x @ s_t + noise_sigma*(np.random.randn(M,1) + 1j*np.random.randn(M,1))

            z_data[:,t_idx] = z_sample[:,0]
            x_data[:,t_idx] = x_sample[:,0]

        # 4) 初始化自适应PARAFAC跟踪参数
        R = K   # 假设已知秩

        # 用前一时段构造初始虚拟张量，做批量ALS初始化因子矩阵
        Y_init = construct_virtual_tensor(z_data[:,0:W+L], x_data[:,0:W+L], W, L)  # MxMxL
        Ahat, Bhat, Chat = parafac_sgd_init(Y_init, R, max_iter=100)

        # 初始化RLS相关参数 for A矩阵更新
        lambda_f = 0.96  # 遗忘因子
        MxR = M * R
        # P_A为RLS逆协方差矩阵，大小(R,R)
        P_A = np.eye(R* M, dtype=complex) * 1e3
        P_B = np.eye(R* M, dtype=complex) * 1e3

        # 因子矩阵维度检查
        # Ahat, Bhat: M x R
        # Chat: L x R

        # 存储跟踪结果
        theta_est = np.zeros((K, T), dtype=float)
        phi_est = np.zeros((K, T), dtype=float)

        # 存储RMSE
        rmse_theta = []
        rmse_phi = []

        # ===================
        # 5) 动态跟踪过程
        # ===================
        for t in range(W+L, T):
            # 构造当前时刻虚拟张量
            z_window = z_data[:, t-W-L+1 : t+1]
            x_window = x_data[:, t-W-L+1 : t+1]
            Y_t = construct_virtual_tensor(z_window, x_window, W, L)  # MxMxL

            # 模1展开
            Y_t_1 = unfold(Y_t, 0)  # M x (M*L)

            # vec形式
            y_vec = vec(Y_t)

            # Step 1: 更新 C 矩阵(时间因子)
            V_c = khatri_rao(Bhat.conj(), Ahat)  # (M*M) x R
            try:
                c_new = np.linalg.lstsq(V_c, y_vec, rcond=None)[0].reshape(-1)
            except Exception as e1:
                print(f"倒推c_new时异常: {e1}")
                traceback.print_exc()
                sys.exit(1)
            Chat = np.vstack([Chat[1:,:], c_new[np.newaxis,:]])  # 滚动更新时间因子矩阵 LxR

            # Step 2: 更新 A (Z轴因子矩阵) - RLS递归更新
            H_A = np.kron(Chat[-1,:], Bhat.conj())  # (M*R)x1
            # 计算增益
            denom = lambda_f + (H_A.conj().T @ P_A @ H_A)
            K_A = (P_A @ H_A) / denom

            # 计算预测误差
            pred_A = Ahat @ H_A.conj().T
            e_A = Y_t_1 - pred_A

            # 更新 Ahat
            Ahat = Ahat + e_A @ K_A.conj().T

            # 更新 P_A
            P_A = (P_A - np.outer(K_A, (H_A.conj().T @ P_A))) / lambda_f

            # Step 3: 更新 B (X轴因子矩阵) - 类似方式更新
            H_B = np.kron(Chat[-1,:], Ahat)
            denom_b = lambda_f + (H_B.conj().T @ P_B @ H_B)
            K_B = (P_B @ H_B) / denom_b

            Y_t_2 = unfold(Y_t, 1)  # M x (M*L)
            pred_B = Bhat @ H_B.conj().T
            e_B = Y_t_2 - pred_B
            Bhat = Bhat + e_B @ K_B.conj().T
            P_B = (P_B - np.outer(K_B, (H_B.conj().T @ P_B))) / lambda_f

            # Step 4: 从因子矩阵提取角度
            theta_k, phi_k = estimate_angles_from_steering_vectors(Ahat, Bhat, M, d, wavelength)
            theta_est[:, t] = theta_k
            phi_est[:, t] = phi_k

        # 将真实值仅截取有效估计段
        theta_true_cut = theta_true[:, W+L:T]
        phi_true_cut = phi_true[:, W+L:T]

        theta_est_cut = theta_est[:, W+L:T]
        phi_est_cut = phi_est[:, W+L:T]

        # 6) 计算RMSE指标
        rmse_theta_val = rmse(theta_true_cut.flatten(), theta_est_cut.flatten())
        rmse_phi_val = rmse(phi_true_cut.flatten(), phi_est_cut.flatten())

        # 7) 对比算法性能 - 仅示范批量PARAFAC
        # 取一个时段数据做对比 (取中段）
        Y_batch = construct_virtual_tensor(z_data[:, T//2 - W - L +1 : T//2 +1],
                                          x_data[:, T//2 - W - L +1 : T//2 +1], W, L)
        A_b, B_b, C_b = batch_parafac_als(Y_batch, R, max_iter=30)
        theta_b, phi_b = estimate_angles_from_steering_vectors(A_b, B_b, M, d, wavelength)

        # 计算批量误差（相对于该时刻真实角度）
        theta_true_mid = theta_true[:, T//2]
        phi_true_mid = phi_true[:, T//2]

        rmse_theta_batch = rmse(theta_true_mid, theta_b)
        rmse_phi_batch = rmse(phi_true_mid, phi_b)

        # 8) 绘制RMSE对比柱状图
        labels = ['Adaptive Tensor Tracking', 'Batch PARAFAC (ALS)']
        theta_rmses = [rmse_theta_val, rmse_theta_batch]
        phi_rmses = [rmse_phi_val, rmse_phi_batch]

        # 保存性能指标到json
        metrics = {
            'RMSE_theta_adaptive': float(rmse_theta_val),
            'RMSE_phi_adaptive': float(rmse_phi_val),
            'RMSE_theta_batch': float(rmse_theta_batch),
            'RMSE_phi_batch': float(rmse_phi_batch),
            'Parameters': {
                'M': M,
                'K': K,
                'T': T,
                'W': W,
                'L': L,
                'SNR_dB': SNR_dB,
                'lambda_f': lambda_f
            }
        }
        with open('metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)

        # 绘图 - RMSE柱状图
        fig, ax = plt.subplots(figsize=(8,5))
        width = 0.35
        ind = np.arange(len(labels))

        ax.bar(ind - width/2, theta_rmses, width, label='RMSE Theta (rad)')
        ax.bar(ind + width/2, phi_rmses, width, label='RMSE Phi (rad)')

        ax.set_xticks(ind)
        ax.set_xticklabels(labels)
        ax.set_ylabel('RMSE (radians)')
        ax.set_title('DOA Estimation RMSE Comparison')
        ax.legend()
        plt.tight_layout()
        plt.savefig("figure_rmse_comparison.png")
        plt.close()

        print("实验完成！")

    except Exception as e:
        print(f"程序运行出错: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()