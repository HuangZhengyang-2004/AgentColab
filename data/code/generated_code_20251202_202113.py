```python
import numpy as np
import json
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

try:
    # ==================== 参数设置 ====================
    # 物理阵列参数
    M = 8  # x轴ULA阵元数
    N = 8  # y轴ULA阵元数
    d = 0.5  # 阵元间距（波长归一化）
    
    # 信源参数
    K = 3  # 信源数
    T = 100  # 总时间块数
    L = 50  # 每个时间块的快拍数
    
    # 虚拟子阵参数
    M_s = 4  # x轴子阵大小
    N_s = 4  # y轴子阵大小
    P = M - M_s + 1  # x轴虚拟维数
    Q = N - N_s + 1  # y轴虚拟维数
    
    # 算法参数
    lambda_forget = 0.98  # 遗忘因子
    delta = 0.01  # 正则化参数
    num_iter = 3  # 每时间块迭代次数
    
    # 信源动态参数
    # 初始角度（度）
    theta_init = np.array([30, 45, 60])  # 俯仰角
    phi_init = np.array([20, 40, 60])    # 方位角
    # 角度变化率（度/时间块）
    theta_rate = np.array([0.1, -0.15, 0.05])
    phi_rate = np.array([0.2, -0.1, 0.15])
    
    # 信噪比
    SNR = 20  # dB
    
    # ==================== 辅助函数 ====================
    def steering_vector_ula(n, spacing, angle_param, axis='x'):
        """
        生成ULA导向矢量
        n: 阵元数
        spacing: 阵元间距（波长归一化）
        angle_param: μ或ν参数
        axis: 'x'或'y'，决定相位符号
        """
        m = np.arange(n)
        if axis == 'x':
            return np.exp(-1j * np.pi * spacing * angle_param * m)
        else:  # 'y'
            return np.exp(-1j * np.pi * spacing * angle_param * m)
    
    def angles_to_params(theta, phi):
        """
        将角度(度)转换为μ和ν参数
        theta: 俯仰角（度）
        phi: 方位角（度）
        """
        theta_rad = np.deg2rad(theta)
        phi_rad = np.deg2rad(phi)
        mu = 2 * d * np.sin(theta_rad) * np.cos(phi_rad)
        nu = 2 * d * np.sin(theta_rad) * np.sin(phi_rad)
        return mu, nu
    
    def params_to_angles(mu, nu):
        """
        将μ和ν参数转换为角度(度)
        """
        # 计算方位角
        phi_rad = np.arctan2(nu, mu)
        # 计算俯仰角
        temp = np.sqrt(mu**2 + nu**2) / (2 * d)
        # 防止数值误差导致|temp|>1
        temp = np.clip(temp, -1, 1)
        theta_rad = np.arcsin(temp)
        return np.rad2deg(theta_rad), np.rad2deg(phi_rad)
    
    def khatri_rao(A, B):
        """
        Khatri-Rao积
        """
        n_cols = A.shape[1]
        C = np.zeros((A.shape[0]*B.shape[0], n_cols), dtype=complex)
        for i in range(n_cols):
            C[:, i] = np.kron(A[:, i], B[:, i])
        return C
    
    def parafac_als(tensor, rank, max_iter=50, tol=1e-6):
        """
        批处理PARAFAC-ALS分解（用于初始化）
        tensor: 五阶张量 (P, Q, M, N, L)
        rank: 秩（信源数）
        """
        # 获取维度
        dims = tensor.shape
        P, Q, M, N, L = dims
        
        # 随机初始化因子矩阵
        B1 = np.random.randn(P, rank) + 1j * np.random.randn(P, rank)
        B2 = np.random.randn(Q, rank) + 1j * np.random.randn(Q, rank)
        B3 = np.random.randn(M, rank) + 1j * np.random.randn(M, rank)
        B4 = np.random.randn(N, rank) + 1j * np.random.randn(N, rank)
        B5 = np.random.randn(L, rank) + 1j * np.random.randn(L, rank)
        
        # 归一化
        for i in range(rank):
            B1[:, i] = B1[:, i] / linalg.norm(B1[:, i])
            B2[:, i] = B2[:, i] / linalg.norm(B2[:, i])
            B3[:, i] = B3[:, i] / linalg.norm(B3[:, i])
            B4[:, i] = B4[:, i] / linalg.norm(B4[:, i])
            B5[:, i] = B5[:, i] / linalg.norm(B5[:, i])
        
        # ALS迭代
        for it in range(max_iter):
            # 模式1更新
            Z = khatri_rao(B5, khatri_rao(B4, khatri_rao(B3, B2)))
            R1 = tensor.reshape(P, -1)
            B1 = R1 @ linalg.pinv(Z.T)
            
            # 模式2更新
            Z = khatri_rao(B5, khatri_rao(B4, khatri_rao(B3, B1)))
            R2 = tensor.transpose(1, 0, 2, 3, 4).reshape(Q, -1)
            B2 = R2 @ linalg.pinv(Z.T)
            
            # 模式3更新
            Z = khatri_rao(B5, khatri_rao(B4, khatri_rao(B2, B1)))
            R3 = tensor.transpose(2, 0, 1, 3, 4).reshape(M, -1)
            B3 = R3 @ linalg.pinv(Z.T)
            
            # 模式4更新
            Z = khatri_rao(B5, khatri_rao(B3, khatri_rao(B2, B1)))
            R4 = tensor.transpose(3, 0, 1, 2, 4).reshape(N, -1)
            B4 = R4 @ linalg.pinv(Z.T)
            
            # 模式5更新
            Z = khatri_rao(B4, khatri_rao(B3, khatri_rao(B2, B1)))
            R5 = tensor.transpose(4, 0, 1, 2, 3).reshape(L, -1)
            B5 = R5 @ linalg.pinv(Z.T)
            
            # 检查收敛
            if it > 0:
                diff = linalg.norm(B1 - B1_old) / linalg.norm(B1_old)
                if diff < tol:
                    break
            
            B1_old = B1.copy()
        
        return B1, B2, B3, B4, B5
    
    def estimate_params_from_factors(B1, B2, B3, B4):
        """
        从因子矩阵估计μ和ν参数
        """
        rank = B1.shape[1]
        mu_est = np.zeros(rank)
        nu_est = np.zeros(rank)
        
        for k in range(rank):
            # 从B1估计μ
            b1 = B1[:, k]
            phases1 = np.angle(b1[1:] / b1[:-1])
            mu_from_b1 = -np.mean(phases1) / (np.pi * d)
            
            # 从B3估计μ
            b3 = B3[:, k]
            phases3 = np.angle(b3[1:] / b3[:-1])
            mu_from_b3 = -np.mean(phases3) / (np.pi * d)
            
            mu_est[k] = 0.5 * (mu_from_b1 + mu_from_b3)
            
            # 从B2估计ν
            b2 = B2[:, k]
            phases2 = np.angle(b2[1:] / b2[:-1])
            nu_from_b2 = -np.mean(phases2) / (np.pi * d)
            
            # 从B4估计ν
            b4 = B4[:, k]
            phases4 = np.angle(b4[1:] / b4[:-1])
            nu_from_b4 = -np.mean(phases4) / (np.pi * d)
            
            nu_est[k] = 0.5 * (nu_from_b2 + nu_from_b4)
        
        return mu_est, nu_est
    
    # ==================== 主程序 ====================
    
    # 存储真实角度和估计角度
    true_theta = np.zeros((T, K))
    true_phi = np.zeros((T, K))
    est_theta = np.zeros((T, K))
    est_phi = np.zeros((T, K))
    
    # 存储因子矩阵用于自适应跟踪
    B1_list = []
    B2_list = []
    B3_list = []
    B4_list = []
    B5_list = []
    
    # 存储逆相关矩阵
    P1 = delta**-1 * np.eye(K)
    P2 = delta**-1 * np.eye(K)
    P3 = delta**-1 * np.eye(K)
    P4 = delta**-1 * np.eye(K)
    P5 = delta**-1 * np.eye(K)
    
    # 主循环：处理每个时间块
    for t in range(T):
        # ========== 1. 生成当前时间块的真实角度 ==========
        current_theta = theta_init + t * theta_rate
        current_phi = phi_init + t * phi_rate
        
        true_theta[t, :] = current_theta
        true_phi[t, :] = current_phi
        
        # 转换为μ和ν参数
        mu_true, nu_true = angles_to_params(current_theta, current_phi)
        
        # ========== 2. 生成接收信号 ==========
        # 生成信源信号（不相关）
        S = (np.random.randn(K, L) + 1j * np.random.randn(K, L)) / np.sqrt(2)
        
        # 生成x轴ULA接收信号
        X = np.zeros((M, L), dtype=complex)
        for k in range(K):
            a_x = steering_vector_ula(M, d, mu_true[k], axis='x')
            X += np.outer(a_x, S[k, :])
        
        # 生成y轴ULA接收信号
        Y = np.zeros((N, L), dtype=complex)
        for k in range(K):
            a_y = steering_vector_ula(N, d, nu_true[k], axis='y')
            Y += np.outer(a_y, S[k, :])
        
        # 添加噪声
        noise_power = 10**(-SNR/10)
        X += np.sqrt(noise_power/2) * (np.random.randn(M, L) + 1j * np.random.randn(M, L))
        Y += np.sqrt(noise_power/2) * (np.random.randn(N, L) + 1j * np.random.randn(N, L))
        
        # ========== 3. 构建三阶张量 ==========
        # x轴三阶张量 (M, P, L)
        X_tensor = np.zeros((M, P, L), dtype=complex)
        for i in range(M):
            for j in range(P):
                for l in range(L):
                    idx = min(i + j, M-1)
                    X_tensor[i, j, l] = X[idx, l]
        
        # y轴三阶张量 (N, Q, L)
        Y_tensor = np.zeros((N, Q, L), dtype=complex)
        for i in range(N):
            for j in range(Q):
                for l in range(L):
                    idx = min(i + j, N-1)
                    Y_tensor[i, j, l] = Y[idx, l]
        
        # ========== 4. 构建五阶互相关张量 ==========
        R_tensor = np.zeros((P, Q, M, N, L), dtype=complex)
        for p in range(P):
            for q in range(Q):
                for m in range(M):
                    for n in range(N):
                        for l in range(L):
                            # 样本平均代替统计期望
                            R_tensor[p, q, m, n, l] = np.mean(X_tensor[m, p, :] * np.conj(Y_tensor[n, q, :]))
        
        # ========== 5. 自适应PARAFAC跟踪 ==========
        if t == 0:
            # 第一个时间块：使用批处理PARAFAC-ALS初始化
            B1, B2, B3, B4, B5 = parafac_als(R_tensor, K, max_iter=30)
        else:
            # 后续时间块：自适应跟踪
            # 使用上一时间块的因子矩阵作为初始值
            B1 = B1_list[-1].copy()
            B2 = B2_list[-1].copy()
            B3 = B3_list[-1].copy()
            B4 = B4_list[-1].copy()
            B5 = B5_list[-1].copy()
            
            # 多次迭代改善收敛性
            for it in range(num_iter):
                # 模式1更新
                Z1 = khatri_rao(B5, khatri_rao(B4, khatri_rao(B3, B2)))
                R1 = R_tensor.reshape(P, -1)
                E1 = R1 - B1 @ Z1.T
                K1 = P1 @ Z1 @ linalg.inv(lambda_forget * np.eye(Z1.shape[1]) + Z1.T.conj() @ P1 @ Z1)
                B1 = B1 + K1 @ E1.T.conj()
                P1 = (1/lambda_forget) * (P1 - K1 @ Z1.T.conj() @ P1)
                
                # 模式2更新
                Z2 = khatri_rao(B5, khatri_rao(B4, khatri_rao(B3, B1)))
                R2 = R_tensor.transpose(1, 0, 2, 3, 4).reshape(Q, -1)
                E2 = R2 - B2 @ Z2.T
                K2 = P2 @ Z2 @ linalg.inv(lambda_forget * np.eye(Z2.shape[1]) + Z2.T.conj() @ P2 @ Z2)
                B2 = B2 + K2 @ E2.T.conj()
                P2 = (1/lambda_forget) * (P2 - K2 @ Z2.T.conj() @ P2)
                
                # 模式3更新
                Z3 = khatri_rao(B5, khatri_rao(B4, khatri_rao(B2, B1)))
                R3 = R_tensor.transpose(2, 0, 1, 3, 4).reshape(M, -1)
                E3 = R3 - B3 @ Z3.T
                K3 = P3 @ Z3 @ linalg.inv(lambda_forget * np.eye(Z3.shape[1]) + Z3.T.conj() @ P3 @ Z3)
                B3 = B3 + K3 @ E3.T.conj()
                P3 = (1/lambda_forget) * (P3 - K3 @ Z3.T.conj() @ P3)
                
                # 模式4更新
                Z4 = khatri_rao(B5, khatri_rao(B3, khatri_rao(B2, B1)))
                R4 = R_tensor.transpose(3, 0, 1, 2, 4).reshape(N, -1)
                E4 = R4 - B4 @ Z4.T
                K4 = P4 @ Z4 @ linalg.inv(lambda_forget * np.eye(Z4.shape[1]) + Z4.T.conj() @ P4 @ Z4)
                B4 = B4 + K4 @ E4.T.conj()
                P4 = (1/lambda_forget) * (P4 - K4 @ Z4.T.conj() @ P4)
                
                # 模式5更新
                Z5 = khatri_rao(B4, khatri_rao(B3, khatri_rao(B2, B1)))
                R5 = R_tensor.transpose(4, 0, 1, 2, 3).reshape(L, -1)
                E5 = R5 - B5 @ Z5.T
                K5 = P5 @ Z5 @ linalg.inv(lambda_forget * np.eye(Z5.shape[1]) + Z5.T.conj() @ P5 @ Z5)
                B5 = B5 + K5 @ E5.T.conj()
                P5 = (1/lambda_forget) * (P5 - K5 @ Z5.T.conj() @ P5)
        
        # 保存因子矩阵
        B1_list.append(B1.copy())
        B2_list.append(B2.copy())
        B3_list.append(B3.copy())
        B4_list.append(B4.copy())
        B5_list.append(B5.copy())
        
        # ==========