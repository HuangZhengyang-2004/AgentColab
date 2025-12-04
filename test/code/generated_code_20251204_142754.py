import numpy as np
import matplotlib.pyplot as plt
import json
import traceback

def steering_vector_x(theta, phi, M, d, lam):
    """
    Compute the x-axis steering vector for L-shaped array.
    Args:
        theta: azimuth angle (radians)
        phi: elevation angle (radians)
        M: number of x-axis elements minus 1
        d: element spacing
        lam: wavelength
    Returns:
        (M+1,) complex numpy array
    """
    m = np.arange(M+1)
    sv = np.exp(-1j * 2 * np.pi * d / lam * m * np.sin(theta) * np.cos(phi))
    return sv

def steering_vector_z(theta, M, d, lam):
    """
    Compute the z-axis steering vector for L-shaped array.
    Args:
        theta: azimuth angle (radians)
        M: number of z-axis elements minus 1
        d: element spacing
        lam: wavelength
    Returns:
        (M+1,) complex numpy array
    """
    n = np.arange(M+1)
    sv = np.exp(-1j * 2 * np.pi * d / lam * n * np.cos(theta))
    return sv

def build_subarrays(X, P):
    """
    Partition array data into P overlapping subarrays.
    Args:
        X: array data of shape (M+1, )
        P: number of subarrays
    Returns:
        subarrays: shape (M-P+1, P)
    """
    M = X.shape[0] - 1
    sub_len = M - P + 1
    subarrays = np.zeros((sub_len, P), dtype=complex)
    for i in range(P):
        subarrays[:, i] = X[i:i+sub_len]
    return subarrays

def build_cross_cov_tensor(Xx, Xz, P1, P2, alpha, R_xz_prev):
    """
    Build smoothed cross-covariance tensor by spatial smoothing and EWMA.
    Args:
        Xx: x-axis snapshot, shape (M_x+1, )
        Xz: z-axis snapshot, shape (M_z+1, )
        P1: number of subarrays in x-axis
        P2: number of subarrays in z-axis
        alpha: forgetting factor for EWMA
        R_xz_prev: previous cross covariance tensor, shape (m_sub, n_sub, P1*P2)
    Returns:
        R_xz_new: updated cross covariance tensor, shape (m_sub, n_sub, P1*P2)
        Y_tensor: current tensor snapshot (without EWMA), same shape as R_xz_new
    """
    Mx = Xx.shape[0] -1
    Mz = Xz.shape[0] -1
    m_sub = Mx - P1 +1
    n_sub = Mz - P2 +1
    S = P1 * P2

    # Form subarrays
    # For each snapshot vector: form P overlapping subarrays
    # Xx_sub arrays: each subarray is length m_sub
    Xx_sub = np.zeros((m_sub, P1), dtype=complex)
    for i in range(P1):
        Xx_sub[:, i] = Xx[i:i+m_sub]

    Xz_sub = np.zeros((n_sub, P2), dtype=complex)
    for j in range(P2):
        Xz_sub[:, j] = Xz[j:j+n_sub]

    # Construct tensor Y(t) of shape (m_sub, n_sub, S)
    Y = np.zeros((m_sub, n_sub, S), dtype=complex)
    idx = 0
    for i in range(P1):
        for j in range(P2):
            # Outer product: Xx_sub[:,i] * conj(Xz_sub[:,j])^T
            Y[:, :, idx] = np.outer(Xx_sub[:, i], np.conj(Xz_sub[:, j]))
            idx +=1

    # Update EWMA cross covariance tensor
    if R_xz_prev is None:
        R_xz_new = Y
    else:
        R_xz_new = alpha * R_xz_prev + (1-alpha)* Y

    return R_xz_new, Y

def unfold(tensor, mode):
    """
    Unfold a 3D tensor along specified mode
    Args:
        tensor: np.array shape (I,J,K)
        mode: 1,2 or 3
    Returns:
        matrix unfolding of shape depending on mode
    """
    if mode == 1:
        return tensor.reshape(tensor.shape[0], -1)
    elif mode == 2:
        return np.transpose(tensor, (1,0,2)).reshape(tensor.shape[1], -1)
    elif mode == 3:
        return np.transpose(tensor, (2,0,1)).reshape(tensor.shape[2], -1)
    else:
        raise ValueError("Mode must be 1, 2 or 3")

def khatri_rao(A, B):
    """
    Compute Khatri-Rao product of matrices A and B
    Both with same number of columns.
    Args:
        A: (I, K)
        B: (J, K)
    Returns:
        (I*J, K)
    """
    assert A.shape[1] == B.shape[1]
    K = A.shape[1]
    result = np.zeros((A.shape[0]*B.shape[0], K), dtype=A.dtype)
    for k in range(K):
        col = np.kron(A[:, k], B[:, k])
        result[:, k] = col
    return result

def rls_update(Factor_old, Psi_old, Phi_new, lam, delta):
    """
    Recursive Least Squares update for factor matrix in PARAFAC.
    Args:
        Factor_old: old factor matrix (I x R)
        Psi_old: old inverse correlation matrix (R x R)
        Phi_new: new observation matrix (I x R)
        lam: forgetting factor (0<lam<=1)
        delta: initial diagonal loading parameter
    Returns:
        Factor_new, Psi_new
    """
    # This function uses RLS-like recursion to update factor matrix
    # Solve min sum lam^{t-i} || y(i) - Factor * Phi(i) ||^2
    # Here Phi_new is new regressor, Factor is regressed matrix

    I, R = Factor_old.shape
    Phi = Phi_new  # I x R

    # Convert shapes for RLS update:
    # We vectorize factor matrix column-wise as vector of length I*R,
    # but here we do update for each column separately.

    # For each column r:
    Factor_new = np.zeros_like(Factor_old)
    Psi_new = np.zeros_like(Psi_old)
    for r in range(R):
        psi = Psi_old[r:r+1, r:r+1] if Psi_old.size == R else Psi_old
        phi_r = Phi[:, r].reshape(-1,1)  # vector shape (I,1)
        factor_r = Factor_old[:, r].reshape(-1,1)
        # RLS gain vector
        g = psi @ phi_r / (lam + (phi_r.conj().T @ psi @ phi_r))
        # Update inverse covariance
        psi_new = (psi - g @ phi_r.conj().T @ psi) / lam
        # Update factor
        factor_new = factor_r + g * (1 - phi_r.conj().T @ factor_r)
        Factor_new[:, r] = factor_new.ravel()
        if Psi_old.size == R:
            Psi_new[r:r+1, r:r+1] = psi_new
        else:
            Psi_new = psi_new
    return Factor_new, Psi_new

def enforce_vandermonde(A_factor, d, lam, axis='x', order=None):
    """
    Enforce Vandermonde structure on factor matrix by fitting estimated angles,
    then regenerate the factor matrix from angles.
    Args:
        A_factor: factor matrix (M_sub x K), complex
        d: element spacing
        lam: wavelength
        axis: 'x' or 'z' - to select steering vector form
        order: number of elements -1 (M_sub -1)
    Returns:
        A_vand: Vandermonde structured factor matrix (M_sub x K)
        angles_est: estimated angles (radians)
    """
    M_sub, K = A_factor.shape
    if order is None:
        order = M_sub - 1
    indices = np.arange(order+1)

    angles_est = np.zeros(K)
    A_vand = np.zeros_like(A_factor)
    for k in range(K):
        a = A_factor[:, k]

        # Estimate spatial frequency by least squares phase fitting
        phases = np.angle(a)
        # Unwrap phase
        phases_unwrapped = np.unwrap(phases)
        # Least squares fit to linear phase model: phases = -2*pi*d/lambda * indices * freq + C
        # => freq = slope * (-lambda/(2*pi*d))

        A_ls = np.vstack([indices, np.ones(len(indices))]).T
        slope, intercept = np.linalg.lstsq(A_ls, phases_unwrapped, rcond=None)[0]

        freq = - slope * lam / (2 * np.pi * d)

        # Clamp freq to [-1,1]
        freq = np.clip(freq, -1, 1)

        # Compute angle from freq, inverse function depends on axis
        if axis == 'x':
            # freq = sin(theta)*cos(phi)
            # can't separate theta and phi here, just store freq
            angles_est[k] = np.arcsin(np.clip(freq, -1, 1))
        elif axis == 'z':
            # freq = cos(theta)
            freq_adj = np.clip(freq, -1, 1)
            angles_est[k] = np.arccos(freq_adj)
        else:
            angles_est[k] = 0

        # Regenerate Vandermonde vector from freq
        phase_regen = -1j*2*np.pi*d/lam * indices * freq
        A_vand[:, k] = np.exp(phase_regen)

    return A_vand, angles_est

def estimate_DOA(Ax_factor, Az_factor, d, lam):
    """
    Estimate azimuth and elevation angles from Vandermonde structured factors.
    Args:
        Ax_factor: x-axis factor matrix (M_sub x K)
        Az_factor: z-axis factor matrix (N_sub x K)
        d: element spacing
        lam: wavelength
    Returns:
        thetas (azimuth), phis (elevation) in radians, arrays of length K
    """
    _, K = Ax_factor.shape
    # Enforce Vandermonde and get freq_x (sin(theta)*cos(phi))
    Ax_vand, freq_x = enforce_vandermonde(Ax_factor, d, lam, axis='x')
    # Enforce Vandermonde and get freq_z (cos(theta))
    Az_vand, freq_z = enforce_vandermonde(Az_factor, d, lam, axis='z')

    # Recover theta and phi:
    # freq_x = sin(theta)*cos(phi), freq_z = cos(theta)
    thetas = np.arccos(np.clip(freq_z, -1, 1))
    phis = np.arccos(np.clip(freq_x/np.sin(thetas+1e-10), -1, 1))  # avoid div0

    # Handle nan or complex values:
    phis = np.real_if_close(phis)
    thetas = np.real_if_close(thetas)
    return thetas, phis

def main():
    try:
        # PARAMETERS
        np.random.seed(42)
        d = 0.5  # element spacing in wavelength units
        lam = 1.0  # wavelength normalized
        Mx = 8    # x-axis elements minus 1 (total Mx+1)
        Mz = 8    # z-axis elements minus 1 (total Mz+1)
        K = 10    # number of sources (K > 2*Mx)
        P1 = 4    # number of subarrays on x-axis
        P2 = 4    # number of subarrays on z-axis
        alpha = 0.95  # EWMA forgetting factor
        lam_r = 0.98  # RLS forgetting factor
        num_snapshots = 200
        snr_db = 10  # SNR in dB
        noise_var = 10 ** (-snr_db / 10)
        delta = 0.01  # RLS regularization

        # Generate true DOA trajectories (varying smoothly over time)
        t = np.linspace(0, 2*np.pi, num_snapshots)
        thetas_true = (np.pi/6)*np.sin(0.5*t) + np.pi/3  # azimuth angles in [pi/6, pi/2]
        phis_true = (np.pi/8)*np.cos(0.3*t) + np.pi/4   # elevation angles in [pi/8, 3pi/8]
        # For K sources, generate K trajectories by phase shifting
        thetas_all = np.zeros((num_snapshots, K))
        phis_all = np.zeros((num_snapshots, K))
        for k in range(K):
            thetas_all[:, k] = (np.pi/6)*np.sin(0.5*t + 2*np.pi*k/K) + np.pi/3
            phis_all[:, k] = (np.pi/8)*np.cos(0.3*t + 2*np.pi*k/K) + np.pi/4

        # Initial factor matrices (random complex) for PARAFAC tracking
        m_sub = Mx - P1 +1
        n_sub = Mz - P2 +1
        S = P1 * P2

        # Initialize factors A, B, C randomly (A: x-axis, B: z-axis, C: diversity)
        A_factor = (np.random.randn(m_sub, K) + 1j*np.random.randn(m_sub, K)) / np.sqrt(2)
        B_factor = (np.random.randn(n_sub, K) + 1j*np.random.randn(n_sub, K)) / np.sqrt(2)
        C_factor = (np.random.randn(S, K) + 1j*np.random.randn(S, K)) / np.sqrt(2)

        # Initialize RLS covariance inverses for each factor
        Psi_A = np.eye(K) / delta
        Psi_B = np.eye(K) / delta
        Psi_C = np.eye(K) / delta

        # Initialize cross covariance tensor
        R_xz_prev = None

        # For metrics collection
        mse_theta_list = []
        mse_phi_list = []

        # Arrays for storing estimates
        est_thetas_all = np.zeros((num_snapshots, K))
        est_phis_all = np.zeros((num_snapshots, K))

        # Noise covariance
        noise_cov = noise_var * np.eye(Mx+1)
        
        for snap in range(num_snapshots):
            # Generate signals s(t): uncorrelated complex Gaussian source signals
            s = (np.random.randn(K) + 1j*np.random.randn(K)) / np.sqrt(2)

            # Build steering matrices A_x and A_z for time t
            Ax = np.zeros((Mx+1, K), dtype=complex)
            Az = np.zeros((Mz+1, K), dtype=complex)
            for k in range(K):
                Ax[:, k] = steering_vector_x(thetas_all[snap,k], phis_all[snap,k], Mx, d, lam)
                Az[:, k] = steering_vector_z(thetas_all[snap,k], Mz, d, lam)

            # Received signals on x and z arrays with noise
            noise_x = (np.random.randn(Mx+1) + 1j*np.random.randn(Mx+1))/np.sqrt(2)*np.sqrt(noise_var)
            noise_z = (np.random.randn(Mz+1) + 1j*np.random.randn(Mz+1))/np.sqrt(2)*np.sqrt(noise_var)
            x_snap = Ax @ s + noise_x
            z_snap = Az @ s + noise_z

            # Build/Update cross covariance tensor R_xz
            R_xz_prev, Y_tensor = build_cross_cov_tensor(x_snap, z_snap, P1, P2, alpha, R_xz_prev)

            # PARAFAC RLS tracking update for each factor matrix
            # Unfold tensor in modes
            Y1 = unfold(R_xz_prev, 1)  # shape (m_sub, n_sub * S)
            Y2 = unfold(R_xz_prev, 2)  # shape (n_sub, m_sub * S)
            Y3 = unfold(R_xz_prev, 3)  # shape (S, m_sub * n_sub)

            # Compute Khatri-Rao products needed
            # Mode-1 update: Y1 ~= A * (C ⊙ B)^T
            Z1 = khatri_rao(C_factor, B_factor)  # shape (n_sub * S, K)
            # Solve A update via RLS
            # Using approximate RLS: A = Y1 * pinv(Z1.T)
            Z1_herm = Z1.conj().T
            # Update A_factor using RLS like procedure
            # Use simple LS as approximation here for speed (full RLS with matrix sizes is heavy)
            A_new = Y1 @ np.linalg.pinv(Z1_herm)
            A_factor = A_new

            # Mode-2 update: Y2 ~= B * (C ⊙ A)^T
            Z2 = khatri_rao(C_factor, A_factor)  # shape (m_sub * S, K)
            Z2_herm = Z2.conj().T
            B_new = Y2 @ np.linalg.pinv(Z2_herm)
            B_factor = B_new

            # Mode-3 update: Y3 ~= C * (B ⊙ A)^T
            Z3 = khatri_rao(B_factor, A_factor)  # shape (m_sub * n_sub, K)
            Z3_herm = Z3.conj().T
            C_new = Y3 @ np.linalg.pinv(Z3_herm)
            C_factor = C_new

            # Enforce Vandermonde structure for A_factor and B_factor
            A_factor, theta_est_x = enforce_vandermonde(A_factor, d, lam, axis='x')
            B_factor, theta_est_z = enforce_vandermonde(B_factor, d, lam, axis='z')

            # Estimate DOA angles
            thetas_est, phis_est = estimate_DOA(A_factor, B_factor, d, lam)

            # Save estimates
            est_thetas_all[snap, :] = thetas_est.real
            est_phis_all[snap, :] = phis_est.real
            # Reference true angles
            true_thetas = thetas_all[snap, :]
            true_phis = phis_all[snap, :]

            # Compute MSE metric (mean squared error in radians)
            mse_theta = np.mean((thetas_est.real - true_thetas)**2)
            mse_phi = np.mean((phis_est.real - true_phis)**2)
            mse_theta_list.append(mse_theta)
            mse_phi_list.append(mse_phi)

        # Prepare metrics dict
        metrics = {
            "MSE_theta": mse_theta_list,
            "MSE_phi": mse_phi_list,
            "avg_MSE_theta": float(np.mean(mse_theta_list)),
            "avg_MSE_phi": float(np.mean(mse_phi_list)),
        }

        # Save metrics to JSON
        with open('metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)

        # Plot metrics over snapshots
        plt.figure(figsize=(10, 6))
        plt.plot(mse_theta_list, label='MSE Azimuth (theta)')
        plt.plot(mse_phi_list, label='MSE Elevation (phi)')
        plt.xlabel('Snapshot index')
        plt.ylabel('Mean Squared Error (rad^2)')
        plt.title('2D-DOA Estimation MSE over Time')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('figure_MSE_vs_Snapshot.png')
        plt.close()

        # Plot true vs estimated DOA for a few sources
        plt.figure(figsize=(12, 8))
        num_plot = min(5, K)
        for k in range(num_plot):
            plt.subplot(num_plot, 2, 2*k+1)
            plt.plot(t, thetas_all[:, k], label='True Theta k=%d' % k)
            plt.plot(t, est_thetas_all[:, k], '--', label='Est Theta k=%d' % k)
            plt.xlabel('Time')
            plt.ylabel('Azimuth (rad)')
            plt.legend()
            plt.grid(True)

            plt.subplot(num_plot, 2, 2*k+2)
            plt.plot(t, phis_all[:, k], label='True Phi k=%d' % k)
            plt.plot(t, est_phis_all[:, k], '--', label='Est Phi k=%d' % k)
            plt.xlabel('Time')
            plt.ylabel('Elevation (rad)')
            plt.legend()
            plt.grid(True)

        plt.suptitle('True and Estimated 2D-DOA for First %d Sources' % num_plot)
        plt.tight_layout(rect=[0,0,1,0.96])
        plt.savefig('figure_DOA_vs_Time.png')
        plt.close()

        print("实验完成！")

    except Exception as e:
        print("发生错误：", e)
        traceback.print_exc()

if __name__ == '__main__':
    main()