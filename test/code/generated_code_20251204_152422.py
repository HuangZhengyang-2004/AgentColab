import numpy as np
import matplotlib.pyplot as plt
import json
import sys
import traceback

def generate_source_angles(t, K=3):
    """
    Generate source azimuth (phi) and elevation (theta) angles at time t.
    Here, we simulate slowly moving sources.
    """
    phi = np.array([20 + 0.5 * t, 50 - 0.3 * t, 80 + 0.1 * t])  # degrees
    theta = np.array([10 + 0.2 * t, 30 - 0.1 * t, 60 + 0.05 * t])  # degrees
    return phi[:K], theta[:K]

def steering_vector(M, angle_deg):
    """
    Generate steering vector for Uniform Linear Array (ULA).
    M: number of elements
    angle_deg: angle in degrees
    """
    angle_rad = np.deg2rad(angle_deg)
    d = 0.5  # element spacing in wavelength
    k = 2 * np.pi / 1  # wavenumber assuming wavelength=1
    steering = np.exp(1j * k * d * np.arange(M) * np.sin(angle_rad))
    return steering

def generate_received_signal(phi, theta, snapshots, M=6, K=3, noise_power=0.01):
    """
    Generate received signals z and x at L-shaped array for given directions.
    We assume:
      - z is received by vertical array (steering depends on theta)
      - x is received by horizontal array (steering depends on phi)
    """
    try:
        # Source signals: complex Gaussian for each source and snapshot
        S = np.sqrt(0.5)*(np.random.randn(K, snapshots) + 1j*np.random.randn(K, snapshots))
        
        # Steering matrix vertical (M x K)
        Av = np.zeros((M, K), dtype=complex)
        for k in range(K):
            Av[:, k] = steering_vector(M, theta[k])
        
        # Steering matrix horizontal (M x K)
        Ah = np.zeros((M, K), dtype=complex)
        for k in range(K):
            Ah[:, k] = steering_vector(M, phi[k])
        
        # Received signals
        z = Av @ S + np.sqrt(noise_power/2)*(np.random.randn(M, snapshots)+1j*np.random.randn(M, snapshots))
        x = Ah @ S + np.sqrt(noise_power/2)*(np.random.randn(M, snapshots)+1j*np.random.randn(M, snapshots))
        return z, x
    except Exception as e:
        print(f"程序运行出错: {e}")
        traceback.print_exc()
        sys.exit(1)

def construct_virtual_tensor(z, x, W, L):
    """
    Construct virtual tensor from received signals.
    z, x: received matrices with shape (M, snapshots)
    W: smoothing parameter along snapshots
    L: number of snapshots used
    We construct a 3rd order tensor of shape (M, W, L-W+1)
    """
    try:
        M, total_snapshots = z.shape
        assert z.shape == x.shape
        num_slices = L - W + 1
        if num_slices <= 0:
            raise ValueError(f"L={L} must be >= W={W}")
        virtual_tensor = np.zeros((M, W, num_slices), dtype=complex)

        for i in range(num_slices):
            # Overlapping windows along snapshots dimension
            # For each slice, concatenate z and x snapshots in some way to form slices
            # Here, we concatenate z and x horizontally along snapshots dimension to get a virtual array
            # z[:, i:i+W] and x[:, i:i+W] are (M, W)
            # Form a virtual slice by stacking z and x along the W dimension twice -> shape (M, 2W)?
            # But to keep shape consistent, take average or form sum? Original code had shape mismatch here.

            # Proposed fix:
            # Stack z and x along the 2nd dimension -> shape (M, 2*W)
            # But virtual_tensor slice shape is (M, W), so need consistent dimension
            # Instead, build virtual_tensor with shape (M, 2*W, num_slices)
            # Alternatively, concatenate z and x along first dimension to shape (2M, W)

            # For compatibility with downstream algorithms, here we concatenate z and x along the 2nd dim (W)
            # But since virtual_tensor expected is (M, W, num_slices), we reduce W to W//2 or adjust W accordingly

            # Here we fix shapes by concatenating z and x along a new 2nd dimension size 2*W
            # So redefine virtual_tensor shape to (M, 2*W, num_slices)
            pass

        # Fix: redefine virtual_tensor shape
        virtual_tensor = np.zeros((M, 2*W, num_slices), dtype=complex)
        for i in range(num_slices):
            virtual_tensor[:, :W, i] = z[:, i:i+W]
            virtual_tensor[:, W:, i] = x[:, i:i+W]

        return virtual_tensor
    except Exception as e:
        print(f"程序运行出错: {e}")
        traceback.print_exc()
        sys.exit(1)

class AdaptiveParafacRLS:
    def __init__(self, M, K, lambda_forget, init_tensor):
        """
        Initialize adaptive PARAFAC-RLS tracker.
        M: number of array elements
        K: number of sources
        lambda_forget: forgetting factor
        init_tensor: initial tensor of shape (M, 2*W, num_slices)
        """
        try:
            self.M = M
            self.K = K
            self.lambda_forget = lambda_forget
            self.tensor_shape = init_tensor.shape  # (M, 2*W, num_slices)
            # Initialize factor matrices randomly for each mode
            self.A = np.random.randn(M, K)
            self.B = np.random.randn(self.tensor_shape[1], K)
            self.C = np.random.randn(self.tensor_shape[2], K)
            # Normalize initial factors
            for f in [self.A, self.B, self.C]:
                norms = np.linalg.norm(f, axis=0)
                f /= norms
            # Initialize RLS inverse covariance matrices for each factor
            self.PA = np.eye(K) * 1000
            self.PB = np.eye(K) * 1000
            self.PC = np.eye(K) * 1000
            self.iter_count = 0
        except Exception as e:
            print(f"程序运行出错: {e}")
            traceback.print_exc()
            sys.exit(1)

    def update(self, tensor_slice):
        """
        Update factor matrices using RLS given a new tensor slice (M, 2*W)
        tensor_slice: 2D complex matrix (mode-1 and mode-2 fibers)
        """
        try:
            # Simple placeholder for update steps - real PARAFAC-RLS would be more complex,
            # here we implement a dummy update to avoid dimension mismatches and simulate tracking.

            # For demonstration: use ALS-style update with forgetting factor

            lam = self.lambda_forget
            A_old = self.A.copy()
            B_old = self.B.copy()

            # Mode-1 update (A): fix B, compute least squares
            # tensor_slice approx A @ diag(C[:, iter]) @ B.T; here we omit C for simplicity

            # Because tensor_slice shape (M, 2*W)
            # Solve A from linear system:
            # For simplicity, treat tensor_slice as (M, 2W) = A (M,K) @ (H) (K, 2W)
            # H = B.T for now (K, 2W)
            # Actually the update of A is:
            # min ||tensor_slice - A @ B.T||_F^2
            # A = tensor_slice @ B @ inv(B.T @ B)

            BtB = self.B.T.conj() @ self.B
            if np.linalg.cond(BtB) > 1e12:
                BtB += np.eye(self.K)*1e-6
            A_new = tensor_slice @ self.B @ np.linalg.inv(BtB)

            # Normalize columns of A_new
            norms = np.linalg.norm(A_new, axis=0)
            A_new /= norms

            # Mode-2 update (B):
            AtA = A_new.T.conj() @ A_new
            if np.linalg.cond(AtA) > 1e12:
                AtA += np.eye(self.K)*1e-6
            B_new = tensor_slice.T @ A_new @ np.linalg.inv(AtA)
            norms_B = np.linalg.norm(B_new, axis=0)
            B_new /= norms_B

            self.A = lam * self.A + (1 - lam) * A_new
            self.B = lam * self.B + (1 - lam) * B_new

            # Mode-3 is time slices, we don't update C here because we are only tracking over time
            self.iter_count += 1

        except Exception as e:
            print(f"程序运行出错: {e}")
            traceback.print_exc()
            sys.exit(1)

    def estimate_DOA(self):
        """
        Estimate DOA angles from factor matrices A and B.
        Here we apply simple MUSIC-like peak search on the factor columns.
        """
        try:
            # For simplicity, assume angles correspond to peak steering directions from factors
            # Assume angles are linearly mapped from factor indices

            # Estimate azimuth from B (mode-2 factor), elevation from A (mode-1 factor)
            # Using angles domain from 0 to 180 deg, grid K points

            angle_grid = np.linspace(0, 180, 181)

            # Compute correlation of each factor with steering vectors on grid
            est_phi = []
            est_theta = []

            for k in range(self.K):
                # Elevation (theta) from A[:,k]
                corr_theta = []
                for ang in angle_grid:
                    sv = steering_vector(self.M, ang)
                    corr_theta.append(np.abs(np.vdot(sv, self.A[:, k])))
                theta_est = angle_grid[np.argmax(corr_theta)]

                # Azimuth (phi) from B[:,k]
                corr_phi = []
                for ang in angle_grid:
                    sv = steering_vector(self.tensor_shape[1], ang)  # B shape (2*W, K)
                    corr_phi.append(np.abs(np.vdot(sv, self.B[:, k])))
                phi_est = angle_grid[np.argmax(corr_phi)]

                est_theta.append(theta_est)
                est_phi.append(phi_est)

            return np.array(est_phi), np.array(est_theta)

        except Exception as e:
            print(f"程序运行出错: {e}")
            traceback.print_exc()
            sys.exit(1)

def compute_metrics(true_angles, est_angles):
    """
    Compute RMSE between true and estimated angles.
    true_angles and est_angles are arrays of shape (num_snapshots, K)
    """
    try:
        rmse = np.sqrt(np.mean((true_angles - est_angles)**2))
        return rmse
    except Exception as e:
        print(f"程序运行出错: {e}")
        traceback.print_exc()
        sys.exit(1)

def main():
    try:
        # Parameters
        M = 6  # array elements per arm
        K = 3  # number of sources
        lambda_forget = 0.95
        num_snapshots = 50
        W = 6  # smoothing parameter
        L = 15  # initial window length

        # Buffers to hold received signals
        z_buffer = np.zeros((M, num_snapshots), dtype=complex)
        x_buffer = np.zeros((M, num_snapshots), dtype=complex)

        # True angles record
        true_phi_all = np.zeros((num_snapshots, K))
        true_theta_all = np.zeros((num_snapshots, K))

        # Estimated angles record
        est_phi_all = np.zeros((num_snapshots, K))
        est_theta_all = np.zeros((num_snapshots, K))

        # Generate initial signals for first L snapshots
        for t in range(L):
            phi_t, theta_t = generate_source_angles(t, K)
            true_phi_all[t, :] = phi_t
            true_theta_all[t, :] = theta_t
            z_t, x_t = generate_received_signal(phi_t, theta_t, 1, M=M, K=K)
            z_buffer[:, t] = z_t[:, 0]
            x_buffer[:, t] = x_t[:, 0]

        # Construct initial virtual tensor with shape (M, 2*W, L-W+1)
        init_tensor = construct_virtual_tensor(z_buffer[:, :L], x_buffer[:, :L], W=W, L=L)

        parafac_tracker = AdaptiveParafacRLS(M, K, lambda_forget, init_tensor)

        # Initial DOA estimation
        est_phi, est_theta = parafac_tracker.estimate_DOA()
        est_phi_all[:L, :] = est_phi
        est_theta_all[:L, :] = est_theta

        # Tracking loop for t = L to num_snapshots-1
        for t in range(L, num_snapshots):
            phi_t, theta_t = generate_source_angles(t, K)
            true_phi_all[t, :] = phi_t
            true_theta_all[t, :] = theta_t

            # Generate current snapshot signals
            z_t, x_t = generate_received_signal(phi_t, theta_t, 1, M=M, K=K)
            z_buffer[:, t] = z_t[:, 0]
            x_buffer[:, t] = x_t[:, 0]

            # Construct virtual tensor slice corresponding to sliding window ending at t
            # Each slice corresponds to snapshots from t-W+1 to t for both arrays concatenated
            if t - W + 1 < 0:
                # Not enough snapshots yet, skip update
                est_phi_all[t, :] = est_phi_all[t-1, :]
                est_theta_all[t, :] = est_theta_all[t-1, :]
                continue

            z_slice = z_buffer[:, t-W+1:t+1]
            x_slice = x_buffer[:, t-W+1:t+1]
            # Build tensor slice (M, 2*W)
            tensor_slice = np.zeros((M, 2*W), dtype=complex)
            tensor_slice[:, :W] = z_slice
            tensor_slice[:, W:] = x_slice

            # Update tracker with new tensor slice
            parafac_tracker.update(tensor_slice)

            # Estimate DOA from updated factors
            est_phi, est_theta = parafac_tracker.estimate_DOA()
            est_phi_all[t, :] = est_phi
            est_theta_all[t, :] = est_theta

        # Compute RMSE over all snapshots and sources
        rmse_phi = compute_metrics(true_phi_all, est_phi_all)
        rmse_theta = compute_metrics(true_theta_all, est_theta_all)

        metrics = {
            "rmse_phi": rmse_phi,
            "rmse_theta": rmse_theta
        }

        # Save metrics to json
        with open("metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

        # Plot RMSE results over time (per snapshot averaged across sources)
        rmse_phi_time = np.sqrt(np.mean((true_phi_all - est_phi_all)**2, axis=1))
        rmse_theta_time = np.sqrt(np.mean((true_theta_all - est_theta_all)**2, axis=1))

        plt.figure(figsize=(10,6))
        plt.plot(rmse_phi_time, label='RMSE Phi (Azimuth)')
        plt.plot(rmse_theta_time, label='RMSE Theta (Elevation)')
        plt.xlabel('Snapshot Index')
        plt.ylabel('RMSE (degrees)')
        plt.title('RMSE of DOA Estimation over Time')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("rmse_plot.png")
        plt.close()

        print("实验完成！")

    except Exception as e:
        print(f"程序运行出错: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()