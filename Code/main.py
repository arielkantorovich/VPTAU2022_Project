def my_lucas_kanade_step(I1: np.ndarray,
                      I2: np.ndarray,
                      window_size: int) -> tuple[np.ndarray, np.ndarray]:
    # Initialize step
    h, w, c = I1.shape
    I1 = I1.astype(np.float32)
    I2 = I2.astype(np.float32)
    du = np.zeros((h, w))
    dv = np.zeros((h, w))
    epsilon = 1e-4
    # First Let's calculate Ix, Iy, It u
    I2_shift_x = np.zeros_like(I2)
    I2_shift_y = np.zeros_like(I2)
    I2_shift_x[:, 1:, :] = I2[:, 0:-1, :]
    I2_shift_y[1:, :, :] = I2[0:-1, :, :]
    Ix = I2_shift_x - I2
    Iy = I2_shift_y - I2
    It = I2 - I1
    # Prepare the data for vectorization use that the channel is RGB
    Ixx = np.sum(Ix * Ix, axis=2)
    Iyy = np.sum(Iy * Iy, axis=2)
    Ixy = np.sum(Ix * Iy, axis=2)
    Ixt = np.sum(-Ix * It, axis=2)
    Iyt = np.sum(-Iy * It, axis=2)
    # Step 3:
    # Compute du and dv without for-loops
    ATA = np.stack((Ixx, Ixy, Ixy, Iyy), axis=1).reshape(h, w, 2, 2)
    ATb = np.stack((Ixt, Iyt), axis=1).reshape(h, w, 2, 1)
    det_ATA = ATA[:, :, 0, 0] * ATA[:, :, 1, 1] - ATA[:, :, 0, 1] * ATA[:, :, 1, 0]
    valid_idx = det_ATA > epsilon
    U_V_LS = np.zeros((h, w, 2, 1))
    U_V_LS[valid_idx] = np.linalg.inv(ATA[valid_idx]) @ ATb[valid_idx]
    du, dv = U_V_LS[:, :, 0, 0], U_V_LS[:, :, 1, 0]
    return du, dv