import numpy as np
import cv2
from sklearn.mixture import GaussianMixture
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

def get_video_parameters(capture: cv2.VideoCapture) -> dict:
    """Get an OpenCV capture object and extract its parameters.
    Args:
        capture: cv2.VideoCapture object.
    Returns:
        parameters: dict. Video parameters extracted from the video.
    """
    fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    return {"fourcc": fourcc, "fps": fps, "height": height, "width": width,
            "frame_count": frame_count}

def fit_gmm(pixel_data, num_components=3):
    gmm = GaussianMixture(n_components=num_components, covariance_type='diag', max_iter=90, warm_start=True)
    gmm.fit(pixel_data.reshape(-1, 1))
    return gmm.means_.flatten(), gmm.covariances_.flatten(), gmm.weights_.flatten()


def calculate_pixel_gmm(video_data, num_components=3, n_jobs=-1):
    # Reshape the video data to (n_pixels, n_frames)
    h, w, n_frames = video_data.shape
    reshaped_data = video_data.reshape(-1, n_frames)

    # Parallelize the GMM fitting across pixels
    results = Parallel(n_jobs=n_jobs)(
        delayed(fit_gmm)(pixel_data, num_components)
        for pixel_data in reshaped_data
    )

    # Unpack the results
    pixel_means, pixel_covariances, pixel_weights = zip(*results)

    # Reshape GMM parameters to pixel-wise shape
    pixel_means = np.array(pixel_means).reshape(h, w, num_components)
    pixel_covariances = np.array(pixel_covariances).reshape(h, w, num_components)
    pixel_weights = np.array(pixel_weights).reshape(h, w, num_components)

    return pixel_means, pixel_covariances, pixel_weights

#################### Start session ##########################################################
ID1 = '308345891'
ID2 = '211670849'
input_video_name = f'../Outputs/{ID1}_{ID2}_stabilized_video.avi'
output_video_name = f'../Outputs/{ID1}_{ID2}_binary.avi'


cap = cv2.VideoCapture(input_video_name)
params = get_video_parameters(cap)
n_frames = params["frame_count"]
out = cv2.VideoWriter(output_video_name, fourcc=cv2.VideoWriter_fourcc(*'XVID'), fps=params["fps"],
                      frameSize=(params["width"]//4, params["height"]//4), isColor=False)



# Randomly select N_frames (samples)
N_samples = 25
frameIds = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=N_samples)
# Store selected frames in an array
h_down, w_down = params["height"]//4, params["width"]//4
frames = np.zeros((h_down, w_down, N_samples))

for index, fid in enumerate(frameIds):
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (w_down, h_down), interpolation=cv2.INTER_AREA)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    frames[:, :, index] = gray

######## Now Let's create GMM to every intensity pixel #############
# Initialize GMM
num_components = 3
T = 0.04
alpha = 0.03 # 30 frames/second
epsilon = 0.0001

pixel_means, pixel_covariances, pixel_weights = calculate_pixel_gmm(frames, num_components=num_components)

#################Start Stauffer and Grimson Algorithm ###############################################
# Start frame from zero
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
# After we have Initialize Gmm we can start Stauffer and Grimson Algorithm
for j in range(n_frames):
    success, frame = cap.read()
    if not success:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (w_down, h_down), interpolation=cv2.INTER_AREA)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray_array = gray[:, :, np.newaxis].repeat(num_components, axis=2)
    ################# This Part we need to decide how I correspond fore ground and background#############
    # Replace negative values
    pixel_covariances = np.abs(pixel_covariances)
    criterion = pixel_weights / np.sqrt(pixel_covariances + epsilon)
    sorted_indices = np.argsort(-criterion, axis=2)
    # Compute the cumulative probabilities
    accumulated_prob = np.cumsum(np.sort(-criterion, axis=2) * -1, axis=2)
    background_indices = np.argmin(accumulated_prob > T, axis=2)
    background_mask = np.zeros_like(accumulated_prob, dtype=bool)
    for i in range(num_components):
        background_mask[:, :, i] = np.where(background_indices >= i, True, False)
    foreground = np.logical_not(np.any(accumulated_prob > T, axis=2)).astype(np.uint8)
    background = np.any(accumulated_prob > T, axis=2).astype(np.uint8)
    mask = foreground * 255
    ######################################################################################################
    # Step 1: calcalute match function
    Match = np.abs(gray_array - pixel_means) < (2.5 * np.sqrt(pixel_covariances))
    mask = mask.astype(np.uint8)
    mask[0:50, :] = 0
    mask[250:, :] = 0
    # Step 2: update weights
    pixel_weights = (1 - alpha) * pixel_weights + alpha * Match
    normalize_factor = np.sum(pixel_weights, axis=2)
    normalize_factor = normalize_factor[:, :, np.newaxis].repeat(num_components, axis=2)
    pixel_weights = pixel_weights / normalize_factor
    # Step 3: Update Gaussian parameters
    pho = alpha / pixel_weights
    pixel_means = (1 - pho) * pixel_means + pho * gray_array
    pixel_covariances = (1 - pho) * pixel_covariances + pho * np.square(gray_array - pixel_means)
    #################### Morphological opreation ######################################################################
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # mask = cv2.erode(mask, kernel, iterations=1)
    # plt.figure(1), plt.imshow(gray, cmap='gray')
    # plt.figure(2), plt.imshow(mask, cmap='gray')
    # plt.show()
    # x = 2
    out.write(mask)


# Clear all capture
cap.release()
out.release()
cv2.destroyAllWindows()


