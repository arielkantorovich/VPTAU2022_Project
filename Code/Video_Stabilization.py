import numpy as np
import cv2
import timeit



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


def movingAverage(curve, radius):
  window_size = 2 * radius + 1
  # Define the filter
  f = np.ones(window_size)/window_size
  # Add padding to the boundaries
  curve_pad = np.lib.pad(curve, (radius, radius), 'edge')
  # Apply convolution
  curve_smoothed = np.convolve(curve_pad, f, mode='same')
  # Remove padding
  curve_smoothed = curve_smoothed[radius:-radius]
  # return smoothed curve
  return curve_smoothed

def smooth(trajectory, n_transform=9, SMOOTHING_RADIUS = 5):
    smoothed_trajectory = np.copy(trajectory)
    # Filter the x, y and angle curves
    for i in range(n_transform):
        smoothed_trajectory[:, i] = movingAverage(trajectory[:, i], radius=SMOOTHING_RADIUS)

    return smoothed_trajectory

def fixBorder(frame: np.ndarray) -> np.ndarray:
    (h, w, channels) = frame.shape
    T = cv2.getRotationMatrix2D((w / 2, h / 2), 0, 1.04)
    frame = cv2.warpAffine(frame, T, (w, h))
    return frame

def Finding_Matching_points(prev_img, curr_img):
    """
    :param prev_img: ndarray (h, w)
    :param curr_img: ndarray (h, w)
    :return: (prev_pts, curr_pts) correspond key points
    """
    # Step1:  let's find interest points
    fast = cv2.FastFeatureDetector_create()
    kp_prev = fast.detect(prev_img, None)
    kp_curr = fast.detect(curr_img, None)
    # Step2: create BRIEF descriptor and describe interest points
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    kp_prev, des_prev = brief.compute(prev_img, kp_prev)
    kp_curr, des_curr = brief.compute(curr_img, kp_curr)
    # Step3: Make feature matching using Brute Force Matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_prev, des_curr)
    # Step 4: Extract corresponding points
    prev_pts = []
    curr_pts = []
    for match in matches:
        prev_idx = match.queryIdx
        curr_idx = match.trainIdx
        prev_pt = kp_prev[prev_idx].pt
        curr_pt = kp_curr[curr_idx].pt
        prev_pts.append(prev_pt)
        curr_pts.append(curr_pt)

    # Convert points to NumPy arrays
    prev_pts = np.array(prev_pts)
    curr_pts = np.array(curr_pts)

    return prev_pts, curr_pts



###############################################################################################################################
# Initialize parameters
# FILL IN YOUR ID
ID1 = '308345891'
ID2 = '211670849'
input_video_name = '../Inputs/INPUT.avi'
output_video_name = f'../Outputs/{ID1}_{ID2}_stabilized_video.avi'

##################### Step 1 : Set Input and Output Videos #######################
# Read input video
cap = cv2.VideoCapture(input_video_name)
# Get frame count
params = get_video_parameters(cap)
n_frames = params["frame_count"]
out = cv2.VideoWriter(output_video_name, fourcc=cv2.VideoWriter_fourcc(*'XVID'), fps=params["fps"],
                      frameSize=(params["width"], params["height"]), isColor=True)

##################### Step 2: Read the first frame and convert it to grayscale #####################
_, prev = cap.read()
h, w, _ = prev.shape
# Convert frame to grayscale
prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

######################## Step 3: Find motion between frames  #####################################
# Pre-define transformation-store array
transforms = np.zeros((n_frames - 1, 9), np.float32)
for i in range(n_frames):
    # Detect feature points in previous frame
    # Read next frame
    success, curr = cap.read()
    if not success:
        break
    # Convert to grayscale
    curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
    prev_pts, curr_pts = Finding_Matching_points(prev_img=prev_gray, curr_img=curr_gray)

    # Find Homography transformation matrix
    H, _ = cv2.findHomography(prev_pts, curr_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)
    transforms[i] = H.flatten()
    prev_gray = curr_gray

###################################### Step 4: Calculate smooth motion between frames ###############################
# Compute trajectory using cumulative sum of transformations
trajectory = np.cumsum(transforms, axis=0)
smoothed_trajectory = smooth(trajectory)
# Calculate difference in smoothed_trajectory and trajectory
difference = smoothed_trajectory - trajectory
# Calculate newer transformation array
transforms_smooth = transforms + difference

#################################### Step 5: Apply smoothed camera motion to frames ########################################

# Reset stream to first frame
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Write n_frames-1 transformed frames
for i in range(n_frames-1):
    # Read next frame
    success, frame = cap.read()
    if not success:
        break

    H_stable = transforms_smooth[i].reshape((3, 3))
    frame_stabilized = cv2.warpPerspective(frame, H_stable, (w, h))

    # Fix border artifacts
    frame_stabilized = fixBorder(frame_stabilized)
    out.write(frame_stabilized)

# Save same size of frames
out.write(frame_stabilized)


# Clear all capture
cap.release()
out.release()
cv2.destroyAllWindows()

