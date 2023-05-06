# Import numpy and OpenCV
import numpy as np
import cv2

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

###############################################################################################################################
# Initialize parameters
# FILL IN YOUR ID
ID1 = '308345891'
ID2 = '211670849'
input_video_name = 'INPUT.avi'
output_video_name = f'{ID1}_{ID2}_stabilized_video.avi'

##################### Step 1 : Set Input and Output Videos #######################
# Read input video
cap = cv2.VideoCapture(input_video_name)
# Get frame count
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
params = get_video_parameters(cap)
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

lk_params = dict(winSize=(15, 15),
                  maxLevel=5,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


for i in range(n_frames):
    # Detect feature points in previous frame
    prev_pts = cv2.goodFeaturesToTrack(prev_gray,
                                       maxCorners=100,
                                       qualityLevel=0.3,
                                       minDistance=7,
                                       blockSize=7)
    # Read next frame
    success, curr = cap.read()
    if not success:
        break
    # Convert to grayscale
    curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
    # Calculate optical flow (i.e. track feature points)
    curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None, **lk_params)
    # Sanity check
    assert prev_pts.shape == curr_pts.shape
    # Filter only valid points
    idx = np.where(status == 1)[0]
    prev_pts = prev_pts[idx]
    curr_pts = curr_pts[idx]
    # Find transformation matrix
    H, _ = cv2.findHomography(prev_pts, curr_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)
    transforms[i] = H.flatten()
    prev_gray = curr_gray
    print("Frame: " + str(i) + "/" + str(n_frames) + " -  Tracked points : " + str(len(prev_pts)))


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