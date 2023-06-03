import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from scipy.ndimage import binary_fill_holes as binary_fill_holes

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

def Extract_Large_Object(binary_image):
    """
    :param binary_image: ndarray after KNN background subtraction
    :return: new_binary: ndarray after extract only the largest components
    """
    new_binary = np.zeros_like(binary_image, dtype=np.uint8)
    labeled_image = label(binary_image)
    regions = regionprops(labeled_image)
    regions = sorted(regions, key=lambda x: x.area, reverse=True)
    largest_component = regions[0]
    new_binary[labeled_image == largest_component.label] = 1
    return new_binary * 255


#################### Start session ##########################################################
ID1 = '308345891'
ID2 = '211670849'
input_video_name = f'../Outputs/{ID1}_{ID2}_stabilized_video.avi'
output_video_name = f'../Outputs/{ID1}_{ID2}_binary.avi'


cap = cv2.VideoCapture(input_video_name)
params = get_video_parameters(cap)
n_frames = params["frame_count"]
w, h = params["width"], params["height"]
w_down, h_down = w//2, h//2

out = cv2.VideoWriter(output_video_name, fourcc=cv2.VideoWriter_fourcc(*'XVID'), fps=params["fps"],
                      frameSize=(w, h), isColor=False)

# Randomly select N_frames (samples)
N_samples = 1000
Thres_KNN = 250
frameIds = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=N_samples)

KNN_bg = cv2.createBackgroundSubtractorKNN(history=N_samples, detectShadows=True, dist2Threshold=Thres_KNN)
# KNN_bg = cv2.createBackgroundSubtractorMOG2(history=N_samples, detectShadows=True, varThreshold=5)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# Initialize background of N_samples
for index, fid in enumerate(frameIds):
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    ret, frame = cap.read()
    KNN_bg.apply(frame)

# Start frame from zero
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
for i in range(n_frames):
    success, frame = cap.read()
    if not success:
        break
    fgMask = KNN_bg.apply(frame)
    # In this part we remove the shadow
    _, fgMask = cv2.threshold(fgMask, 250, 255, cv2.THRESH_BINARY)
    # Use morpholgical dilation & Erosion to make more stable binary image
    fgMask = cv2.erode(fgMask, kernel, iterations=1)
    fgMask = cv2.dilate(fgMask, kernel, iterations=2)
    # Extract the walking person
    new_image = Extract_Large_Object(fgMask)
    # Fill contour of the walking person
    contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    mask = np.zeros_like(fgMask)
    cv2.drawContours(mask, [contours[0]], 0, (255), thickness=cv2.FILLED)
    # Use morpholgical closing fellowing open
    opened_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closed_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, kernel)
    out.write(closed_mask)