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

def Extract_Large_Object(binary_image):
    _, labels = cv2.connectedComponents(binary_image)
    # Find the largest connected component
    unique_labels, counts = np.unique(labels, return_counts=True)
    largest_component_label = unique_labels[np.argmax(counts[1:]) + 1]
    # Create a mask for the largest component
    largest_component_mask = np.uint8(labels == largest_component_label)
    # Apply the mask to the original image to extract the largest component
    largest_component = cv2.bitwise_and(fgMask, fgMask, mask=largest_component_mask)
    return largest_component

def fill_holes(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    mask = np.zeros_like(largest_component_mask)
    cv2.drawContours(mask, [contours[0]], 0, (255), thickness=cv2.FILLED)
    return mask



#################### Start session ##########################################################
ID1 = '308345891'
ID2 = '211670849'
input_video_name = f'../Outputs/{ID1}_{ID2}_stabilized_video.avi'
output_video_name = f'../Outputs/{ID1}_{ID2}_binary.avi'
output_video_name_2 = f'../Outputs/{ID1}_{ID2}_extracted.avi'


cap = cv2.VideoCapture(input_video_name)
params = get_video_parameters(cap)
n_frames = params["frame_count"]
w, h = params["width"], params["height"]
w_down, h_down = w//4, h//4


out = cv2.VideoWriter(output_video_name, fourcc=cv2.VideoWriter_fourcc(*'XVID'), fps=params["fps"],
                      frameSize=(w, h), isColor=False)
out2 = cv2.VideoWriter(output_video_name_2, fourcc=params["fourcc"], fps=params["fps"],
                      frameSize=(w, h), isColor=True)


# Randomly select N_frames (samples)
N_samples = 1500
Thres_KNN = 310
frameIds = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=N_samples)

KNN_bg = cv2.createBackgroundSubtractorKNN(history=N_samples, detectShadows=True, dist2Threshold=Thres_KNN)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# Initialize background of N_samples
for index, fid in enumerate(frameIds):
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    ret, frame = cap.read()
    # Let's resize to reduce the shake motion
    frame = cv2.blur(frame, (5, 5))
    frame = cv2.resize(frame, (w_down, h_down), interpolation=cv2.INTER_AREA)
    KNN_bg.apply(frame)

# Start frame from zero
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
for i in range(n_frames):
    success, frame = cap.read()
    if not success:
        break
    new_frame = frame.copy()
    # Let's resize to reduce the shake motion
    frame = cv2.blur(frame, (5, 5))
    frame = cv2.resize(frame, (w_down, h_down), interpolation=cv2.INTER_AREA)
    # Pre Processing gaussian blur
    fgMask = KNN_bg.apply(frame)
    # In this part we remove the shadow
    _, fgMask = cv2.threshold(fgMask, 250, 255, cv2.THRESH_BINARY)
    largest_component_mask = Extract_Large_Object(fgMask)
    # Fill contour of the walking person
    mask = fill_holes(largest_component_mask)
    opened_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    opened_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_OPEN, kernel)
    closed_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, kernel)
    final_results = cv2.resize(closed_mask, (w, h), interpolation=cv2.INTER_CUBIC)
    closed_mask = final_results.astype(np.int32)
    new_frame[:, :, 0] = closed_mask*new_frame[:, :, 0] / 255
    new_frame[:, :, 1] = closed_mask*new_frame[:, :, 1] / 255
    new_frame[:, :, 2] = closed_mask*new_frame[:, :, 2] / 255
    out.write(closed_mask.astype(np.uint8))
    out2.write(new_frame.astype(np.uint8))





