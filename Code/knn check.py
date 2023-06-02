import numpy as np
import cv2
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


#################### Start session ##########################################################
ID1 = '308345891'
ID2 = '211670849'
input_video_name = f'../Outputs/{ID1}_{ID2}_stabilized_video.avi'
output_video_name = f'../Outputs/{ID1}_{ID2}_binary.avi'


cap = cv2.VideoCapture(input_video_name)
params = get_video_parameters(cap)
n_frames = params["frame_count"]
w, h = params["width"], params["height"]
w_down, h_down = w, h

out = cv2.VideoWriter(output_video_name, fourcc=cv2.VideoWriter_fourcc(*'XVID'), fps=params["fps"],
                      frameSize=(w_down, h_down), isColor=False)

fgbg = cv2.createBackgroundSubtractorKNN(history=n_frames, detectShadows=False, dist2Threshold=800)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
learning_rate = -1

for i in range(n_frames):
    success, frame = cap.read()
    if not success:
        break
    # frame = cv2.resize(frame, (w_down, h_down), interpolation=cv2.INTER_AREA)
    fgmask = fgbg.apply(frame)



# Start frame from zero
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
for i in range(n_frames):
    success, frame = cap.read()
    if not success:
        break
    # frame = cv2.resize(frame, (w_down, h_down), interpolation=cv2.INTER_AREA)
    fgmask = fgbg.apply(frame)
    fgmask = cv2.medianBlur(fgmask, ksize=5)
    fgmask = cv2.erode(fgmask, kernel, iterations=4)
    fgmask = cv2.dilate(fgmask, kernel, iterations=2)
    out.write(fgmask)