import cv2
import time
import json
from collections import OrderedDict
from lucas_kanade import lucas_kanade_video_stabilization, get_video_parameters


# FILL IN YOUR ID
ID1 = '308345891'
ID2 = '211670849'

# Choose parameters
WINDOW_SIZE_TAU = 5  # Add your value here!
MAX_ITER_TAU = 5  # Add your value here!
NUM_LEVELS_TAU = 5  # Add your value here!


# Output dir and statistics file preparations:
STATISTICS_PATH = f'TAU_VIDEO_{ID1}_{ID2}_mse_and_time_stats.json'
statistics = OrderedDict()


def calc_mean_mse_video(path: str) -> float:
    """Calculate the mean MSE across all frames.

    The mean MSE is computed between every two consecutive frames in the video.

    Args:
        path: str. Path to the video.

    Returns:
        mean_mse: float. The mean MSE.
    """
    input_cap = cv2.VideoCapture(path)
    video_info = get_video_parameters(input_cap)
    frame_amount = video_info['frame_count']
    input_cap.grab()
    # extract first frame
    prev_frame = input_cap.retrieve()[1]
    # convert to greyscale
    prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    mse = 0.0
    for i in range(1, frame_amount):
        input_cap.grab()
        frame = input_cap.retrieve()[1]  # grab next frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mse += ((frame - prev_frame) ** 2).mean()
        prev_frame = frame
    mean_mse = mse / (frame_amount - 1)
    return mean_mse


# Load video file
input_video_name = 'input2.avi'

output_video_name = f'{ID1}_{ID2}_stabilized_video.avi'
start_time = time.time()
lucas_kanade_video_stabilization(input_video_name,
                                 output_video_name,
                                 WINDOW_SIZE_TAU,
                                 MAX_ITER_TAU,
                                 NUM_LEVELS_TAU)
end_time = time.time()
print(f'LK-Video Stabilization Taking all pixels into account took: '
      f'{end_time - start_time:.2f}[sec]')
statistics["[TAU, TIME] naive LK implementation"] = end_time - start_time



print("The Following MSE values should make sense to you:")
original_mse = calc_mean_mse_video(input_video_name)
print(f"Mean MSE between frames for original video: {original_mse:.2f}")
naive_mse = calc_mean_mse_video(output_video_name)
print(f"Mean MSE between frames for Lucas Kanade Stabilized output video: "
      f"{naive_mse:.2f}")
