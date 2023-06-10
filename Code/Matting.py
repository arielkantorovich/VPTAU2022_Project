import numpy as np
import cv2
import timeit


def extract_color(i, j, alpha_frame, curr_stab, w, h):
    most_prob_color = [0, 0, 0]
    prob = 0
    for k in range(-2,3):
        for l in range(-2,3):
            if i+k < 0 or i+k >= w:
                continue
            if j+l < 0 or j+l >= h:
                continue
            if alpha_frame[j+k][i+l][0] > prob:
                most_prob_color = curr_stab[j+k, i+l, :]
    return most_prob_color



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

start = timeit.default_timer()


###############################################################################################################################
# Initialize parameters
# FILL IN YOUR ID
ID1 = '308345891'
ID2 = '211670849'
input_video_name = f'../Outputs/{ID1}_{ID2}_binary.avi'
input_video_name_2 = f'../Outputs/{ID1}_{ID2}_extracted.avi'
input_video_name_3 = f'../Outputs/{ID1}_{ID2}_stabilized_video.avi'
input_picture = f'../Outputs/backround.jpg'
output_video_name = f'../Outputs/{ID1}_{ID2}_matted.avi'
output_video_name_2 = f'../Outputs/{ID1}_{ID2}_alpha.avi'


##################### Set Input and Output Videos #######################
# Read input video
cap_binary = cv2.VideoCapture(input_video_name)
cap_extracted = cv2.VideoCapture(input_video_name_2)
cap_stab = cv2.VideoCapture(input_video_name_3)
n_frames = int(cap_binary.get(cv2.CAP_PROP_FRAME_COUNT))
params = get_video_parameters(cap_binary)

out_matted = cv2.VideoWriter(output_video_name, fourcc=cv2.VideoWriter_fourcc(*'XVID'), fps=params["fps"],
                      frameSize=(params["width"], params["height"]), isColor=True)
out2_alpha = cv2.VideoWriter(output_video_name_2, fourcc=cv2.VideoWriter_fourcc(*'XVID'), fps=params["fps"],
                      frameSize=(params["width"], params["height"]), isColor=False)

backround = cv2.imread('../Inputs/background.jpg')
backround = cv2.resize(backround, (params["width"], params["height"]))

for i in range(n_frames):
    #print(f"frame{i}/{n_frames}")
    # Read next frame
    success, curr_binary = cap_binary.read()
    if not success:
        break
    success, curr_extracted = cap_extracted.read()
    if not success:
        break
    success, curr_stab = cap_stab.read()
    if not success:
        break


    alpha_frame = curr_binary.copy()
    alpha_frame = alpha_frame.astype(np.float32)
    backround = backround.astype(np.int32)
    curr_extracted = curr_extracted.astype(np.int32)
    curr_extracted_orig = curr_extracted.copy()

    ##################### Create alpha map #######################
    alpha_frame_orig = alpha_frame.copy()
    kernel = np.ones((5, 5), dtype=np.float32)/25.0
    alpha_frame = cv2.filter2D(alpha_frame_orig, -1, kernel)
    alpha_frame = np.minimum(alpha_frame_orig, alpha_frame)


    ##################### Extract the real color for pixels on bounderies #######################
    kernel = np.ones((5, 5), dtype=np.float32)/25.0
    curr_extracted = cv2.filter2D(curr_extracted_orig*alpha_frame/255, -1, kernel)
    curr_extracted_norm = cv2.filter2D(alpha_frame/255, -1, kernel)
    curr_extracted_norm = np.where(curr_extracted_norm == 0, 1, curr_extracted_norm)
    curr_extracted = curr_extracted / curr_extracted_norm
    alpha_frame_grey = cv2.cvtColor(alpha_frame.copy(), cv2.COLOR_BGR2GRAY)
    curr_extracted[:,:,0] = np.where(alpha_frame_grey > 254, curr_extracted_orig[:,:,0], curr_extracted[:,:,0])
    curr_extracted[:,:,0] = np.where(alpha_frame_grey == 0, curr_extracted_orig[:,:,0], curr_extracted[:,:,0])
    curr_extracted[:,:,1] = np.where(alpha_frame_grey > 254, curr_extracted_orig[:,:,1], curr_extracted[:,:,1])
    curr_extracted[:,:,1] = np.where(alpha_frame_grey == 0, curr_extracted_orig[:,:,1], curr_extracted[:,:,1])
    curr_extracted[:,:,2] = np.where(alpha_frame_grey > 254, curr_extracted_orig[:,:,2], curr_extracted[:,:,2])
    curr_extracted[:,:,2] = np.where(alpha_frame_grey == 0, curr_extracted_orig[:,:,2], curr_extracted[:,:,2])

    #curr_extracted = [[[0, 0, 0] if alpha_frame[j][i][0] == 0 else curr_stab[j, i, :] if alpha_frame[j][i][0] == 1 else extract_color(i, j, alpha_frame, curr_stab, params["width"], params["height"]) for i in range(params["width"])] for j in range(params["height"])]
    #curr_extracted = np.array(curr_extracted).reshape(alpha_frame.shape)

    ##################### Matting #######################
    matted_frame = (255 - alpha_frame) * backround / 255 + alpha_frame * curr_extracted / 255
    matted_frame = matted_frame.astype(np.uint8)
    alpha_frame = alpha_frame.astype(np.uint8)
    alpha_frame = cv2.cvtColor(alpha_frame, cv2.COLOR_BGR2GRAY)
    out_matted.write(matted_frame)
    out2_alpha.write(alpha_frame)

# Clear all capture
cap_binary.release()
cap_extracted.release()
cap_stab.release()
out_matted.release()
out2_alpha.release()
cv2.destroyAllWindows()


stop = timeit.default_timer()

print('Time to video matting: ', stop - start)
