import numpy as np
import cv2
import json


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


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

def get_object_bbox(mask_frame):
    """Get an OpenCV capture object and extract its parameters.
    Args:
        mask_frame: The binary mask
    Returns:
        the bounding box of the largest connected component
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_frame, 4, cv2.CV_32S)

    sizes = stats[:, -1]
    max_label = 1
    max_size = sizes[1]
    for i in range(2, num_labels):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]
    return stats[max_label][0], stats[max_label][1], stats[max_label][2], stats[max_label][3], stats[max_label][4], centroids[max_label][0], centroids[max_label][1]




data = {}
#################### Start session ##########################################################
ID1 = '308345891'
ID2 = '211670849'
input_video_name = f'../Outputs/{ID1}_{ID2}_matted.avi'
input_video_name_2 = f'../Outputs/{ID1}_{ID2}_binary.avi'
output_video_name = f'../Outputs/{ID1}_{ID2}_OUTPUT.avi'


cap = cv2.VideoCapture(input_video_name)
cap_2 = cv2.VideoCapture(input_video_name_2)
params = get_video_parameters(cap)
n_frames = params["frame_count"]
w, h = params["width"], params["height"]

out = cv2.VideoWriter(output_video_name, fourcc=cv2.VideoWriter_fourcc(*'XVID'), fps=params["fps"],
                      frameSize=(w, h), isColor=True)


for i in range(n_frames):
    success, frame = cap.read()
    if not success:
        break
    success, mask_frame = cap_2.read()
    if not success:
        break


    mask_frame = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)

    ##################### Find bounding boxes of the biggest object #######################
    l, t, w, h, s, x, y = get_object_bbox(mask_frame)
    new_frame = frame.copy()
    data[str(i+1)] = np.array([t, l, h, w])
    cv2.rectangle(new_frame, (l, t), (l+w, t+h), (0, 255, 0), 2)
    out.write(new_frame)


# Clear all capture
cap.release()
cap_2.release()
out.release()
cv2.destroyAllWindows()


with open('../Outputs/tracking.json', 'w') as fp:
    json.dump(data, fp, cls=NumpyEncoder)



