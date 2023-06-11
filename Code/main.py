import os
import timeit
import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


data = {
    "time_to_stabilize": 1,
    "time_to_binary": 2,
    "time_to_alpha": 3,
    "time_to_matted": 4,
    "time_to_output": 5,
}


# Set the working directory
os.chdir('Code')

# Call Video_Stabilization.py
start = timeit.default_timer()
os.system('python Video_Stabilization.py')
stop = timeit.default_timer()
duration = stop - start
print('Time to video stabilization: ', duration)
data["time_to_stabilize"] = duration

# Call background subtraction
start = timeit.default_timer()
os.system('python Background_Subtraction.py')
stop = timeit.default_timer()
duration = stop - start
print('Time to background  subtraction: ', duration)
data["time_to_binary"] = duration

# Call Matting
start = timeit.default_timer()
os.system('python Matting.py')
stop = timeit.default_timer()
duration = stop - start
print('Time to video matting: ', duration)
data["time_to_matted"] = duration
data["time_to_alpha"] = 'Same as Matted we solve them together'

# Call Tracking
start = timeit.default_timer()
os.system('python Tracking.py')
stop = timeit.default_timer()
duration = stop - start
print('Time to tracking: ', duration)
data["time_to_output"] = duration

with open('../Outputs/timing.json', 'w') as fp:
    json.dump(data, fp, cls=NumpyEncoder)