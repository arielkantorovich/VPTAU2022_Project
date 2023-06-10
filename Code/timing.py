import json

data = {
    "time_to_stabilize": 47,
    "time_to_binary": 87,
    "time_to_alpha": 106,
    "time_to_matted": 106,
    "time_to_output": 7,
}

with open('../Outputs/timing.json', 'w') as fp:
    json.dump(data, fp)