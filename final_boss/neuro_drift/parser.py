import torch

def parse_calibration_to_tensor(cal_obj, qubit_map):
    """Flattens calibration dictionary into vector y."""
    features = []
    # Extract T1s
    t1_map = cal_obj.get('T1', {})
    for q in qubit_map:
        val = t1_map.get(q, [0.0])
        features.append(val if isinstance(val, float) else val[0])
    
    return torch.tensor(features, dtype=torch.float32)