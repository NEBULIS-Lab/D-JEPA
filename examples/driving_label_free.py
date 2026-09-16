"""Synthetic CPU interface example, not an experimental driving result."""
import numpy as np
from djepa.driving.alignment import Aligner
from djepa.driving.score import predict

prediction = {
    'candidate_id': np.arange(32),
    'query_feature': np.zeros((32, 256), dtype=np.float32),
    'trajectory': np.zeros((32, 8, 3), dtype=np.float32),
    'native_score': np.linspace(0, 1, 32, dtype=np.float32),
    'predicted_factors': np.zeros((32, 6), dtype=np.float32),
}
model = Aligner(288, 'relation')  # Zero-initialized correction preserves native.
bundle = {'models': {'relation': {
    'state': model.state_dict(), 'dim': 288, 'alpha': 1.,
    'mean': np.zeros(288, dtype=np.float32), 'std': np.ones(288, dtype=np.float32),
}}}
result = predict(bundle, prediction)
assert result['candidate_id'] == int(prediction['native_score'].argmax())
assert result['labels_used'] is False
print('PASS: synthetic label-free interface; no simulator or trained weights used')
