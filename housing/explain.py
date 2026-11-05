"""Trace a saved QNN prediction through every numerical stage for study/demo."""
from pathlib import Path
import numpy as np
import torch
from .data import FEATURES, transform
from .models import build_model
from .experiment import load_run


def explain_prediction(output, index=0):
    torch.set_num_threads(2)
    x, y, prep, splits, records = load_run(output)
    if not 0 <= index < len(splits['test']):
        raise ValueError('Index is outside the saved test partition.')
    record = next(r for r in records if r['model'] == 'qnn')
    seed = record['seed']
    row = splits['test'][index]
    model = build_model('qnn')
    model.load_state_dict(torch.load(Path(output) / 'models' / f'qnn_{seed}.pt',
                                     map_location='cpu', weights_only=True))
    model.eval()
    z = torch.from_numpy(transform(x[row:row+1], prep))
    angles = 2 * torch.atan(z)
    with torch.no_grad():
        measured = torch.stack(model.circuit(angles, model.theta), dim=-1).to(z.dtype)
        contributions = measured[0] * model.readout.weight[0]
        predicted_standardized = model.readout.bias.item() + contributions.sum().item()
        direct = model(z).item()
    np.testing.assert_allclose(predicted_standardized, direct, atol=1e-6)
    prediction = predicted_standardized * float(prep['y_scale']) + float(prep['y_mean'])
    print(f'QNN NUMERICAL WALKTHROUGH | seed {seed} | held-out raw row {row}')
    print(f'{"Feature":12s} {"Raw":>11s} {"Standard z":>11s} {"Angle(rad)":>11s} {"<Z>":>9s} {"w * <Z>":>10s}')
    for j, feature in enumerate(FEATURES):
        print(f'{feature:12s} {x[row,j]:11.4f} {z[0,j]:11.4f} {angles[0,j]:11.4f} '
              f'{measured[0,j]:9.4f} {contributions[j]:10.4f}')
    print(f'Readout bias: {model.readout.bias.item():.6f}')
    print(f'Sum of contributions + bias = {predicted_standardized:.6f} (standardized target)')
    print(f'Undo scaling: {predicted_standardized:.6f} * {float(prep["y_scale"]):.6f} '
          f'+ {float(prep["y_mean"]):.6f} = {prediction:.6f}')
    print(f'Prediction: ${prediction*100000:,.0f}; actual: ${y[row]*100000:,.0f}')
    print('The measurements and contributions are per output wire, NOT feature importance:')
    print('CNOT gates allow each output wire to depend on multiple input features.')
