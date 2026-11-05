"""Paper-inspired ResNet and the project's quantum extension."""
import numpy as np
import torch
from torch import nn
import pennylane as qml


class ResidualBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.branch = nn.Sequential(nn.BatchNorm1d(width), nn.Linear(width, width * 2),
                                    nn.ReLU(), nn.Dropout(0.1),
                                    nn.Linear(width * 2, width), nn.Dropout(0.1))

    def forward(self, x):
        return x + self.branch(x)


class TabularResNet(nn.Module):
    """Equation 2 structure; modest fixed widths, not the paper's tuned model."""
    def __init__(self, width=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(8, width), ResidualBlock(width),
                                 ResidualBlock(width), nn.BatchNorm1d(width),
                                 nn.ReLU(), nn.Linear(width, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class QuantumRegressor(nn.Module):
    """Eight qubits, data re-uploading, trainable RY/RX, ring CNOTs, Z readout.

    Analytic expectation values (shots=None); simulator backpropagation.
    No dense input network and no direct classical skip around the circuit.
    """
    def __init__(self, layers=2):
        super().__init__()
        self.layers = layers
        self.theta = nn.Parameter(0.1 * torch.randn(layers, 8, 2))
        self.readout = nn.Linear(8, 1)
        device = qml.device('default.qubit', wires=8, shots=None)

        @qml.qnode(device, interface='torch', diff_method='backprop')
        def circuit(angles, theta):
            for layer in range(layers):
                for wire in range(8):
                    qml.RY(angles[..., wire], wires=wire)
                    qml.RY(theta[layer, wire, 0], wires=wire)
                    qml.RX(theta[layer, wire, 1], wires=wire)
                for wire in range(8):
                    qml.CNOT(wires=[wire, (wire + 1) % 8])
            return [qml.expval(qml.PauliZ(wire)) for wire in range(8)]

        self.circuit = circuit

    def forward(self, x):
        # Smoothly bound standardized features to (-pi, pi), without hard clipping.
        angles = 2 * torch.atan(x)
        measured = torch.stack(self.circuit(angles, self.theta), dim=-1).to(x.dtype)
        return self.readout(measured).squeeze(-1)


def build_model(name):
    if name == 'resnet':
        return TabularResNet()
    if name in ('qnn', 'qnn_1layer'):
        return QuantumRegressor(layers=2 if name == 'qnn' else 1)
    raise ValueError(f'Unknown neural model: {name}')


def predict_neural(model, x, batch_size=128):
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(torch.as_tensor(x[i:i + batch_size], dtype=torch.float32))
                               .cpu().numpy() for i in range(0, len(x), batch_size)])
