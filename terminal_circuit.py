"""Print the California Housing QNN gate diagram, without training or inference."""
import argparse
import shutil
import sys
import textwrap

from housing.models import QuantumRegressor
import pennylane as qml
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, help="diagram width (default: terminal width)")
    parser.add_argument("--ascii", action="store_true", help="use plain ASCII characters")
    args = parser.parse_args()
    width = args.width if args.width is not None else shutil.get_terminal_size((80, 24)).columns
    if width < 24:
        parser.error("Use at least 24 columns so the wire labels and gates fit.")

    model = QuantumRegressor(layers=2)
    # The arguments provide shapes. decimals=None hides example parameter values.
    diagram = qml.draw(model.circuit, decimals=None, max_length=width, show_matrices=False)(
        torch.zeros(8), model.theta
    )
    try:
        diagram.encode(sys.stdout.encoding or "ascii")
    except UnicodeEncodeError:
        args.ascii = True
    if args.ascii:
        diagram = diagram.translate(str.maketrans({
            "─": "-", "│": "|", "╭": "+", "╰": "+", "╮": "+", "╯": "+",
            "├": "+", "┤": "+", "┼": "+", "┊": ":", "╵": "|", "╷": "|",
            "●": "o", "⟨": "<", "⟩": ">", "·": ".", "…": ":",
        }))

    def note(message):
        print(textwrap.fill(message, width=width))

    note("California Housing QNN | 8 qubits, 2 layers")
    print("=" * min(width, 54))
    note("Read left to right. If the drawing splits into blocks, follow the same numbered wire in the next block.")
    print()
    print(diagram)
    print()
    note("Each layer: the first RY on each wire encodes its input feature. The next RY and RX use learned angles.")
    note("CNOT: the dot (o in ASCII) controls the X on the connected wire. The last CNOT in each ring connects wire 7 back to wire 0.")
    note("Readout: <Z> is the Pauli-Z expectation on each wire. A classical linear layer combines the eight values into one prediction.")


if __name__ == "__main__":
    main()
