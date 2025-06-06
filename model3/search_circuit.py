"""Search MOSFET circuits by target gain and optional component constraints.

This script searches a training CSV file to find circuits that meet a desired
gain at a specific frequency.
It returns the closest match based on absolute gain error.

Example usage::

    python search_circuit.py --frequency 1e4 --gain 20 --constraints rd=1e3 rs=470
    # use a custom dataset path
    python search_circuit.py --data mydata.csv --frequency 5e3 --gain 15
"""

import argparse
from typing import Dict, List
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

GAIN_PREFIX = "gain_"
# default dataset packaged with the repository
DEFAULT_DATA_FILE = "csvexample.csv"

# Pre-computed netlist template (same topology as ``predict_and_netlist.py``)
NETLIST_TEMPLATE = """Version 4.1
SHEET 1 1116 680
WIRE 336 -240 256 -240
WIRE 256 -224 256 -240
WIRE 336 -224 336 -240
WIRE 336 -128 336 -144
WIRE 336 -128 256 -128
WIRE 400 -128 336 -128
WIRE 256 -112 256 -128
WIRE 400 -112 400 -128
WIRE 400 -16 400 -32
WIRE 480 -16 400 -16
WIRE 576 -16 544 -16
WIRE 400 0 400 -16
WIRE 576 16 576 -16
WIRE 48 80 16 80
WIRE 160 80 128 80
WIRE 256 80 256 -32
WIRE 256 80 224 80
WIRE 352 80 256 80
WIRE 400 112 400 96
WIRE 496 112 400 112
WIRE 576 112 576 96
WIRE 16 128 16 80
WIRE 256 128 256 80
WIRE 400 128 400 112
WIRE 496 128 496 112
WIRE 16 224 16 208
WIRE 256 224 256 208
WIRE 400 224 400 208
WIRE 496 224 496 192
FLAG 256 224 0
FLAG 16 224 0
FLAG 576 112 0
FLAG 400 224 0
FLAG 256 -224 0
FLAG 16 80 in
IOPIN 16 80 In
FLAG 576 -16 out
IOPIN 576 -16 Out
FLAG 496 224 0
SYMBOL voltage 16 112 R0
WINDOW 123 24 124 Left 2
WINDOW 39 0 0 Left 0
SYMATTR Value2 AC 1
SYMATTR InstName V1
SYMATTR Value 0
SYMBOL cap 224 64 R90
WINDOW 0 0 32 VBottom 2
WINDOW 3 32 32 VTop 2
SYMATTR InstName C1
SYMATTR Value {C1:.6g}
SYMBOL res 240 -128 R0
SYMATTR InstName RG1
SYMATTR Value {RG1:.6g}
SYMBOL res 240 112 R0
SYMATTR InstName RG2
SYMATTR Value {RG2:.6g}
SYMBOL nmos 352 0 R0
SYMATTR InstName M1
SYMATTR Value {MOS_MODEL}
SYMBOL res 384 -128 R0
SYMATTR InstName RD
SYMATTR Value {RD:.6g}
SYMBOL cap 544 -32 R90
WINDOW 0 0 32 VBottom 2
WINDOW 3 32 32 VTop 2
SYMATTR InstName C2
SYMATTR Value {C2:.6g}
SYMBOL res 560 0 R0
SYMATTR InstName RL
SYMATTR Value {RL:.6g}
SYMBOL voltage 336 -240 R0
SYMATTR InstName VDD
SYMATTR Value {VDD:.6g}
SYMBOL res 384 112 R0
SYMATTR InstName RS
SYMATTR Value {RS:.6g}
SYMBOL cap 480 128 R0
SYMATTR InstName CS
SYMATTR Value {CS:.6g}
SYMBOL res 144 64 R90
WINDOW 0 0 56 VBottom 2
WINDOW 3 32 56 VTop 2
SYMATTR InstName Rin
SYMATTR Value {Rin:.6g}
TEXT 448 -232 Left 2 !.ac dec 100 1 2G
"""


def parse_constraints(items: List[str]) -> Dict[str, float]:
    """Parse name=value pairs into a dictionary of floats."""
    result: Dict[str, float] = {}
    for item in items:
        if "=" not in item:
            continue
        name, val = item.split("=", 1)
        try:
            result[name.lower()] = float(val)
        except ValueError:
            pass
    return result


def load_data(path: str) -> pd.DataFrame:
    """Load the CSV containing gain and component values."""
    return pd.read_csv(path)


def frequency_to_index(freq: float, num_points: int) -> int:
    """Return the nearest gain index for the given frequency."""
    freqs = np.logspace(np.log10(1), np.log10(2e9), num_points)
    idx = int(np.argmin(np.abs(freqs - freq)))
    return idx


def search(df: pd.DataFrame, gain_col: str, gain_target: float, constraints: Dict[str, float]) -> pd.Series:
    """Return the row in *df* with minimum error that also satisfies constraints."""
    subset = df.copy()
    for name, val in constraints.items():
        name = name.lower()
        if name in subset.columns:
            subset = subset[np.isclose(subset[name], val, rtol=0.1)]
    if subset.empty:
        subset = df  # fallback to full data
    idx = (subset[gain_col] - gain_target).abs().idxmin()
    return subset.loc[idx]


def ascii_circuit(row: pd.Series) -> str:
    """Generate a simple ASCII diagram of the amplifier with component values."""
    return f"""
    VDD({row['vdd']:.2f}V)
      |
     RD {row['rd']:.2e}Ω
      |
Gate--M1--Drain ----> OUT via C2 {row['c2']:.2e}F and RL {row['rl']:.2e}Ω
      |
     RS {row['rs']:.2e}Ω bypassed by CS {row['cs']:.2e}F
      |
     GND
Input -> Rin {row['rin']:.2e}Ω -> [ RG1 {row['rg1']:.2e}Ω | RG2 {row['rg2']:.2e}Ω ] -> Gate
    C1 {row['c1']:.2e}F couples input
"""


def plot_frequency_response(row: pd.Series, gain_cols: List[str]) -> None:
    """Display the gain vs. frequency curve for the selected row."""
    freqs = np.logspace(np.log10(1), np.log10(2e9), len(gain_cols))
    gains = row[gain_cols].values
    plt.figure(figsize=(8, 3))
    plt.semilogx(freqs, gains)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Gain (dB)")
    plt.title("Matched Circuit Frequency Response")
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Search trained circuits")
    parser.add_argument("--frequency", type=float, required=True, help="Frequency in Hz")
    parser.add_argument("--gain", type=float, required=True, help="Target gain in dB")
    parser.add_argument("--constraints", nargs="*", default=[], help="Additional constraints like rd=1000")
    parser.add_argument("--output", default="matched.asc", help="Output ASC file")
    parser.add_argument("--data", default=DEFAULT_DATA_FILE, help="CSV data file")
    args = parser.parse_args()

    df = load_data(args.data)
    gain_cols = [c for c in df.columns if c.startswith(GAIN_PREFIX)]
    idx = frequency_to_index(args.frequency, len(gain_cols))
    gain_col = gain_cols[idx]

    cons = parse_constraints(args.constraints)
    row = search(df, gain_col, args.gain, cons)
    error = float(abs(row[gain_col] - args.gain))

    params = {
        'Rin': row['rin'],
        'RG1': row['rg1'],
        'RG2': row['rg2'],
        'RD': row['rd'],
        'RS': row['rs'],
        'CS': row['cs'],
        'C1': row['c1'],
        'C2': row['c2'],
        'RL': row['rl'],
        'VDD': row['vdd'],
        'MOS_MODEL': 'Si7336ADP',
    }

    netlist = NETLIST_TEMPLATE.format(**params)
    with open(args.output, "w") as f:
        f.write(netlist)

    print(f"Matched row index: {row.name} (error {error:.2f} dB)")
    print(ascii_circuit(row))
    print(f"Netlist written to {args.output}")
    plot_frequency_response(row, gain_cols)


if __name__ == "__main__":
    main()
