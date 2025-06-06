import csv
from typing import Dict, Any

CSV_FILE = "model3/csvexample.csv"
COMPONENT_COLS = [
    'rin', 'rg1', 'rg2', 'rd', 'rs', 'cs', 'c1', 'c2', 'rl', 'vdd'
]
UNITS = {
    'rin': 'Ω', 'rg1': 'Ω', 'rg2': 'Ω', 'rd': 'Ω', 'rs': 'Ω',
    'rl': 'Ω', 'cs': 'F', 'c1': 'F', 'c2': 'F', 'vdd': 'V'
}


def load_row(index: int, csv_file: str = CSV_FILE) -> Dict[str, Any]:
    """Return the component row at the given index from the CSV file."""
    with open(csv_file, newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i == index:
                return {k: float(row[k]) for k in COMPONENT_COLS}
    raise IndexError(f"Row {index} out of range")


def fmt(name: str, value: float) -> str:
    """Format component value with unit."""
    unit = UNITS.get(name, '')
    return f"{value:.6g}{unit}"


def ascii_circuit(row: Dict[str, float]) -> str:
    """Return an ASCII schematic for the given component row."""
    return f"""
          VDD ({fmt('vdd', row['vdd'])})
           |
          RD ({fmt('rd', row['rd'])})
           |
 in--Rin({fmt('rin', row['rin'])})--C1({fmt('c1', row['c1'])})--+--M1--C2({fmt('c2', row['c2'])})--RL({fmt('rl', row['rl'])})--> out
                         |           |
                      RG1({fmt('rg1', row['rg1'])})   RS({fmt('rs', row['rs'])})
                         |           |
                      RG2({fmt('rg2', row['rg2'])})  CS({fmt('cs', row['cs'])})
                         |           |
                        GND         GND
"""


def main(index: int = 0) -> None:
    row = load_row(index)
    print(ascii_circuit(row))


if __name__ == "__main__":
    main()
