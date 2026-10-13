"""Render the saved Plotly scenes used for the validation geometry figures."""
import argparse
from pathlib import Path

import plotly.io as pio

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for case, name in (("axisymmetric", "nwl_axisym"), ("iter", "nwl_axisym_shaped"), ("twisty", "nwl_twisty")):
        fig = pio.from_json((ROOT / "geometry" / f"{case}.json").read_text())
        fig.write_html(args.output_dir / f"{name}.html", include_plotlyjs=True)
        fig.write_image(args.output_dir / f"{name}.png", scale=2)


if __name__ == "__main__":
    main()
