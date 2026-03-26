#!/usr/bin/env python3
"""
Script 00: Generate a printable ChArUco board image from config.

Creates a high-resolution PNG file suitable for printing. Measure the
printed square_length_mm with a caliper and update calibration.yaml
if it doesn't match (printers may scale).

Usage:
    python scripts/00_generate_board.py
    python scripts/00_generate_board.py --output board.png --pps 120
"""

from __future__ import annotations

import argparse

import cv2

from zed2i_calibrate.board import CharucoBoard
from zed2i_calibrate.config import load_config, repo_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate printable ChArUco board image.")
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path (default: results/charuco_board.png)",
    )
    parser.add_argument(
        "--pps", type=int, default=100,
        help="Pixels per square (default: 100). Use 150+ for high-res print.",
    )
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    board = CharucoBoard.from_config(cfg)

    output = args.output or str(repo_root() / "results" / "charuco_board.png")

    img = board.draw_image(pixels_per_square=args.pps)

    cv2.imwrite(output, img)
    print(f"Board:  {board}")
    print(f"Image:  {img.shape[1]}x{img.shape[0]} px")
    print(f"Saved:  {output}")
    print()
    print("IMPORTANT: After printing, measure the square side length with a caliper.")
    print(f"           Expected: {board.square_length_mm} mm")
    print("           Update calibration.yaml if it differs.")


if __name__ == "__main__":
    main()
