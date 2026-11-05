#!/usr/bin/env python3
"""Generate a printable ChArUco board (PDF + PNG) at exact physical size.

Default: A4, 5 x 7 squares, 35 mm squares, 26 mm markers, DICT_5X5_100.
Print at 100 % ("actual size"), then measure a square and put the measured
value in charuco_board.yaml.

Usage:
  python3 make_charuco_board.py                 # defaults
  python3 make_charuco_board.py --square 30 --marker 22 --out my_board
"""
import argparse
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PAPER_MM = {'A4': (210, 297), 'A3': (297, 420)}
DICTS = {
    'DICT_4X4_50': cv2.aruco.DICT_4X4_50,
    'DICT_5X5_100': cv2.aruco.DICT_5X5_100,
    'DICT_6X6_250': cv2.aruco.DICT_6X6_250,
}


def mm2px(mm, dpi):
    return int(round(mm / 25.4 * dpi))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cols', type=int, default=5, help='squares in x')
    p.add_argument('--rows', type=int, default=7, help='squares in y')
    p.add_argument('--square', type=float, default=35.0, help='square size [mm]')
    p.add_argument('--marker', type=float, default=26.0, help='marker size [mm]')
    p.add_argument('--dict', default='DICT_5X5_100', choices=DICTS)
    p.add_argument('--paper', default='A4', choices=PAPER_MM)
    p.add_argument('--dpi', type=int, default=600)
    p.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                 'charuco_board'))
    a = p.parse_args()

    if a.cols % 2 == 0 and a.rows % 2 == 0:
        raise SystemExit('cols and rows both even -> marker layout differs between OpenCV '
                         'versions; use at least one odd count')
    pw, ph = PAPER_MM[a.paper]
    bw, bh = a.cols * a.square, a.rows * a.square
    if bw > pw - 20 or bh > ph - 30:
        raise SystemExit(f'board {bw}x{bh} mm does not fit {a.paper} with margins')

    d = cv2.aruco.getPredefinedDictionary(DICTS[a.dict])
    board = cv2.aruco.CharucoBoard((a.cols, a.rows), a.square / 1000.0, a.marker / 1000.0, d)

    # Square size in whole pixels so every square prints the same size.
    sq_px = mm2px(a.square, a.dpi)
    board_img = board.generateImage((a.cols * sq_px, a.rows * sq_px), marginSize=0,
                                    borderBits=1)

    page = np.full((mm2px(ph, a.dpi), mm2px(pw, a.dpi)), 255, np.uint8)
    y0 = mm2px(10, a.dpi)
    x0 = (page.shape[1] - board_img.shape[1]) // 2
    page[y0:y0 + board_img.shape[0], x0:x0 + board_img.shape[1]] = board_img

    # 100 mm scale bar + parameter label below the board, to check print scaling.
    img = Image.fromarray(page)
    draw = ImageDraw.Draw(img)
    yb = y0 + board_img.shape[0] + mm2px(6, a.dpi)
    xb = mm2px(15, a.dpi)
    lw = max(2, mm2px(0.4, a.dpi))
    draw.line([(xb, yb), (xb + mm2px(100, a.dpi), yb)], fill=0, width=lw)
    for t in range(0, 101, 10):
        x = xb + mm2px(t, a.dpi)
        draw.line([(x, yb - mm2px(2 if t % 50 else 4, a.dpi)), (x, yb)], fill=0, width=lw)
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', mm2px(3.5, a.dpi))
    except OSError:
        font = ImageFont.load_default()
    label = (f'ChArUco {a.cols}x{a.rows}  square {a.square:g} mm  marker {a.marker:g} mm  '
             f'{a.dict}  (scale bar = 100 mm, print at 100 %)')
    draw.text((xb, yb + mm2px(2, a.dpi)), label, fill=0, font=font)

    img.save(a.out + '.png', dpi=(a.dpi, a.dpi))
    img.convert('RGB').save(a.out + '.pdf', resolution=a.dpi)

    with open(a.out + '.yaml', 'w') as f:
        f.write('# ChArUco board used for camera + hand-eye calibration.\n'
                '# After printing, measure one square (ideally 5 squares / 5) and replace\n'
                '# square_length / marker_length with the measured values (metres).\n'
                f'dictionary: {a.dict}\n'
                f'squares_x: {a.cols}\n'
                f'squares_y: {a.rows}\n'
                f'square_length: {a.square / 1000.0}\n'
                f'marker_length: {a.marker / 1000.0}\n')

    # Self check: detect the generated board.
    corners, ids, _, _ = cv2.aruco.CharucoDetector(board).detectBoard(board_img)
    n = 0 if corners is None else len(corners)
    print(f'wrote {a.out}.pdf/.png/.yaml  ({bw:g} x {bh:g} mm on {a.paper}, '
          f'square {sq_px / a.dpi * 25.4:.3f} mm at {a.dpi} dpi)')
    print(f'self-check: {n}/{(a.cols - 1) * (a.rows - 1)} charuco corners detected')


if __name__ == '__main__':
    main()
