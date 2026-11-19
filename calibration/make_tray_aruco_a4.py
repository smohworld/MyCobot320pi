#!/usr/bin/env python3
"""A4 sheets of ArUco markers for the tray handles (DICT_4X4_50).

Tray t uses ID 10+2t (left) and 11+2t (right).
Each row = one tray: [L, R] + [spare L, spare R] (same IDs, backup copies -
stick only ONE pair on each tray).

Each tile is 30 mm x 30 mm = 1-cell white quiet zone + 22.5 mm black marker.
Dashed grey line = 30 mm cut line. Print at 100 % ("actual size"), then check
the 100 mm scale bar with a ruler.

Detection side:  marker_length = 0.0225 (m)   dictionary = DICT_4X4_50
  tray = (id - 10) // 2,  side = id % 2  (0 = left, 1 = right)
(Does not clash with the ChArUco board, which uses DICT_5X5_100.)

Usage:  python3 make_tray_aruco_a4.py [-o tray_aruco_4x4_a4.pdf]
"""
import argparse

import cv2
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

DICT = cv2.aruco.DICT_4X4_50
MARKER_BITS = 4            # data bits per side
TILE_MM = 30.0             # cut size incl. white quiet zone
CELLS = MARKER_BITS + 2 + 2  # data + black border(2) + white quiet zone(2) = 8
CELL_MM = TILE_MM / CELLS  # 3.75 mm
MARKER_MM = CELL_MM * (MARKER_BITS + 2)  # 22.5 mm

ID_OFFSET = 10             # first marker ID
N_TRAYS = 10               # tray t -> ID 10+2t (left), 11+2t (right)  => IDs 10..29
SPARES = True              # add a backup copy of every pair
TRAYS_PER_PAGE = 5

GAP_X, GAP_Y = 8.0, 12.0   # mm between tiles (room for labels)
SET_GAP_X = 24.0           # extra space between main pair and spare pair


def marker_bits(d, mid):
    # 1 px per cell -> (MARKER_BITS+2)^2 matrix incl. black border, 0=black
    return cv2.aruco.generateImageMarker(d, mid, MARKER_BITS + 2, borderBits=1)


def draw_tile(c, d, mid, tx, ty, label):
    c.setStrokeGray(0.6)
    c.setLineWidth(0.3)
    c.setDash(2, 2)
    c.rect(tx * mm, ty * mm, TILE_MM * mm, TILE_MM * mm, stroke=1, fill=0)
    c.setDash()

    bits = marker_bits(d, mid)
    n = bits.shape[0]
    mx, my = tx + CELL_MM, ty + CELL_MM
    c.setFillGray(0)
    for i in range(n):          # image row (top -> bottom)
        for j in range(n):
            if bits[i, j] == 0:
                c.rect((mx + j * CELL_MM) * mm, (my + (n - 1 - i) * CELL_MM) * mm,
                       CELL_MM * mm, CELL_MM * mm, stroke=0, fill=1)

    c.setFont('Helvetica', 8)
    c.drawCentredString((tx + TILE_MM / 2) * mm, (ty - 4) * mm, label)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-o', '--out', default='tray_aruco_4x4_a4.pdf')
    a = ap.parse_args()

    d = cv2.aruco.getPredefinedDictionary(DICT)
    W, H = A4
    c = canvas.Canvas(a.out, pagesize=A4)

    # column x offsets (mm, relative): L, R [, spare L, spare R]
    cols = [0.0, TILE_MM + GAP_X]
    if SPARES:
        s0 = cols[1] + TILE_MM + SET_GAP_X
        cols += [s0, s0 + TILE_MM + GAP_X]
    grid_w = cols[-1] + TILE_MM
    x0 = (W / mm - grid_w) / 2
    hx = 14  # header / scale bar x (mm)
    last_id = ID_OFFSET + 2 * N_TRAYS - 1
    n_pages = (N_TRAYS + TRAYS_PER_PAGE - 1) // TRAYS_PER_PAGE

    for page in range(n_pages):
        c.setFont('Helvetica-Bold', 11)
        c.drawString(hx * mm, (H / mm - 14) * mm,
                     f'ArUco DICT_4X4_50  |  tile {TILE_MM:.0f} mm (cut on dashed line)  |  '
                     f'marker_length = {MARKER_MM:.1f} mm')
        c.setFont('Helvetica', 8)
        c.drawString(hx * mm, (H / mm - 19) * mm,
                     f'OpenCV {cv2.__version__}  |  IDs {ID_OFFSET}-{last_id}  |  tray = (id-{ID_OFFSET}) // 2, '
                     f'side = id % 2 (0=L, 1=R)  |  print at 100 %  |  page {page + 1}/{n_pages}')

        y_top = H / mm - 34
        if SPARES:
            c.setFont('Helvetica-Bold', 9)
            c.drawCentredString((x0 + cols[1] / 2 + TILE_MM / 2) * mm, (y_top + 3) * mm, 'USE')
            c.drawCentredString((x0 + (cols[2] + cols[3]) / 2 + TILE_MM / 2) * mm,
                                (y_top + 3) * mm, 'SPARE (backup copy)')

        trays = range(page * TRAYS_PER_PAGE, min(N_TRAYS, (page + 1) * TRAYS_PER_PAGE))
        for r, t in enumerate(trays):
            ty = y_top - TILE_MM - r * (TILE_MM + GAP_Y + 8)
            ids = [ID_OFFSET + 2 * t, ID_OFFSET + 2 * t + 1]
            for k, cx in enumerate(cols):
                mid = ids[k % 2]
                side = 'L' if k % 2 == 0 else 'R'
                spare = ' spare' if k >= 2 else ''
                draw_tile(c, d, mid, x0 + cx, ty, f'Tray {t} {side}{spare}  (ID {mid})')

        # 100 mm scale bar
        sy = 14
        c.setStrokeGray(0)
        c.setLineWidth(0.6)
        c.line(hx * mm, sy * mm, (hx + 100) * mm, sy * mm)
        for t in range(0, 101, 10):
            h = 3 if t % 50 == 0 else 1.5
            c.line((hx + t) * mm, sy * mm, (hx + t) * mm, (sy + h) * mm)
        c.setFont('Helvetica', 8)
        c.drawString((hx + 103) * mm, (sy - 1) * mm, '100 mm - check with a ruler')
        c.showPage()

    c.save()
    print('saved', a.out, f'({n_pages} pages, IDs {ID_OFFSET}-{last_id})')


if __name__ == '__main__':
    main()
