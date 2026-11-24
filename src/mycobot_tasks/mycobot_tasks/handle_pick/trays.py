"""Several trays: marker IDs, which tray is seen, and the (future) wrong-tray check.

Tray t (0 .. TRAY_COUNT - 1) carries ArUco DICT_4X4_50 markers L = 10 + 2t, R = 11 + 2t
(t 0: 10 / 11, 1: 12 / 13, ... 5: 20 / 21). Every detection grabs whichever tray is seen.

TrayCheck is the place for a later wrong-tray check (e.g. a task list / the shelf system
saying which tray should come): today every tray is accepted.
"""
import math

ID_OFFSET = 10            # the first tray's L marker (same as tray_handle.ID_OFFSET)
TRAY_COUNT = 6            # trays 0 .. 5 = marker IDs 10 .. 21
TRAYS = tuple(range(TRAY_COUNT))


def ids_of(t):
    """tray number -> (L, R) marker IDs"""
    return ID_OFFSET + 2 * t, ID_OFFSET + 2 * t + 1


def tray_of(marker_id):
    """marker ID -> tray number, or None (not a tray marker)"""
    t = (int(marker_id) - ID_OFFSET) // 2
    return t if int(marker_id) >= ID_OFFSET and t in TRAYS else None


def pick_tray(found, img_size=None, prefer=None):
    """found {marker id: (corners px, ...)} -> the tray to measure (or None): a tray with a
    marker in view; both markers seen first, then the current one (prefer), then the one
    nearest the image centre."""
    seen = {}
    for i, v in found.items():
        t = tray_of(i)
        if t is not None:
            seen.setdefault(t, []).append(v[0])
    if not seen:
        return None

    def centre_dist(cs):
        if img_size is None:
            return 0.0
        w, h = img_size
        cx = sum(float(c[:, 0].mean()) for c in cs) / len(cs)
        cy = sum(float(c[:, 1].mean()) for c in cs) / len(cs)
        return math.hypot(cx - w / 2, cy - h / 2)
    return min(seen, key=lambda t: (-len(seen[t]), t != prefer, centre_dist(seen[t])))


class TrayCheck:
    """Is tray t the right one? check(t) -> None (ok) or why not. Today: every tray is ok.
    For a later check, replace check() (or app.tray_check) with the expected-tray source."""

    def check(self, t):
        return None
