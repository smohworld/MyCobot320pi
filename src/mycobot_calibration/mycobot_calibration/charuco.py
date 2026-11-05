"""ChArUco board (calibration/charuco_board.yaml): detection + PnP, drawing, synthetic views.

Board frame (OpenCV): origin at the first chessboard corner, x along squares_x,
y along squares_y, z into the board (camera looks along +z when it faces the print).
"""
import cv2
import numpy as np
import yaml

from mycobot_calibration.geometry import rvec_tvec_to_T

DICTS = {k: getattr(cv2.aruco, k) for k in dir(cv2.aruco) if k.startswith('DICT_')}


class CharucoTarget:

    def __init__(self, yaml_path):
        with open(yaml_path) as f:
            c = yaml.safe_load(f)
        self.cols, self.rows = int(c['squares_x']), int(c['squares_y'])
        self.square = float(c['square_length'])
        self.marker = float(c['marker_length'])
        self.dictionary = cv2.aruco.getPredefinedDictionary(DICTS[c['dictionary']])
        self.board = cv2.aruco.CharucoBoard((self.cols, self.rows), self.square, self.marker,
                                            self.dictionary)
        self.detector = cv2.aruco.CharucoDetector(self.board)
        self.corners3d = self.board.getChessboardCorners().reshape(-1, 3).astype(np.float64)
        self.n_corners = len(self.corners3d)
        self.size = (self.cols * self.square, self.rows * self.square)
        self.center = np.array([self.size[0] / 2, self.size[1] / 2, 0.0])
        self.outline = np.array([[0, 0, 0], [self.size[0], 0, 0],
                                 [self.size[0], self.size[1], 0], [0, self.size[1], 0]], float)
        self._tex = None

    def describe(self):
        return (f'{self.cols}x{self.rows} squares, square {self.square * 1000:.1f} mm, '
                f'marker {self.marker * 1000:.1f} mm, {self.n_corners} corners')

    # ------------------------------------------------------------- detect
    def detect(self, image, K, D, min_corners=8):
        """dict(ok, T_cam_board, n, rms, corners, ids, rvec, tvec)."""
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ch_corners, ch_ids, _, _ = self.detector.detectBoard(gray)
        out = {'ok': False, 'n': 0 if ch_ids is None else len(ch_ids),
               'corners': ch_corners, 'ids': ch_ids, 'rms': float('nan')}
        if ch_ids is None or len(ch_ids) < min_corners:
            return out
        obj = self.corners3d[ch_ids.reshape(-1)]
        img = ch_corners.reshape(-1, 2).astype(np.float64)
        if np.linalg.matrix_rank(obj[:, :2] - obj[:, :2].mean(0), tol=1e-6) < 2:
            return out                      # collinear corners: degenerate PnP
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, D, flags=cv2.SOLVEPNP_IPPE)
        if not ok:
            return out
        rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, D, rvec, tvec)
        proj, _ = cv2.projectPoints(obj, rvec, tvec, K, D)
        rms = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img) ** 2, axis=1))))
        out.update(ok=True, T_cam_board=rvec_tvec_to_T(rvec, tvec), rms=rms,
                   rvec=rvec.reshape(3), tvec=tvec.reshape(3))
        return out

    def draw(self, image, det, K, D):
        vis = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if det.get('corners') is not None and det['n'] > 0:
            cv2.aruco.drawDetectedCornersCharuco(vis, det['corners'], det['ids'], (0, 255, 0))
        if det.get('ok'):
            cv2.drawFrameAxes(vis, K, D, det['rvec'], det['tvec'], self.square * 2)
            txt = f"{det['n']}/{self.n_corners} corners  rms {det['rms']:.2f}px"
        else:
            txt = f"board NOT detected ({det['n']} corners)"
        cv2.putText(vis, txt, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 255, 0) if det.get('ok') else (0, 0, 255), 2)
        return vis

    def visible(self, T_cam_board, K, D, width, height, margin_px=20, max_incidence_deg=60,
                min_fraction=0.8):
        """True if >= min_fraction of the chessboard corners project inside the image
        (margin_px from the border) and the board is not seen too obliquely."""
        Pc = (T_cam_board[:3, :3] @ self.corners3d.T).T + T_cam_board[:3, 3]
        if np.any(Pc[:, 2] < 0.05):
            return False
        n = T_cam_board[:3, 2]                         # board z axis in camera frame
        c = T_cam_board[:3, :3] @ self.center + T_cam_board[:3, 3]
        cosang = abs(np.dot(n, c / np.linalg.norm(c)))
        if np.degrees(np.arccos(np.clip(cosang, -1, 1))) > max_incidence_deg:
            return False
        rvec = cv2.Rodrigues(T_cam_board[:3, :3])[0]
        p, _ = cv2.projectPoints(self.corners3d, rvec, T_cam_board[:3, 3], K, D)
        p = p.reshape(-1, 2)
        inside = ((p[:, 0] > margin_px) & (p[:, 0] < width - margin_px)
                  & (p[:, 1] > margin_px) & (p[:, 1] < height - margin_px))
        return bool(np.mean(inside) >= min_fraction)

    # ---------------------------------------------------------- simulation
    def render(self, T_cam_board, K, width, height, px_per_m=4000, margin_m=0.01):
        """Synthetic grey image of the printed board (pinhole, no distortion)."""
        if self._tex is None:
            sq = int(round(self.square * px_per_m))
            m = int(round(margin_m * px_per_m))
            img = self.board.generateImage((self.cols * sq, self.rows * sq), marginSize=0,
                                           borderBits=1)
            self._tex = cv2.copyMakeBorder(img, m, m, m, m, cv2.BORDER_CONSTANT, value=255)
            self._tex_s, self._tex_m = self.square / sq, m
        h, w = self._tex.shape
        bg = np.full((height, width), 110, np.uint8)
        src = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float64)
        pts = np.column_stack([(src - self._tex_m) * self._tex_s, np.zeros(4)])
        Pc = (T_cam_board[:3, :3] @ pts.T).T + T_cam_board[:3, 3]
        if np.any(Pc[:, 2] < 0.02):
            return bg
        dst = (K @ Pc.T).T
        dst = dst[:, :2] / dst[:, 2:3]
        H = cv2.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32))
        warped = cv2.warpPerspective(self._tex, H, (width, height), flags=cv2.INTER_AREA)
        mask = cv2.warpPerspective(np.full_like(self._tex, 255), H, (width, height),
                                   flags=cv2.INTER_NEAREST)
        return np.where(mask > 0, warped, bg).astype(np.uint8)
