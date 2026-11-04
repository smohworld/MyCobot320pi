"""ChArUco board: load from yaml, detect + PnP, and render synthetic views (simulation)."""
import cv2
import numpy as np
import yaml

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
        self.size = (self.cols * self.square, self.rows * self.square)   # [m] x, y
        self.center = np.array([self.size[0] / 2, self.size[1] / 2, 0.0])
        self._texture = None

    # --- detection -------------------------------------------------------
    def detect(self, image, K, D, min_corners=8):
        """Return dict(ok, T_cam_board, n, rms, corners, ids) for one image."""
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ch_corners, ch_ids, _, _ = self.detector.detectBoard(gray)
        out = {'ok': False, 'n': 0 if ch_ids is None else len(ch_ids),
               'corners': ch_corners, 'ids': ch_ids}
        if ch_ids is None or len(ch_ids) < min_corners:
            return out
        obj = self.corners3d[ch_ids.reshape(-1)]
        img = ch_corners.reshape(-1, 2).astype(np.float64)
        # collinear corners give a degenerate PnP
        if np.linalg.matrix_rank(obj[:, :2] - obj[:, :2].mean(0), tol=1e-6) < 2:
            return out
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, D, flags=cv2.SOLVEPNP_IPPE)
        if not ok:
            return out
        rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, D, rvec, tvec)
        proj, _ = cv2.projectPoints(obj, rvec, tvec, K, D)
        rms = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img) ** 2, axis=1))))
        from mycobot_calibration.geometry import rvec_tvec_to_T
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
        cv2.putText(vis, txt, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 255, 0) if det.get('ok') else (0, 0, 255), 2)
        return vis

    # --- simulation ------------------------------------------------------
    def render(self, T_cam_board, K, D, width, height, px_per_m=4000, margin_m=0.01):
        """Synthetic camera image of the board (printed on white paper) on a grey table."""
        if self._texture is None:
            sq_px = int(round(self.square * px_per_m))
            m_px = int(round(margin_m * px_per_m))
            img = self.board.generateImage((self.cols * sq_px, self.rows * sq_px),
                                           marginSize=0, borderBits=1)
            self._texture = cv2.copyMakeBorder(img, m_px, m_px, m_px, m_px,
                                               cv2.BORDER_CONSTANT, value=255)
            self._tex_scale = (self.cols * self.square) / (self.cols * sq_px)  # m per px
            self._tex_margin = m_px
        h_t, w_t = self._texture.shape
        # texture pixel (u, v) -> board metres (x, y) -> camera
        s, m = self._tex_scale, self._tex_margin
        corners_px = np.array([[0, 0], [w_t, 0], [w_t, h_t], [0, h_t]], np.float64)
        corners_m = np.column_stack([(corners_px - m) * s, np.zeros(4)])
        Pc = (T_cam_board[:3, :3] @ corners_m.T).T + T_cam_board[:3, 3]
        bg = np.full((height, width), 110, np.uint8)
        if np.any(Pc[:, 2] < 0.02):
            return bg
        rvec = cv2.Rodrigues(T_cam_board[:3, :3])[0]
        dst, _ = cv2.projectPoints(corners_m, rvec, T_cam_board[:3, 3], K, np.zeros(5))
        # Warp with the pinhole model (homography), then apply lens distortion via remap.
        H = cv2.getPerspectiveTransform(corners_px.astype(np.float32),
                                        dst.reshape(-1, 2).astype(np.float32))
        ones = np.full_like(self._texture, 255)
        warped = cv2.warpPerspective(self._texture, H, (width, height), flags=cv2.INTER_AREA)
        mask = cv2.warpPerspective(ones, H, (width, height), flags=cv2.INTER_NEAREST)
        img = np.where(mask > 0, warped, bg).astype(np.uint8)
        if D is not None and np.any(np.abs(D) > 0):
            # for each distorted output pixel, its undistorted source pixel (cached)
            key = (width, height, tuple(np.ravel(K)), tuple(np.ravel(D)))
            if getattr(self, '_map_key', None) != key:
                uv = np.stack(np.meshgrid(np.arange(width), np.arange(height)), -1)
                uv = uv.reshape(-1, 1, 2).astype(np.float32)
                und = cv2.undistortPoints(uv, K, D, P=K).reshape(height, width, 2)
                self._map = (und[..., 0].copy(), und[..., 1].copy())
                self._map_key = key
            img = cv2.remap(img, self._map[0], self._map[1], cv2.INTER_LINEAR,
                            borderValue=110)
        return img

    def project_outline(self, T_cam_board, K, D):
        """Pixel coords of the 4 outer board corners (None if behind the camera)."""
        c = np.array([[0, 0, 0], [self.size[0], 0, 0], [self.size[0], self.size[1], 0],
                      [0, self.size[1], 0]], np.float64)
        Pc = (T_cam_board[:3, :3] @ c.T).T + T_cam_board[:3, 3]
        if np.any(Pc[:, 2] <= 0.02):
            return None
        rvec = cv2.Rodrigues(T_cam_board[:3, :3])[0]
        p, _ = cv2.projectPoints(c, rvec, T_cam_board[:3, 3], K, D)
        return p.reshape(-1, 2)
