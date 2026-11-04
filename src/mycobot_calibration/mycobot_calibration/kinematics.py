"""Serial-chain FK / numerical IK straight from the URDF (no MoveIt, no KDL)."""
import xml.etree.ElementTree as ET

import numpy as np

from mycobot_calibration.geometry import make_T, rpy_to_R


def _axis_rot(axis, q):
    a = np.asarray(axis, float)
    a /= np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return make_T(np.eye(3) + np.sin(q) * K + (1 - np.cos(q)) * K @ K)


class Chain:
    """Kinematic chain base_link -> tip_link parsed from a URDF string."""

    def __init__(self, urdf_xml, base_link, tip_link):
        root = ET.fromstring(urdf_xml)
        by_child = {j.find('child').get('link'): j for j in root.findall('joint')}
        joints = []
        link = tip_link
        while link != base_link:
            if link not in by_child:
                raise ValueError(f'no path from {base_link} to {tip_link}')
            j = by_child[link]
            joints.append(j)
            link = j.find('parent').get('link')
        joints.reverse()

        self.elements = []   # (origin T, axis or None, joint name, child link)
        self.joint_names, self.lower, self.upper = [], [], []
        for j in joints:
            o = j.find('origin')
            xyz = [float(v) for v in (o.get('xyz', '0 0 0') if o is not None else '0 0 0').split()]
            rpy = [float(v) for v in (o.get('rpy', '0 0 0') if o is not None else '0 0 0').split()]
            T0 = make_T(rpy_to_R(*rpy), xyz)
            if j.get('type') in ('revolute', 'continuous'):
                axis = [float(v) for v in j.find('axis').get('xyz').split()]
                lim = j.find('limit')
                self.joint_names.append(j.get('name'))
                self.lower.append(float(lim.get('lower')) if lim is not None else -np.pi)
                self.upper.append(float(lim.get('upper')) if lim is not None else np.pi)
                self.elements.append((T0, axis, j.get('name'), j.find('child').get('link')))
            elif j.get('type') == 'fixed':
                self.elements.append((T0, None, j.get('name'), j.find('child').get('link')))
            else:
                raise ValueError(f'unsupported joint type {j.get("type")}')
        self.lower, self.upper = np.array(self.lower), np.array(self.upper)
        self.n = len(self.joint_names)

    def fk(self, q, all_links=False):
        T = np.eye(4)
        frames = {}
        k = 0
        for T0, axis, _, child in self.elements:
            T = T @ T0
            if axis is not None:
                T = T @ _axis_rot(axis, q[k])
                k += 1
            frames[child] = T
        return frames if all_links else T

    def _err(self, T, Tt):
        dp = Tt[:3, 3] - T[:3, 3]
        Re = Tt[:3, :3] @ T[:3, :3].T
        w = np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]]) / 2.0
        return np.concatenate([dp, w])

    def ik(self, T_target, q_seed, restarts=8, iters=150, tol_pos=1e-4, tol_rot=1e-3,
           rng=None):
        """Damped least squares from q_seed, then random restarts. Returns q or None."""
        rng = rng or np.random.default_rng(0)
        seeds = [np.asarray(q_seed, float)] + [
            rng.uniform(self.lower, self.upper) for _ in range(restarts)]
        best = None
        for q in seeds:
            q = np.clip(q.copy(), self.lower, self.upper)
            lam = 1e-2
            for _ in range(iters):
                T, J = self.fk_jacobian(q)
                e = self._err(T, T_target)
                if np.linalg.norm(e[:3]) < tol_pos and np.linalg.norm(e[3:]) < tol_rot:
                    break
                dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(6), e)
                q = np.clip(q + dq, self.lower, self.upper)
            e = self._err(self.fk(q), T_target)
            if np.linalg.norm(e[:3]) < tol_pos * 10 and np.linalg.norm(e[3:]) < tol_rot * 10:
                if best is None or np.abs(q - q_seed).sum() < np.abs(best - q_seed).sum():
                    best = q
                if best is q and np.abs(q - q_seed).max() < np.pi / 2:
                    break   # near the seed: good enough, avoid big reconfigurations
        return best

    def fk_jacobian(self, q):
        """Tip pose and geometric Jacobian (world-frame [v; w]) in one pass."""
        T = np.eye(4)
        axes, origins = [], []
        k = 0
        for T0, axis, _, _ in self.elements:
            T = T @ T0
            if axis is not None:
                a = np.asarray(axis, float)
                axes.append(T[:3, :3] @ (a / np.linalg.norm(a)))
                origins.append(T[:3, 3].copy())
                T = T @ _axis_rot(axis, q[k])
                k += 1
        J = np.zeros((6, self.n))
        for i, (z, o) in enumerate(zip(axes, origins)):
            J[:3, i] = np.cross(z, T[:3, 3] - o)
            J[3:, i] = z
        return T, J
