"""Conservative cross-leg geometry checks without Isaac Sim.

Each checked link is enclosed by a local box containing its visual meshes AND
URDF collision shapes. A positive separating-axis gap is a lower bound on the
surface distance, not an exact signed distance. Negative values mean that the
boxes overlap, not necessarily that the original meshes collide.
Scope: opposite thighs, shanks and feet (including ankle pitch housings), plus
their clearance above the ground. Not a whole-robot collision certificate.
"""

from itertools import product
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def rotation_rpy(rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                     [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr], [-sp, cp*sr, cp*cr]])


def origin(element):
    node = element.find("origin")
    if node is None:
        return np.eye(3), np.zeros(3)
    return rotation_rpy(np.fromstring(node.get("rpy", "0 0 0"), sep=" ")), np.fromstring(node.get("xyz", "0 0 0"), sep=" ")


def box_gap(center_a, rotation_a, half_a, center_b, rotation_b, half_b):
    """Vectorized normalized 15-axis OBB SAT; positive result certifies separation."""
    a, b = rotation_a.swapaxes(-1, -2), rotation_b.swapaxes(-1, -2)
    cross = np.cross(a[:, :, None, :], b[:, None, :, :]).reshape(-1, 9, 3)
    axes = np.concatenate([a, b, cross], axis=1)
    norms = np.linalg.norm(axes, axis=-1)
    axes = axes / np.maximum(norms[..., None], 1e-12)
    radius_a = np.sum(np.abs(axes @ rotation_a) * half_a, axis=-1)
    radius_b = np.sum(np.abs(axes @ rotation_b) * half_b, axis=-1)
    gaps = np.abs(np.einsum("nki,ni->nk", axes, center_b-center_a)) - radius_a - radius_b
    gaps[norms < 1e-9] = -np.inf
    return gaps.max(axis=1)


class LegClearance:
    def __init__(self, urdf, joint_names, base_height=1.5):
        import trimesh

        self.urdf = Path(urdf)
        self.names = list(joint_names)
        self.base_height = base_height
        root = ET.parse(self.urdf).getroot()
        self.children = {}
        self.limits = {}
        for j in root.findall("joint"):
            name = j.get("name")
            R, p = origin(j)
            axis_node = j.find("axis")
            axis = np.fromstring(axis_node.get("xyz"), sep=" ") if axis_node is not None else np.array([0., 0., 1.])
            axis = axis / np.linalg.norm(axis)
            a, b, c = axis
            K = np.array([[0., -c, b], [c, 0., -a], [-b, a, 0.]])
            self.children.setdefault(j.find("parent").get("link"), []).append(
                (name, j.find("child").get("link"), R, p, K))
            if name in self.names and j.find("limit") is not None:
                self.limits[name] = tuple(float(j.find("limit").get(k)) for k in ("lower", "upper"))
        self.links = [f"LINK_{part}_{side}" for side in ("L", "R")
                      for part in ("HIP_YAW", "KNEE_PITCH", "ANKLE_PITCH", "ANKLE_ROLL")]
        self.boxes = {}
        corners = np.array(list(product((-1., 1.), repeat=3)))
        for link in root.findall("link"):
            name = link.get("name")
            if name not in self.links:
                continue
            points = []
            for shape in list(link.findall("visual")) + list(link.findall("collision")):
                R, p = origin(shape)
                geometry = shape.find("geometry")
                if geometry.find("mesh") is not None:
                    mesh = geometry.find("mesh")
                    path = (self.urdf.parent / mesh.get("filename")).resolve()
                    loaded = trimesh.load(path, force="mesh", process=False)
                    vertices = np.asarray(loaded.vertices) * np.fromstring(mesh.get("scale", "1 1 1"), sep=" ")
                elif geometry.find("box") is not None:
                    vertices = corners * np.fromstring(geometry.find("box").get("size"), sep=" ") / 2
                elif geometry.find("sphere") is not None:
                    vertices = corners * float(geometry.find("sphere").get("radius"))
                elif geometry.find("cylinder") is not None:
                    node = geometry.find("cylinder")
                    vertices = corners * [float(node.get("radius")), float(node.get("radius")), float(node.get("length"))/2]
                else:
                    raise ValueError(f"Unsupported geometry in {name}")
                points.append(vertices @ R.T + p)
            if not points:
                raise ValueError(f"Missing geometry for {name}")
            vertices = np.concatenate(points)
            lo, hi = vertices.min(axis=0), vertices.max(axis=0)
            self.boxes[name] = ((lo+hi)/2, (hi-lo)/2)
        if set(self.boxes) != set(self.links):
            raise ValueError("Incomplete leg geometry")
        self.pairs = list(product(self.links[:4], self.links[4:]))

    def frames(self, q):
        q = np.atleast_2d(np.asarray(q, dtype=float))
        if q.ndim != 2 or not len(q) or q.shape[1] != len(self.names) or not np.isfinite(q).all():
            raise ValueError("Invalid physical joint positions")
        positions = {name: q[:, i] for i, name in enumerate(self.names)}
        frames = {}

        def walk(link, R, p):
            if link in self.boxes:
                center, half = self.boxes[link]
                frames[link] = (p + np.einsum("nij,j->ni", R, center), R, half)
            for name, child, R0, p0, K in self.children.get(link, []):
                # Upper-body branches do not affect fixed-base leg geometry.
                if name == "J12_TORSO_YAW":
                    continue
                pj = p + np.einsum("nij,j->ni", R, p0)
                theta = np.asarray(positions.get(name, np.zeros(len(q))))[:, None, None]
                Rj = R @ R0 @ (np.eye(3) + np.sin(theta)*K + (1-np.cos(theta))*(K@K))
                walk(child, Rj, pj)

        walk("LINK_BASE", np.broadcast_to(np.eye(3), (len(q), 3, 3)), np.zeros((len(q), 3)))
        return frames

    def check(self, physical_q, margin=.02, joint_margin=.01):
        if not np.isfinite([margin, joint_margin]).all() or margin < 0 or joint_margin < 0:
            raise ValueError("Margins must be finite and nonnegative")
        q = np.atleast_2d(np.asarray(physical_q, dtype=float))
        frames = self.frames(q)
        gaps = np.stack([box_gap(*frames[a], *frames[b]) for a, b in self.pairs], axis=1)
        frame, pair = np.unravel_index(gaps.argmin(), gaps.shape)
        ground = min(float((p[:, 2] + self.base_height - np.sum(np.abs(R[:, 2, :])*half, axis=1)).min())
                     for p, R, half in frames.values())
        limit_gap = min(min(float((q[:, i]-self.limits[name][0]).min()),
                            float((self.limits[name][1]-q[:, i]).min()))
                        for i, name in enumerate(self.names) if name in self.limits and
                        any(k in name for k in ("HIP_", "KNEE_", "ANKLE_")))
        gap = float(gaps[frame, pair])
        return {"passed": gap >= margin and ground >= margin and limit_gap >= joint_margin,
                "min_gap_lower_bound_m": gap, "pair": list(self.pairs[pair]),
                "frame": int(frame), "min_ground_gap_m": ground,
                "min_joint_limit_gap_rad": limit_gap, "margin_m": margin,
                "scope": "16 cross-leg box pairs (visual+collision envelopes), ground, leg joint limits"}


def with_midpoints(q):
    q = np.asarray(q)
    if q.ndim != 2 or not len(q):
        raise ValueError("Expected a nonempty joint trajectory")
    dense = np.empty((2*len(q)-1, q.shape[1]), dtype=q.dtype)
    dense[::2] = q
    dense[1::2] = (q[:-1]+q[1:])/2
    return dense
