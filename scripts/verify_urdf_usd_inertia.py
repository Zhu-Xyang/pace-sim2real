#!/usr/bin/env python3
"""Verify that URDF link mass / COM / inertia match the converted USD.

Usage:
    python scripts/pace/verify_urdf_usd_inertia.py \
        --urdf assets_robot/s800/urdf/serial_robot_s_42dof.urdf \
        --usd  assets_robot/s800/usd/serial_robot_s_42dof/payloads/Physics/physics.usda

The script parses <inertial> from the URDF and physics:mass /
physics:centerOfMass / physics:diagonalInertia from the USDA text,
then compares:
  - mass           (direct float compare)
  - center of mass (direct float compare)
  - inertia        (eigenvalues of URDF 6-element matrix vs USD diagonal)

URDF stores the full 6-element symmetric inertia tensor in the
<inertial><origin> frame.  USD stores the diagonalised principal
inertia plus a principalAxes quaternion.  Comparing eigenvalues of
the URDF matrix to the USD diagonalInertia is the correct check.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

# ── eigenvalues of 3×3 symmetric matrix (Jacobi) ──────────────────────────

def jacobi_eigenvalues_3x3(A: list[list[float]],
                           max_iter: int = 100,
                           tol: float = 1e-15) -> list[float]:
    """Return sorted eigenvalues of a 3×3 real symmetric matrix."""
    a = [row[:] for row in A]
    for _ in range(max_iter):
        # find largest off-diagonal element
        p, q = 0, 1
        max_off = abs(a[0][1])
        if abs(a[0][2]) > max_off:
            p, q = 0, 2
            max_off = abs(a[0][2])
        if abs(a[1][2]) > max_off:
            p, q = 1, 2
            max_off = abs(a[1][2])
        if max_off < tol:
            break
        if abs(a[p][p] - a[q][q]) < 1e-30:
            theta = math.pi / 4
        else:
            theta = 0.5 * math.atan2(2 * a[p][q], a[p][p] - a[q][q])
        c, s = math.cos(theta), math.sin(theta)
        app, aqq, apq = a[p][p], a[q][q], a[p][q]
        a[p][p] = c * c * app + 2 * s * c * apq + s * s * aqq
        a[q][q] = s * s * app - 2 * s * c * apq + c * c * aqq
        a[p][q] = a[q][p] = 0.0
        for r in range(3):
            if r != p and r != q:
                arp, arq = a[r][p], a[r][q]
                a[r][p] = a[p][r] = c * arp + s * arq
                a[r][q] = a[q][r] = -s * arp + c * arq
    return sorted([a[0][0], a[1][1], a[2][2]])

# ── data containers ────────────────────────────────────────────────────────

@dataclass
class LinkInertia:
    mass: float
    com: list[float]
    inertia: list[list[float]]  # 3×3 symmetric

# ── URDF parser ────────────────────────────────────────────────────────────

def parse_urdf(path: str) -> dict[str, LinkInertia]:
    tree = ET.parse(path)
    root = tree.getroot()
    links: dict[str, LinkInertia] = {}
    for link in root.findall("link"):
        name = link.get("name")
        inertial = link.find("inertial")
        if inertial is None:
            continue
        mass_elem = inertial.find("mass")
        mass = float(mass_elem.get("value", "0")) if mass_elem is not None else 0.0
        origin = inertial.find("origin")
        com = [float(x) for x in origin.get("xyz", "0 0 0").split()] if origin is not None else [0, 0, 0]
        inertia_elem = inertial.find("inertia")
        if inertia_elem is not None:
            ixx = float(inertia_elem.get("ixx", "0"))
            ixy = float(inertia_elem.get("ixy", "0"))
            ixz = float(inertia_elem.get("ixz", "0"))
            iyy = float(inertia_elem.get("iyy", "0"))
            iyz = float(inertia_elem.get("iyz", "0"))
            izz = float(inertia_elem.get("izz", "0"))
        else:
            ixx = ixy = ixz = iyy = iyz = izz = 0.0
        links[name] = LinkInertia(
            mass=mass,
            com=com,
            inertia=[[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]],
        )
    return links

# ── USDA text parser ───────────────────────────────────────────────────────

def parse_usd_physics(path: str) -> dict[str, dict]:
    """Parse physics:mass / centerOfMass / diagonalInertia from USDA text."""
    with open(path) as f:
        lines = f.read().split("\n")

    usd_links: dict[str, dict] = {}
    i = 0
    while i < len(lines):
        m = re.match(r'\s*over "(LINK_[A-Z_0-9]+)"\s*\(', lines[i])
        if m:
            link_name = m.group(1)
            data: dict = {}
            for j in range(i + 1, min(i + 8, len(lines))):
                l = lines[j].strip()
                if "physics:mass" in l and "diagonalInertia" not in l:
                    data["mass"] = float(re.search(r"=\s*([\d.eE+-]+)", l).group(1))
                elif "physics:centerOfMass" in l:
                    vals = re.search(r"\(([^)]+)\)", l).group(1).split(",")
                    data["com"] = [float(x) for x in vals]
                elif "physics:diagonalInertia" in l:
                    vals = re.search(r"\(([^)]+)\)", l).group(1).split(",")
                    data["diagonal_inertia"] = [float(x) for x in vals]
                elif l.startswith('over "') and j > i + 1:
                    break
            if "mass" in data:
                usd_links[link_name] = data
        i += 1
    return usd_links

# ── main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Verify URDF→USD inertia consistency")
    parser.add_argument("--urdf", required=True, help="Path to the URDF file")
    parser.add_argument("--usd", required=True, help="Path to the physics.usda file")
    parser.add_argument("--tol", type=float, default=1e-4, help="Tolerance for float comparison")
    args = parser.parse_args()

    urdf_links = parse_urdf(args.urdf)
    usd_links = parse_usd_physics(args.usd)

    print(f"URDF links with inertia: {len(urdf_links)}")
    print(f"USD  links with inertia: {len(usd_links)}")
    print()

    header = f"{'Link':<28s} {'mass_URDF':>10s} {'mass_USD':>10s} {'Δmass':>8s} {'COM':>6s} {'Inertia':>8s}"
    print(header)
    print("-" * len(header))

    all_ok = True
    mismatches = []

    for name in sorted(urdf_links.keys()):
        u = urdf_links[name]
        if name not in usd_links:
            print(f"{name:<28s} {u.mass:>10.6f} {'MISSING':>10s}")
            all_ok = False
            continue

        w = usd_links[name]

        # mass
        mass_err = abs(u.mass - w["mass"])
        mass_ok = mass_err < args.tol

        # COM
        com_ok = True
        com_err = 0.0
        if "com" in w:
            com_err = max(abs(a - b) for a, b in zip(u.com, w["com"]))
            com_ok = com_err < args.tol

        # inertia: eigenvalues of URDF matrix vs USD diagonal
        eigvals = jacobi_eigenvalues_3x3(u.inertia)
        usd_di = sorted(w.get("diagonal_inertia", [0, 0, 0]))
        inertia_err = max(abs(a - b) for a, b in zip(eigvals, usd_di))
        inertia_ok = inertia_err < args.tol

        ok = mass_ok and com_ok and inertia_ok
        if not ok:
            all_ok = False
            mismatches.append((name, u, w, eigvals, usd_di, mass_err, com_err, inertia_err))

        print(f"{name:<28s} {u.mass:>10.6f} {w['mass']:>10.6f} {mass_err:>8.1e} "
              f"{'OK' if com_ok else 'FAIL':>6s} {'OK' if inertia_ok else 'FAIL':>8s}")

    print(f"\n{'=' * len(header)}")
    print(f"Overall: {'ALL MATCH ✅' if all_ok else 'MISMATCHES FOUND ❌'}")

    if mismatches:
        print(f"\n=== {len(mismatches)} mismatches ===")
        for name, u, w, eigvals, usd_di, mass_err, com_err, inertia_err in mismatches:
            print(f"\n  {name}:")
            mat = u.inertia
            print(f"    URDF inertia: ixx={mat[0][0]:.7f} iyy={mat[1][1]:.7f} izz={mat[2][2]:.7f}")
            print(f"                 ixy={mat[0][1]:.7f} ixz={mat[0][2]:.7f} iyz={mat[1][2]:.7f}")
            print(f"    URDF eigenvalues:   {eigvals}")
            print(f"    USD  diagonalInertia: {usd_di}")
            print(f"    Inertia max error: {inertia_err:.2e}")
            if com_err > 0:
                print(f"    COM: URDF={u.com}  USD={w.get('com', 'N/A')}  err={com_err:.2e}")
            if mass_err > args.tol:
                print(f"    Mass error: {mass_err:.2e}")

    sys.exit(0 if all_ok else 1)

if __name__ == "__main__":
    main()
