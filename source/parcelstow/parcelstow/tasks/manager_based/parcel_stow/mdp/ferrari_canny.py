"""Ferrari-Canny inscribed-ball margin from realized contacts.

The contacts-based grasp wrench space uses linearized friction-cone edges
(8 per contact, unit-norm edge forces) and torques about the object
center, and the margin is the radius of the largest origin-centered ball
inscribed in the convex hull of the edge wrenches. A contact set whose
hull does not contain the origin has no force closure and scores -1.0,
the sentinel convention of the released records. Numpy and scipy only,
no simulator import.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull, QhullError


def _local_tangent_basis(n):
    """Two orthonormal tangents to the unit vector n."""
    ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    t1 = ref - n * np.dot(ref, n)
    nrm = np.linalg.norm(t1)
    if nrm < 1e-12:
        ref = np.array([0.0, 0.0, 1.0])
        t1 = ref - n * np.dot(ref, n)
        nrm = np.linalg.norm(t1)
    t1 = t1 / max(nrm, 1e-12)
    t2 = np.cross(n, t1)
    return t1, t2


def grasp_wrench_space(points, normals, mu, n_friction_edges=8, points_about_com=None):
    """Contacts-based basis wrench matrix W (6, N*n_friction_edges).

    Builds the linearized friction-cone-edge wrenches from contact points,
    inward normals, and per-contact friction. mu may be a scalar or one
    value per contact. Returns a (6, 0) array for degenerate inputs.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    normals = np.asarray(normals, dtype=float).reshape(-1, 3)
    mu = np.atleast_1d(np.asarray(mu, dtype=float)).reshape(-1)
    nc = points.shape[0]
    if mu.size == 1:
        mu = np.full(nc, float(mu[0]))
    if nc < 2 or normals.shape[0] != nc or mu.shape[0] != nc:
        return np.zeros((6, 0))
    com = (
        np.asarray(points_about_com, dtype=float).reshape(-1, 3)
        if points_about_com is not None
        else points
    )
    thetas = np.linspace(0.0, 2.0 * np.pi, n_friction_edges, endpoint=False)
    cos_th = np.cos(thetas)
    sin_th = np.sin(thetas)
    wrenches = np.empty((nc * n_friction_edges, 6))
    w = 0
    for i in range(nc):
        n_hat = normals[i] / max(np.linalg.norm(normals[i]), 1e-12)
        t1, t2 = _local_tangent_basis(n_hat)
        m = float(mu[i])
        p_com = com[i]
        for k in range(n_friction_edges):
            f = n_hat + m * (cos_th[k] * t1 + sin_th[k] * t2)
            f = f / max(np.linalg.norm(f), 1e-12)
            wrenches[w, 0:3] = f
            wrenches[w, 3:6] = np.cross(p_com, f)
            w += 1
    return wrenches.T


def inscribed_ball(W):
    """Radius of the largest origin-centered ball inscribed in conv(W columns).

    Returns the minimum signed distance from the origin to the convex-hull
    facets, positive when the origin is contained. A non-contained origin
    clamps to -1.0, the force-closure sentinel. A degenerate hull returns
    -1.0.
    """
    W = np.asarray(W, dtype=float)
    try:
        hull = ConvexHull(W.T)
    except QhullError:
        return -1.0
    margin = float(np.min(-hull.equations[:, -1]))
    return margin if margin >= 0.0 else -1.0


def ferrari_canny_from_contacts(points, normals, mu, n_friction_edges=8, points_about_com=None):
    """Ferrari-Canny inscribed ball from contacts, the contacts-based path."""
    W = grasp_wrench_space(points, normals, mu, n_friction_edges, points_about_com)
    if W.shape[1] == 0:
        return -1.0
    return inscribed_ball(W)
