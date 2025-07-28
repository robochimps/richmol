from typing import Literal

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial.transform import Rotation


plt.rcParams.update(
    {"text.usetex": True, "font.family": "serif", "font.serif": ["Computer Modern"]}
)


def plot_mc_dens(
    hyper_states,
    states,
    f: float,
    sym: str,
    m: float,
    istate: int,
    atom_xyz: np.ndarray,
    atom_labels: list[str] | None = None,
    filename: str = "dens",
    npoints: int = 1000000,
    npoints_1d: int = 30,
    coef_thresh: float = 1e-8,
):
    alpha = np.linspace(0, 2 * np.pi, npoints_1d)
    beta = np.linspace(0, np.pi, npoints_1d)
    gamma = np.linspace(0, 2 * np.pi, npoints_1d)

    dens = hyper_states.rot_dens(
        states,
        f,
        sym,
        (m, istate),
        f,
        sym,
        (m, istate),
        spin_dens=False,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        coef_thresh=coef_thresh,
    )

    fdens = RegularGridInterpolator((alpha, beta, gamma), dens)
    pts = np.random.uniform(
        low=[0, 0, 0], high=[2 * np.pi, np.pi, 2 * np.pi], size=(npoints, 3)
    )
    w = fdens(pts)
    abs_w = np.abs(w) / np.max(np.abs(dens))
    eta = np.random.uniform(0.0, 1.0, size=len(w))
    ind = np.where(abs_w > eta)
    points = pts[ind]
    rot_mat = Rotation.from_euler("ZYZ", points).as_matrix()

    natoms = len(atom_xyz)
    xyz_samples = []
    weights_samples = []

    for iatom in range(natoms):
        xyz = np.dot(rot_mat, atom_xyz[iatom] / np.linalg.norm(atom_xyz[iatom]))
        kernel = stats.gaussian_kde(xyz.T)
        weights = kernel(xyz.T)
        xyz_samples.append(xyz)
        weights_samples.append(np.real(weights))

    fig = plt.figure(figsize=(8, 5))
    axs = [
        fig.add_subplot(1, natoms, iatom + 1, projection="3d")
        for iatom in range(natoms)
    ]
    for ax in axs:
        ax.set_box_aspect(aspect=(1, 1, 1))
        ax.axes.set_xlim3d(left=-1.1, right=1.1)
        ax.axes.set_ylim3d(bottom=-1.1, top=1.1)
        ax.axes.set_zlim3d(bottom=-1.1, top=1.1)

    for iatom, ax in enumerate(axs):
        sc = ax.scatter(
            *xyz_samples[iatom].T,
            c=weights_samples[iatom],
            s=1,
            edgecolor="none",
            marker=".",
            cmap="plasma",
        )
        ax.view_init(elev=30, azim=-60)
        cbar = plt.colorbar(sc, ax=ax, shrink=0.5, pad=0.1)
        if atom_labels is None:
            ax.set_title(f"atom {iatom}\n F={f}, m={m}, sym={sym}, ind={istate}")
        else:
            ax.set_title(
                f"atom {iatom}: {atom_labels[iatom]}\n F={f}, m={m}, sym={sym}, ind={istate}"
            )

    plt.savefig(filename, dpi=300)
    plt.close()


def plot_mc_spin_dens(
    hyper_states,
    states,
    f: float,
    sym: str,
    m: float,
    istate: int,
    atom_xyz: np.ndarray,
    atom_labels: list[str] | None = None,
    cart_comp: Literal["x", "y", "z"] = "z",
    filename: str = "dens",
    npoints: int = 1000000,
    npoints_1d: int = 30,
    coef_thresh: float = 1e-8,
):
    # print(
    #     f"Compute spin-density {cart_comp}-component for state F={f}, m={m}, sym={sym}, ind={istate}"
    # )

    alpha = np.linspace(0, 2 * np.pi, npoints_1d)
    beta = np.linspace(0, np.pi, npoints_1d)
    gamma = np.linspace(0, 2 * np.pi, npoints_1d)

    ix = ["x", "y", "z"].index(cart_comp)

    dens = hyper_states.rot_dens(
        states,
        f,
        sym,
        (m, istate),
        f,
        sym,
        (m, istate),
        spin_dens=True,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        coef_thresh=coef_thresh,
    )[:, :, :, ix, :]

    natoms = dens.shape[-1]

    # print(f"min and max spin density for each nucleus:")
    # for iatom in range(natoms):
    #     print(
    #         f"{iatom}: {np.round(np.min(dens[:,:,:,iatom]),3)}, {np.round(np.max(dens[:,:,:,iatom]),3)}"
    #     )

    xyz_samples = []
    weights_samples = []
    for iatom in range(natoms):
        fdens = RegularGridInterpolator((alpha, beta, gamma), dens[:, :, :, iatom])
        pts = np.random.uniform(
            low=[0, 0, 0], high=[2 * np.pi, np.pi, 2 * np.pi], size=(npoints, 3)
        )
        w = fdens(pts)
        signs = np.sign(w)
        abs_w = np.abs(w) / np.max(np.abs(dens))
        eta = np.random.uniform(0.0, 1.0, size=len(w))
        ind = np.where(abs_w > eta)
        points = pts[ind]
        signs = signs[ind]
        rot_mat = Rotation.from_euler("ZYZ", points).as_matrix()
        xyz = np.dot(rot_mat, atom_xyz[iatom] / np.linalg.norm(atom_xyz[iatom]))
        kernel = stats.gaussian_kde(xyz.T)
        weights = kernel(xyz.T)
        xyz_samples.append(xyz)
        weights_samples.append(np.real(weights * signs))

    # print("number of samples for each nucleus:")
    # for iatom in range(natoms):
    #     print(f"{iatom}: {len(xyz_samples[iatom])}")

    fig = plt.figure(figsize=(8, 5))
    axs = [
        fig.add_subplot(1, natoms, iatom + 1, projection="3d")
        for iatom in range(natoms)
    ]
    for ax in axs:
        ax.set_box_aspect(aspect=(1, 1, 1))
        ax.axes.set_xlim3d(left=-1.1, right=1.1)
        ax.axes.set_ylim3d(bottom=-1.1, top=1.1)
        ax.axes.set_zlim3d(bottom=-1.1, top=1.1)

    vmin = min(np.min([np.min(elem) for elem in weights_samples]), -1e-6)
    vmax = max(np.max([np.max(elem) for elem in weights_samples]), 1e-6)
    vmin = -max(np.abs(vmin), np.abs(vmax))
    vmax = max(np.abs(vmin), np.abs(vmax))

    for iatom, ax in enumerate(axs):
        sc = ax.scatter(
            *xyz_samples[iatom].T,
            c=weights_samples[iatom],
            s=1,
            edgecolor="none",
            marker=".",
            cmap="coolwarm",
            norm=mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax),
        )
        ax.view_init(elev=30, azim=-60)
        cbar = plt.colorbar(sc, ax=ax, shrink=0.5, pad=0.1)
        if atom_labels is None:
            ax.set_title(f"atom {iatom}\n F={f}, m={m}, sym={sym}, ind={istate}")
        else:
            ax.set_title(
                f"atom {iatom}: {atom_labels[iatom]}\n F={f}, m={m}, sym={sym}, ind={istate}"
            )

    plt.savefig(filename, dpi=300)
    plt.close()
