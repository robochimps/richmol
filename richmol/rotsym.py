import numpy as np

from .symtop import symtop_on_grid_split_angles, wang_coefs


class R0:
    """Represent E-symmetry equivalent rotation (i.e., no rotation)"""

    def act_on_jk(self, j: int, k: int):
        return 1, j, k

    def act_on_euler(self, alpha, beta, gamma):
        return alpha, beta, gamma

    def __str__(self):
        return f"R0"


class RalphaPi:
    """Represent symmetry equivalent rotation by an angle Pi
    about an axis in xy plane that forms an angle `alpha`
    with the x axis.
    """

    def __init__(self, alpha: float):
        self.alpha = alpha

    def act_on_jk(self, j: int, k: int):
        return (-1) ** j * np.exp(-2j * k * self.alpha), j, -k

    def act_on_euler(self, alpha, beta, gamma):
        return alpha + np.pi, np.pi - beta, 2 * np.pi - 2 * self.alpha - gamma

    def __str__(self):
        alpha_deg = self.alpha * 180 / np.pi
        return f"R_{round(alpha_deg, 1)}^π"


class RzBeta:
    """Represent symmetry equivalent rotation by an angle `beta`
    about the z axis.
    """

    def __init__(self, beta: float):
        self.beta = beta

    def act_on_jk(self, j: int, k: int):
        return np.exp(-1j * k * self.beta), j, k

    def act_on_euler(self, alpha, beta, gamma):
        return alpha, beta, gamma + self.beta

    def __str__(self):
        beta_deg = self.beta * 180 / np.pi
        return f"R_z^{round(beta_deg, 1)}"


def wang_symmetry_by_sampling(
    j: int,
    linear: bool,
    rotations: list[RalphaPi | RzBeta | R0],
    irreps: dict[str, list[int]],
    npoints: int = 20,
    tol: float = 1e-6,
) -> dict[list[int], str]:
    """Numerically determines the symmetry of rotational wavefunctions in the Wang
    basis representation by evaluating their transformation properties under
    a set of symmetry-equivalent rotation operations.

    Args:
        j (int):
            Rotational angular momentum quantum number (J).
        linear (bool):
            Set to True if the molecule is linear, False otherwise.
        rotations (list[RalphaPi | RzBeta | R0]):
            List of symmetry-equivalent rotations to apply.
        irreps (dict[str, list[int]]):
            Dictionary mapping irreducible representation labels to their
            character values, in the same order as the `rotations` list.
        npoints (int, optional):
            Number of random points per Euler angle for numerical sampling.
            Default is 20.
        tol (float, optional):
            Numerical tolerance for interpreting characters as 1 or -1.
            Default is 1e-6.

    Returns:
        dict[list[int], str]:
            A dictionary mapping rotational quantum number tuples (J, k, tau)
            to symmetry labels, as determined from the `irreps` dictionary.
    """
    # check if E-symmetry equivalent rotation is present
    if not any(isinstance(rot, R0) for rot in rotations):
        raise ValueError(
            "List of symmetry equivalent rotations 'rotations' must contain 'R0', "
            + "which is equivalent to E"
        )

    # check if number of rotations and number of characters match
    if any(len(rotations) != len(char) for char in irreps.values()):
        raise ValueError(
            "Number characters for each symmetry in 'irreps' dictionary must be equal "
            + "to the number of symmetry equivalent rotations in 'rotations' list"
        )

    # check if there are duplicate irreps
    seen = set()
    for sym, char in irreps.items():
        char0 = tuple(char)
        if char0 in seen:
            raise ValueError(
                f"Found at least two symmetries with the same list of characters in 'irreps'\n"
                + "\n".join(f"{sym}: {char}" for sym, char in irreps.items())
            )
        seen.add(char0)

    # default behaviour
    if len(rotations) == 1 and isinstance(rotations[0], R0):
        k_list, jktau_list, _ = wang_coefs(j, linear)
        sym = list(irreps.keys())[0]
        symmetry_table = {jktau: sym for jktau in jktau_list}
        return symmetry_table

    # random Euler angles for numerical identification of symmetry
    def strat_uniform(a, b, n):
        bins = np.linspace(a, b, n + 1)
        return bins[:-1] + np.random.uniform(0, (b - a) / n, n)

    alpha0 = strat_uniform(0, 2 * np.pi, npoints)
    beta0 = strat_uniform(0, np.pi, npoints)
    gamma0 = strat_uniform(0, 2 * np.pi, npoints)

    # reference Wang functions on grid of Euler angles
    rot_k0, rot_m0, rot_l0, k_list, jktau_list = symtop_on_grid_split_angles(
        j, alpha0, beta0, gamma0, linear
    )
    psi0 = np.einsum(
        "klg,lb,la->kgba", rot_k0, rot_l0, np.mean(rot_m0, axis=0), optimize="optimal"
    )

    # determine effect of symmetry equivalent rotation in `sym_rot` on Wang functions

    character_table = {}

    for rot in rotations:
        # rotate Euler angles
        alpha, beta, gamma = rot.act_on_euler(alpha0, beta0, gamma0)

        # Wang functions on grid of Euler angles
        rot_k, rot_m, rot_l, _, _ = symtop_on_grid_split_angles(
            j, alpha, beta, gamma, linear
        )
        psi = np.einsum(
            "klg,lb,la->kgba", rot_k, rot_l, np.mean(rot_m, axis=0), optimize="optimal"
        )

        # determine effect of rotation on each Wang function
        for jktau, char in zip(
            jktau_list, np.mean(np.real(psi / psi0), axis=(1, 2, 3))
        ):
            if abs(char - 1) < tol:
                p = 1
            elif abs(char + 1) < tol:
                p = -1
            else:
                raise ValueError(
                    f"Symmetry equivalent rotation {rot} does not transform |J,k,τ⟩ to ±|J,k,τ⟩ "
                    + f"for |J,k,τ⟩ = |{jktau}⟩,\nmean(psi / psi0) = {char}, "
                    + f"tol = {tol}"
                )
            try:
                character_table[jktau].append(p)
            except KeyError:
                character_table[jktau] = [p]

    # assign symmetry labels to Wang functions

    symmetry_table = {}
    for jktau, char in character_table.items():
        jktau_sym = [sym for sym, char0 in irreps.items() if char0 == char]
        if not jktau_sym:
            raise ValueError(
                f"Can't assign characters {char} of |J,k,τ⟩ = |{jktau}⟩ function "
                + f"to any symmetry in 'irreps' table\n"
                + "\n".join(f"{sym}: {char}" for sym, char in irreps.items())
            )
        symmetry_table[jktau] = jktau_sym[0]

    return symmetry_table
