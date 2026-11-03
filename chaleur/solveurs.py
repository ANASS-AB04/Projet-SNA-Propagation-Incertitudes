"""Solveurs déterministes pour un matériau nu fixé : régime transitoire et équilibre."""
from scipy.integrate import solve_ivp
from scipy.sparse.linalg import spsolve


def transitoire(pb, nu, t, rtol=1e-6, atol=1e-9):
    """Trajectoire u(t; nu) aux instants t, par BDF (EDO raide) : tableau (len(t), N)."""
    A, b = pb.operateur(nu)
    sol = solve_ivp(lambda s, u: A @ u + b, (t[0], t[-1]), pb.u0, method="BDF",
                    t_eval=t, jac=A, rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y.T


def stationnaire(pb, nu):
    """Équilibre u*(nu), solution de A(nu) u + b(nu) = 0."""
    A, b = pb.operateur(nu)
    return spsolve(A, -b)
