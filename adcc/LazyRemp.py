#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2019 by the adcc authors
##
## This file is part of adcc.
##
## adcc is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published
## by the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## adcc is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with adcc. If not, see <http://www.gnu.org/licenses/>.
##
## ---------------------------------------------------------------------
from .AdcMatrix import AdcMatrixlike
from .AmplitudeVector import AmplitudeVector
from .functions import direct_sum, einsum
from .GroundState import GroundState
from .ReferenceState import ReferenceState
from .misc import cached_member_function
from . import block as b

import libadcc

import numpy as np
from typing import Union


class LazyRemp(GroundState):
    """
    Hybrid REMP ground state class.
    In the REMP pertubation theoretical approach both parts of the Hamiltonian
    are defined combinations of the corresponding ones for the parent RE and MP
    partitioning schemes.

    {H^{REMP}_{0}} = A {H^{MP}_{0}} + (1-A) {H^{RE}_{0}}

    {H^{REMP}_{I}} = {H} - {H^{REMP}_{0}}
                   = A {H^{MP}_{I}} + (1-A) {H^{RE}_{I}}

    The scalar parameter A (0 <= A <= 1) dictates the degree of mixing between
    the two parent schemes.
    Special case: A = 0 -> standard RE scheme.
    Special case: A = 1 -> standard MP scheme.

    Parameters
    ----------
    hf : ReferenceState
        The SCF reference state.
    remp_A : float, optional
        The parameter defining the mixing of RE and MP schemes.
    conv_tol : float, optional
        Convergence tolerance for the RE ground state amplitudes
        (default: SCF tolerance).
    max_iter : int, optional
        Maximum number of iterations for the iterative determination of the
        REMP ground state amplitudes (default: 100).
    """

    def __init__(self, hf: Union[ReferenceState, libadcc.HartreeFockSolution_i],
                 remp_A: float = None,
                 conv_tol: float = None, max_iter: int = None):
        if remp_A is None :
            remp_A = 0.20
        elif not isinstance(remp_A, float):
            raise TypeError(f"Parameter remp_A must be a float, got {remp_A} "
                            f"of type {type(remp_A)}.")
        else :
            if not (0.0 <= remp_A <= 1.0):
                raise ValueError(f"Parameter remp_A must be a float in [0,1]"
                                 f", got {remp_A}.")
        self.remp_A = remp_A
        if conv_tol is None:
            conv_tol = hf.conv_tol
        self.conv_tol = conv_tol
        if max_iter is None:
            max_iter = 100
        self.max_iter = max_iter
        super().__init__(hf)

    @cached_member_function()
    def ts1(self, space: str) -> libadcc.Tensor:
        """
        First order REMP ground state singles amplitudes.
        Zero for a block diagonal Fock matrix.
        """
        raise NotImplementedError("The first order REMP singles amplitudes vanish "
                                  "for a block diagonal fock matrix. Probably you "
                                  "don't need this tensor.")

    @cached_member_function()
    def td1(self, space: str) -> libadcc.Tensor:
        """
        1st-order REMP ground state doubles amplitudes.
        """
        from .solver.conjugate_gradient import conjugate_gradient, default_print
        from .solver.preconditioner import JacobiPreconditioner
        from .LazyMp import LazyMp

        if space != b.oovv:
            raise NotImplementedError("1st-order REMP doubles"
                                      f"not implemented for space {space}.")
        hf = self.reference_state
        remp_A = self.remp_A

        # build the right hand side of Ax = b
        rhs = -hf.oovv
        rhs = AmplitudeVector(pphh=rhs)

        # build a guess for the t-amplitudes: use MP amplitudes as they only
        # scale N^4, while each iteration scales as N^6
        guess = LazyMp(self.reference_state).t2(space)
        guess = AmplitudeVector(pphh=guess)

        print("\nIterating 1st-order REMP doubles amplitudes...")
        t2 = conjugate_gradient(Doubles(hf, remp_A), rhs, guess,
                                callback=default_print,
                                explicit_symmetrisation=None,
                                conv_tol=self.conv_tol,
                                max_iter=self.max_iter,
                                Pinv=JacobiPreconditioner)
        t2 = t2.solution.pphh
        return t2

    @cached_member_function()
    def ts2(self, space: str, apply_cvs: bool = False) -> libadcc.Tensor:
        """
        2nd-order REMP ground state singles amplitudes.
        """
        from .solver.conjugate_gradient import conjugate_gradient, default_print
        from .solver.preconditioner import JacobiPreconditioner
        from .LazyMp import LazyMp

        if apply_cvs:
            raise NotImplementedError("CVS-REMP not implemented.")
        if space != b.ov:
            raise NotImplementedError("2nd-order singles not implemented for "
                                      f"space {space}.")
        hf = self.reference_state
        remp_A = self.remp_A
        t2_1 = self.t2(b.oovv)

        # build the right hand side of Ax = b
        # Note that the Fock matrix was assumed to be block diagonal.
        rhs = (
            # N^5: O^2V^3 / N^4: O^1V^3
            - 0.5 * einsum('jabc,ijbc->ia', hf.ovvv, t2_1)
            # N^5: O^3V^2 / N^4: O^2V^2
            - 0.5 * einsum('jkib,jkab->ia', hf.ooov, t2_1)
        )
        rhs = AmplitudeVector(ph=rhs)

        # can use MP amplitudes as guess, since the rhs scales N^5 anyway
        guess = LazyMp(self.reference_state).ts2(space)
        guess = AmplitudeVector(ph=guess)

        print("\nIterating 2nd-order REMP singles amplitudes...")
        t1 = conjugate_gradient(Singles(hf, remp_A), rhs, guess,
                                callback=default_print,
                                explicit_symmetrisation=None,
                                conv_tol=self.conv_tol,
                                max_iter=self.max_iter,
                                Pinv=JacobiPreconditioner)
        t1 = t1.solution.ph
        return t1

    @cached_member_function()
    def td2(self, space: str) -> libadcc.Tensor:
        """
        Second order REMP ground state doubles amplitudes.
        """
        from .solver.conjugate_gradient import conjugate_gradient, default_print
        from .solver.preconditioner import JacobiPreconditioner
        from .LazyMp import LazyMp

        if space != b.oovv:
            raise NotImplementedError("2nd-order doubles not implemented for "
                                      f"space {space}.")
        hf = self.reference_state
        remp_A = self.remp_A
        t2_1 = self.t2(b.oovv)

        # build the right hand side of Ax = b
        # Note that the Fock matrix was assumed to be block diagonal.
        rhs = (
            # N^6: O^2V^4 / N^4: O^2V^2
            + 0.5 * remp_A * einsum('abcd,ijcd->ijab', hf.vvvv, t2_1)
            # N^6: O^4V^2 / N^4: O^2V^2
            + 0.5 * remp_A * einsum('ijkl,klab->ijab', hf.oooo, t2_1)
            # N^6: O^3V^3 / N^4: O^2V^2
            + 4 * remp_A * einsum('jcka,ikbc->ijab', hf.ovov, t2_1
                                  ).antisymmetrise(0, 1).antisymmetrise(2, 3)
        )
        rhs = AmplitudeVector(pphh=rhs)

        # build a guess for the t-amplitudes: use MP amplitudes as they only
        # scale N^4, while each iteration scales as N^6
        guess = LazyMp(self.reference_state).t2(space)
        guess = AmplitudeVector(pphh=guess)

        print("\nIterating 2nd-order REMP doubles amplitudes...")
        t2 = conjugate_gradient(Doubles(hf, remp_A), rhs, guess,
                                callback=default_print,
                                explicit_symmetrisation=None,
                                conv_tol=self.conv_tol,
                                max_iter=self.max_iter,
                                Pinv=JacobiPreconditioner)
        t2 = t2.solution.pphh
        return t2

    @cached_member_function()
    def energy_correction(self, level: int = 2) -> float:
        """
        Obtain the REMP energy correction at a particular level.
        """
        assert level >= 0
        if level == 0:
            return 0.0
        hf = self.reference_state
        if level == 1:
            remp_A = self.remp_A
            return -0.5 * remp_A * np.einsum("ijij->", hf.oooo.to_ndarray())
        if level == 2:
            terms = [(1.0, hf.oovv, self.t2oo)]
        elif level == 3:
            terms = [(1.0, hf.oovv, self.td2(b.oovv))]
        else:
            raise NotImplementedError(f"REMP({level}) energy correction "
                                      "not implemented.")
        return sum(
            -0.25 * pref * eri.dot(t2)
            for pref, eri, t2 in terms
        )

    def energy(self, level: int = 2) -> float:
        """
        Obtain the total REMP energy (SCF energy plus all corrections)
        consistent through a particular level of perturbation theory.
        """
        assert level >= 0
        if level == 0:
            remp_A = self.remp_A
            return (self.reference_state.energy_scf +
                    + 0.5 * remp_A * np.einsum("ijij->", hf.oooo.to_ndarray())
            )

        # Accumulator for all energy terms
        energies = [self.reference_state.energy_scf]

        for il in range(2, level + 1):
            energies.append(self.energy_correction(il))
        return sum(energies)


    def to_qcvars(self, properties: bool = False,
                  recurse: bool = False, maxlevel: int = 2) -> dict:
        """
        Return a dictionary with property keys compatible to a Psi4 wavefunction
        or a QCEngine Atomicresults object.
        """
        return self._to_qcvars(
                gs_type="REMP", properties=properties, recurse=recurse,
                maxlevel=maxlevel
        )

    @property
    def remp2_diffdm(self):
        """
        Return the REMP2 difference density in the MO basis.
        """
        return self.diffdm(2)

    @property
    def remp2_density(self):
        """
        Return the REMP2 ground state density in the MO basis.
        """
        return self.density(2)

    @property
    def remp2_dipole_moment(self):
        """
        Return the REMP2 ground state dipole moment.
        """
        return self.dipole_moment(2)

    @cached_member_function()
    def third_order_dm_correction_oo(self, apply_cvs: bool = False
                                     ):
        """
        Return the third-order contribution to the ground state
        difference density in the MO basis.
        """
        if self.has_core_occupied_space:
            raise NotImplementedError(
                "CVS-REMP3 occ-occ block of difference density "
                "not implemented yet"
            )
        assert not apply_cvs

        return (- einsum("ikab,jkab->ij", self.t2oo,
                         self.td2(b.oovv)).symmetrise(0, 1)
        ).evaluate()

    @cached_member_function()
    def third_order_dm_correction_vv(self, apply_cvs: bool = False
                                     ):
        """
        Return the third-order contribution to the ground state
        difference density in the MO basis.
        """
        if self.has_core_occupied_space:
            raise NotImplementedError(
                "CVS-REMP3 virt-virt block of difference density "
                "not implemented yet"
            )
        assert not apply_cvs

        return (einsum("ijac,ijbc->ab", self.t2oo,
                       self.td2(b.oovv)).symmetrise(0, 1)
        ).evaluate()


class RempAmplitude(AdcMatrixlike):
    # k fold excited n'th order REMP amplitudes are defined according to:
    #   <k|H0|n> - E0 tn_k = - <k|H1|n-1> + sum_{m=1}^{n-1} Em t(n-m)_k
    # The structure of the left hand side is for all orders n the same!
    # Only the right hand side (the inhomogenity) varies with order n.
    def __init__(self, hf, remp_A):
        self.reference_state = hf
        self.remp_A = remp_A

    def __matmul__(self, vec):
        raise NotImplementedError(f"MVP not implemented for {self.__class__}")

    def diagonal(self):
        raise NotImplementedError(f"Diagonal not implemented for {self.__class__}")


class Singles(RempAmplitude):
    def __matmul__(self, vec):
        if isinstance(vec, list):
            return [self.__matmul__(v) for v in vec]
        hf = self.reference_state
        remp_A = self.remp_A
        t1 = (einsum('ab,ib->ia', hf.fvv, vec.ph)  # N^3
              - einsum('ij,ja->ia', hf.foo, vec.ph)  # N^3
              - (1-remp_A) * einsum('ibja,jb->ia', hf.ovov, vec.ph))  # N^4
        return AmplitudeVector(ph=t1)

    def diagonal(self): # TO BE DONE
        hf = self.reference_state
        diag = direct_sum('-i+a->ia', hf.foo.diagonal(), hf.fvv.diagonal())
        return AmplitudeVector(ph=diag.evaluate())


class Doubles(RempAmplitude):
    def __matmul__(self, vec):
        if isinstance(vec, list):
            return [self.__matmul__(v) for v in vec]
        hf = self.reference_state
        remp_A = self.remp_A
        t2 = (
            4 * (1-remp_A) * einsum('icka,jkbc->ijab', hf.ovov, vec.pphh
                                   ).antisymmetrise(0, 1).antisymmetrise(2, 3)
            + 2 * einsum('ac,ijbc->ijab',
                         hf.fvv, vec.pphh).antisymmetrise(2, 3)
            + 2 * einsum('jk,ikab->ijab',
                         hf.foo, vec.pphh).antisymmetrise(0, 1)
            - 0.5 * (1-remp_A) * einsum('abcd,ijcd->ijab', hf.vvvv, vec.pphh)
            - 0.5 * (1-remp_A) * einsum('ijkl,klab->ijab', hf.oooo, vec.pphh)
        )
        return AmplitudeVector(pphh=t2)

    def diagonal(self): # TO BE DONE
        hf = self.reference_state
        # NOTE: only terms containing the Fock matrix have been considered.
        # For a canonical orbital basis, the diagonal is defined by the
        # usual orbital energy difference.
        diag = direct_sum("+i+j-a-b->ijab",
                          hf.foo.diagonal(), hf.foo.diagonal(),
                          hf.fvv.diagonal(), hf.fvv.diagonal()).symmetrise(2, 3)
        return AmplitudeVector(pphh=diag.evaluate())
