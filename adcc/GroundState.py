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
import libadcc
import numpy as np

from .functions import direct_sum, einsum, evaluate
from .misc import cached_member_function, cached_property
from .MoSpaces import split_spaces
from .OneParticleOperator import OneParticleOperator, product_trace
from .ReferenceState import ReferenceState
from .timings import Timer, timed_member_call
from . import block as b


class GroundState:
    def __init__(self, hf):
        """
        Base class for ground states.
        """
        if isinstance(hf, libadcc.HartreeFockSolution_i):
            hf = ReferenceState(hf)
        if not isinstance(hf, ReferenceState):
            raise TypeError("hf needs to be a ReferenceState "
                            "or a HartreeFockSolution_i")
        self.reference_state = hf
        self.mospaces = hf.mospaces
        self.timer = Timer()
        self.has_core_occupied_space = hf.has_core_occupied_space

    def __getattr__(self, attr):
        # Shortcut some quantities, which are needed most often
        if attr.startswith("t2") and len(attr) == 4:  # t2oo, t2oc, t2cc
            xxvv = b.__getattr__(attr[2:4] + "vv")
            return self.t2(xxvv)
        else:
            raise AttributeError

    @cached_member_function
    def df(self, space: str):
        """Delta Fock matrix"""
        hf = self.reference_state
        s1, s2 = split_spaces(space)
        fC = hf.fock(s1 + s1).diagonal()
        fv = hf.fock(s2 + s2).diagonal()
        return direct_sum("-i+a->ia", fC, fv)

    @cached_member_function
    def t2eri(self, space: str, contraction):
        """
        Return the T2 tensor with ERI tensor contraction intermediates.
        These are called pi1 to pi7 in libadc.
        """
        hf = self.reference_state
        key = space + contraction
        expressions = {
            # space + contraction
            b.ooov + b.vv: ('ijbc,kabc->ijka', b.ovvv),
            b.ooov + b.ov: ('ilab,lkjb->ijka', b.ooov),
            b.oovv + b.oo: ('klab,ijkl->ijab', b.oooo),
            b.oovv + b.ov: ('jkac,kbic->ijab', b.ovov),
            b.oovv + b.vv: ('ijcd,abcd->ijab', b.vvvv),
            b.ovvv + b.oo: ('jkbc,jkia->iabc', b.ooov),
            b.ovvv + b.ov: ('ijbd,jcad->iabc', b.ovvv),
        }
        if key not in expressions:
            raise NotImplementedError("t2eri intermediate not implemented "
                                      f"for space '{space}' and contraction "
                                      f"'{contraction}'.")
        contraction_str, eri_block = expressions[key]
        return einsum(contraction_str, self.t2oo, hf.eri(eri_block))

    def density(self, level=2):
        """
        Return the ground state density in the MO basis with all corrections
        up to the specified order of perturbation theory
        """
        if level == 1:
            return self.reference_state.density
        elif level == 2:
            return self.reference_state.density + self.second_order_diffdm
        else:
            raise NotImplementedError("Only densities for level 1 and 2"
                                      " are implemented.")

    @property
    def mp2_density(self):  # Keep, remove or rename?
    #def density_through_2nd_order(self):
        return self.density(2)

    def dipole_moment(self, level=2):
        """
        Return the ground state dipole moment at the specified level of
        perturbation theory.
        """
        if level == 1:
            return self.reference_state.dipole_moment
        elif level == 2:
            return self.reference_state.dipole_moment + 
                                             self.second_order_dipole_moment
        else:
            raise NotImplementedError("Only dipole moments for level 1 and 2"
                                      " are implemented.")

    @cached_property
    def second_order_dipole_moment(self):
        """
        Return the 2nd-order correction to the ground state dipole moment.
        """
        refstate = self.reference_state
        dipole_integrals = refstate.operators.electric_dipole
        correction = -np.array([product_trace(comp, self.second_order_diffdm)
                                for comp in dipole_integrals])
        return correction

    def diffdm(self, level=2):
        """
        Return the n-th order contribution to the ground state difference density
        in the MO basis.
        """
        if level == 2:
            return self.second_order_diffdm
        else:
            raise NotImplementedError("Difference density only implemented for "
                                      "level 2.")

    @cached_property
    @timed_member_call(timer="timer")
    def second_order_diffdm(self):
        """
        Return the 2nd order ground state density contribution in the MO basis.
        """
        ret = OneParticleOperator(self.mospaces, is_symmetric=True)
        ret.oo = -0.5 * einsum("ikab,jkab->ij", self.t2oo, self.t2oo)
        ret.ov = self.ts2(b.ov)
        ret.vv = 0.5 * einsum("ijac,ijbc->ab", self.t2oo, self.t2oo)

        if self.has_core_occupied_space:
            hf = self.reference_state
            # additional terms to "revert" CVS for ground state density
            ret.oo += -0.5 * einsum("iLab,jLab->ij", self.t2oc, self.t2oc)
            ret.ov += -0.5 * (
                + einsum("jMib,jMab->ia", hf.ocov, self.t2oc)
                + einsum("iLbc,Labc->ia", self.t2oc, hf.cvvv)
                + einsum("kLib,kLab->ia", hf.ocov, self.t2oc)
                + einsum("iMLb,LMab->ia", hf.occv, self.t2cc)
                - einsum("iLMb,LMab->ia", hf.occv, self.t2cc)
            ) / self.df(b.ov)
            ret.vv += (
                + 0.5 * einsum("IJac,IJbc->ab", self.t2cc, self.t2cc)
                + 1.0 * einsum("kJac,kJbc->ab", self.t2oc, self.t2oc)
            )
            # compute extra CVS blocks
            ret.cc = -0.5 * (
                + einsum("kIab,kJab->IJ", self.t2oc, self.t2oc)
                + einsum('LIab,LJab->IJ', self.t2cc, self.t2cc)
            )
            ret.co = -0.5 * (
                + einsum("kIab,kjab->Ij", self.t2oc, self.t2oo)
                + einsum("ILab,jLab->Ij", self.t2cc, self.t2oc)
            )
            ret.cv = -0.5 * (
                - einsum("jIbc,jabc->Ia", self.t2oc, hf.ovvv)
                + einsum("jkIb,jkab->Ia", hf.oocv, self.t2oo)
                + einsum("jMIb,jMab->Ia", hf.occv, self.t2oc)
                + einsum("ILbc,Labc->Ia", self.t2cc, hf.cvvv)
                + einsum("kLIb,kLab->Ia", hf.occv, self.t2oc)
                + einsum("LMIb,LMab->Ia", hf.cccv, self.t2cc)
            ) / self.df(b.cv)
        ret.reference_state = self.reference_state
        return evaluate(ret)

    def _to_qcvars(self, properties=False, recurse=False, maxlevel=2, method="MP"):
        """
        Return a dictionary with property keys compatible to a Psi4 wavefunction
        or a QCEngine Atomicresults object.
        """
        qcvars = {}
        for level in range(2, maxlevel + 1):
            try:
                mpcorr = self.energy_correction(level)
                qcvars[f"{method}{level} CORRELATION ENERGY"] = mpcorr
                qcvars[f"{method}{level} TOTAL ENERGY"] = self.energy(level)
            except NotImplementedError:
                pass
            except ValueError:
                pass

        if properties:
            for level in range(2, maxlevel + 1):
                try:
                    qcvars[f"{method}{level} DIPOLE"] = self.dipole_moment(level)
                except NotImplementedError:
                    pass

        if recurse:
            qcvars.update(self.reference_state.to_qcvars(properties, recurse))
        return qcvars

    @cached_property
    @timed_member_call(timer="timer")
    def first_order_dm_correction_2p(self) -> TwoParticleDensity:
        """
        Return the 1st-order correction to the two-particle difference density 
        in the MO basis.
        """
        ret = TwoParticleDensity(self.mospaces,
                                 symmetry=OperatorSymmetry.HERMITIAN)
        ret.oovv = -1.0 * self.t2oo
        return ret.evaluate()

    @cached_property
    @timed_member_call(timer="timer")
    def second_order_dm_correction_2p(self) -> TwoParticleDensity:
        """
        Return the 2nd-order correction to the two-particle difference density 
        in the MO basis.
        """
        hf: ReferenceState = self.reference_state
        ret = TwoParticleDensity(self.mospaces,
                                 symmetry=OperatorSymmetry.HERMITIAN)
        p0: OneParticleDensity = self.second_order_diffdm

        # constuct Kronecker Delta
        d_oo = zeros_like(hf.foo)
        d_oo.set_mask("ii", 1)

        ret.oooo = (
            + 4.0 * einsum("ik,jl->ijkl", p0.oo, d_oo)
            .antisymmetrise(0, 1).antisymmetrise(2, 3)
            + 0.5 * einsum("ijab,klab->ijkl", self.t2oo, self.t2oo)
        )
        ret.ooov = (
            + 2.0 * einsum("ja,ik->ijka", p0.ov, d_oo).antisymmetrise(0, 1)
        )
        ret.oovv = (
            - 1.0 * self.td2(b.oovv)
        )
        ret.ovov = (
            + 1.0 * einsum("ab,ij->iajb", p0.vv, d_oo)
            - 1.0 * einsum("jkac,ikbc->iajb", self.t2oo, self.t2oo)
        )
        ret.vvvv = (
            + 0.5 * einsum("ijab,ijcd->abcd", self.t2oo, self.t2oo)
        )
        return evaluate(ret)

    def diffdm_2p(self, level=2) -> TwoParticleDensity:
        """
        Return the two-particle difference density in the MO basis with all
        corrections up to the specified order of perturbation theory.
        """
        if level == 1:
            return self.first_order_dm_correction_2p
        elif level == 2:
            return (self.mp1_dm_correction_2p
                    + self.second_order_dm_correction_2p)
        else:
            raise NotImplementedError("Only first and second-order two-particle "
                                      "density corrections are implemented.")

    def density_2p(self, level=2) -> TwoParticleDensity:
        """
        Return the two-particle density in the MO basis with all corrections
        up to the specified order of perturbation theory.
        """
        if level == 0:
            return self.reference_state.density_2p
        diffdm = self.diffdm_2p(level)
        return self.reference_state.density_2p + diffdm

    @cached_member_function()
    def ssq(self, level=2):
        """
        Return <S^2> of the ground state.
        """
        if self.reference_state.restricted:
            raise NotImplementedError(
                "<S^2> is not implemented for restricted HF references."
            )
        ssq_1p_op = self.reference_state.operators.ssq_1p
        ssq_2p_op = self.reference_state.operators.ssq_2p
        # the trace of the second-order (and higher) correction to the RDM1
        # is zero -> no influence on top of HF density for ground state
        ssq_1p = product_trace(ssq_1p_op, self.density(0))
        ssq_2p = product_trace(ssq_2p_op, self.density_2p(level))
        return (ssq_1p + ssq_2p)

# The following functions are defined to avoid TypeChecking issues.
# The quantities they are intended to return depend on the specific
# definition of the partitioning scheme H = H_0 + H_1 
# As such, they are only calculated by child classes LazyMp and LazyRe.

    @cached_member_function
    def t2(self, space: str):
        """T2 amplitudes"""
        raise NotImplementedError("1st-order doubles amplitudes ",
                                  "not implemented for the GroundState ",
                                  "base class.")

    @cached_member_function
    def td2(self, space: str):
        """Return the T^D_2 term"""
        raise NotImplementedError("2nd-order doubles amplitudes ",
                                  "not implemented for the GroundState ",
                                  "base class.")

    @cached_member_function
    def tt2(self, space: str):
        """
        Return the second order MP triples amplitudes for the given space
        (e.g. o1o1o1v1v1v1).
        """
        raise NotImplementedError("2nd-order triples amplitudes ",
                                  "not implemented for the GroundState ",
                                  "base class.")

    @cached_member_function()
    def energy_correction(self, level=2):
        """Obtain the energy correction at a particular level"""
        raise NotImplementedError("Energy corrections ",
                                  "not defined for the GroundState ",
                                  "base class.")

    def energy(self, level=2):
        """
        Obtain the total energy (SCF energy plus all corrections)
        at a particular level of perturbation theory.
        """
        raise NotImplementedError("Total energy ",
                                  "not defined for the GroundState ",
                                  "base class.")
