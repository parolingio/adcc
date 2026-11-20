## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2020 by the adcc authors
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
from collections import namedtuple
from math import sqrt

from adcc import block as b
from adcc.AmplitudeVector import AmplitudeVector
from adcc.functions import direct_sum, einsum, zeros_like
from adcc.Intermediates import Intermediates, register_as_intermediate

__all__ = ["block"]

# TODO One thing one could still do to improve timings is implement a "fast einsum"
#      that does not call opt_einsum, but directly dispatches to libadcc. This could
#      lower the call overhead in the applies for the cases where we have only a
#      trivial einsum to do. For the moment I'm not convinced that is worth the
#      effort ... I suppose it only makes a difference for the cheaper ADC variants
#      (ADC(0), ADC(1), CVS-ADC(0-2)-x), but then on the other hand they are not
#      really so much our focus.


#
# Dispatch routine
#
"""
`apply` is a function mapping an AmplitudeVector to the contribution of this
block to the result of applying the ADC matrix. `diagonal` is an `AmplitudeVector`
containing the expression to the diagonal of the ADC matrix from this block.
"""
AdcBlock = namedtuple("AdcBlock", ["apply", "diagonal"])


def block(ground_state, spaces, order, variant=None, intermediates=None):
    """
    Gets ground state, potentially intermediates, spaces (ph, pphh and so on)
    and the perturbation theory order for the block,
    variant is "cvs" or sth like that.

    It is assumed largely, that CVS is equivalent to mp.has_core_occupied_space,
    while one would probably want in the long run that one can have an "o2" space,
    but not do CVS.
    """
    if isinstance(variant, str):
        variant = [variant]
    elif variant is None:
        variant = []
    reference_state = ground_state.reference_state
    if intermediates is None:
        intermediates = Intermediates(ground_state)

    if ground_state.has_core_occupied_space and "cvs" not in variant:
        raise ValueError(
            "Cannot run a general (non-core-valence approximated) "
            "ADC method on top of a ground state with a "
            "core-valence separation."
        )
    if not ground_state.has_core_occupied_space and "cvs" in variant:
        raise ValueError(
            "Cannot run a core-valence approximated ADC method on "
            "top of a ground state without a "
            "core-valence separation."
        )

    fn = "_".join(["block"] + variant + spaces + [str(order)])

    if fn not in globals():
        raise ValueError(
            "Could not dispatch: "
            f"spaces={spaces} order={order} variant={variant}. "
            "Probably the secular matrix is not implemented for "
            "the requested method."
        )
    return globals()[fn](reference_state, ground_state, intermediates)


#
# 0th order main
#
def block_ph_ph_0(hf, mp, intermediates):
    fCC = hf.fcc if hf.has_core_occupied_space else hf.foo
    diagonal = AmplitudeVector(ph=direct_sum("a-i->ia", hf.fvv.diagonal(), fCC.diagonal()))

    def apply(ampl):
        return AmplitudeVector(ph=(
            + einsum("ib,ab->ia", ampl.ph, hf.fvv)
            - einsum("IJ,Ja->Ia", fCC, ampl.ph)
        ))  # fmt: skip

    return AdcBlock(apply, diagonal)


block_cvs_ph_ph_0 = block_ph_ph_0


def diagonal_pphh_pphh_0(hf):
    # since we use orbital energies to construct the diagonal,
    # we have to symmetrise both occupied and virtual indices!
    # (for dfs a single symmetrisation would be fine)
    # -> distinct treatment of CVS-ADC
    # If the occupied indices are not symmetrised the preconditioned
    # residuals do no longer have the correct symmetry in the davidson:
    # the antisymmetry of the occupied indices is missing
    fCC = hf.fcc if hf.has_core_occupied_space else hf.foo
    res = direct_sum(
        "-i-J+a+b->iJab",
        hf.foo.diagonal(), fCC.diagonal(), hf.fvv.diagonal(), hf.fvv.diagonal()
    ).symmetrise(2, 3)  # fmt: skip
    if not hf.has_core_occupied_space:
        res = res.symmetrise(0, 1)
    return AmplitudeVector(pphh=res)


def block_pphh_pphh_0(hf, mp, intermediates):
    def apply(ampl):
        return AmplitudeVector(pphh=(
            + 2 * einsum("ijac,bc->ijab", ampl.pphh, hf.fvv).antisymmetrise(2, 3)
            - 2 * einsum("ik,kjab->ijab", hf.foo, ampl.pphh).antisymmetrise(0, 1)
        ))  # fmt: skip

    return AdcBlock(apply, diagonal_pphh_pphh_0(hf))


def block_cvs_pphh_pphh_0(hf, mp, intermediates):
    def apply(ampl):
        return AmplitudeVector(pphh=(
            + 2 * einsum("iJac,bc->iJab", ampl.pphh, hf.fvv).antisymmetrise(2, 3)
            - einsum("ik,kJab->iJab", hf.foo, ampl.pphh)
            - einsum("JK,iKab->iJab", hf.fcc, ampl.pphh)
        ))  # fmt: skip

    return AdcBlock(apply, diagonal_pphh_pphh_0(hf))


def diagonal_ppphhh_ppphhh_0(intermediates):
    return AmplitudeVector(ppphhh=intermediates.df3)


def block_ppphhh_ppphhh_0(hf, mp, intermediates):
    def apply(ampl):
        ur3 = ampl.ppphhh
        diag = intermediates.df3
        return AmplitudeVector(ppphhh=diag * ur3)

    return AdcBlock(apply, diagonal_ppphhh_ppphhh_0(intermediates))


#
# 0th order coupling
#
def block_ph_pphh_0(hf, mp, intermediates):
    return AdcBlock(lambda ampl: 0, 0)


def block_pphh_ph_0(hf, mp, intermediates):
    return AdcBlock(lambda ampl: 0, 0)


block_cvs_ph_pphh_0 = block_ph_pphh_0
block_cvs_pphh_ph_0 = block_pphh_ph_0


def block_ph_ppphhh_0(hf, mp, intermediates):
    return AdcBlock(lambda ampl: 0, 0)


def block_ppphhh_ph_0(hf, mp, intermediates):
    return AdcBlock(lambda ampl: 0, 0)


def block_pphh_ppphhh_0(hf, mp, intermediates):
    return AdcBlock(lambda ampl: 0, 0)


def block_ppphhh_pphh_0(hf, mp, intermediates):
    return AdcBlock(lambda ampl: 0, 0)


#
# 1st order main
#
def block_ph_ph_1(hf, mp, intermediates):
    fCC = hf.fcc if hf.has_core_occupied_space else hf.foo
    CvCv = hf.cvcv if hf.has_core_occupied_space else hf.ovov
    diagonal = AmplitudeVector(ph=(
        + direct_sum("a-i->ia", hf.fvv.diagonal(), fCC.diagonal())  # order 0
        - einsum("IaIa->Ia", CvCv)  # order 1
    ))  # fmt: skip

    def apply(ampl):
        return AmplitudeVector(ph=(                 # PT order
            + einsum("ib,ab->ia", ampl.ph, hf.fvv)  # 0
            - einsum("IJ,Ja->Ia", fCC, ampl.ph)     # 0
            - einsum("JaIb,Jb->Ia", CvCv, ampl.ph)  # 1
        ))  # fmt: skip

    return AdcBlock(apply, diagonal)


block_cvs_ph_ph_1 = block_ph_ph_1


def diagonal_pphh_pphh_1(hf):
    # Fock matrix and ovov diagonal term (sometimes called "intermediate diagonal")
    dinterm_ov = (
        direct_sum("a-i->ia", hf.fvv.diagonal(), hf.foo.diagonal())
        - 2.0 * einsum("iaia->ia", hf.ovov)
    ).evaluate()

    if hf.has_core_occupied_space:
        dinterm_Cv = (
            direct_sum("a-I->Ia", hf.fvv.diagonal(), hf.fcc.diagonal())
            - 2.0 * einsum("IaIa->Ia", hf.cvcv)
        ).evaluate()
        diag_oC = einsum("iJiJ->iJ", hf.ococ)
    else:
        dinterm_Cv = dinterm_ov
        diag_oC = einsum("ijij->ij", hf.oooo).symmetrise()

    diag_vv = einsum("abab->ab", hf.vvvv).symmetrise()
    return AmplitudeVector(pphh=(
        # Both objects are 'df like' and thus it is sufficient to either symmetrise
        # the occupied or virtual indices
        + direct_sum("ia+Jb->iJab", dinterm_ov, dinterm_Cv).symmetrise(2, 3)
        + direct_sum("iJ+ab->iJab", diag_oC, diag_vv)
    ))  # fmt: skip


def block_pphh_pphh_1(hf, mp, intermediates):
    def apply(ampl):
        return AmplitudeVector(pphh=(  # 0th order
            + 2 * einsum("ijac,bc->ijab", ampl.pphh, hf.fvv).antisymmetrise(2, 3)
            - 2 * einsum("ik,kjab->ijab", hf.foo, ampl.pphh).antisymmetrise(0, 1)
            # 1st order
            + (
                -4 * einsum("ikac,kbjc->ijab", ampl.pphh, hf.ovov)
            ).antisymmetrise(0, 1).antisymmetrise(2, 3)
            + 0.5 * einsum("ijkl,klab->ijab", hf.oooo, ampl.pphh)
            + 0.5 * einsum("ijcd,abcd->ijab", ampl.pphh, hf.vvvv)
        ))  # fmt: skip

    return AdcBlock(apply, diagonal_pphh_pphh_1(hf))


def block_cvs_pphh_pphh_1(hf, mp, intermediates):
    def apply(ampl):
        return AmplitudeVector(pphh=(
            # 0th order
            + 2.0 * einsum("iJac,bc->iJab", ampl.pphh, hf.fvv).antisymmetrise(2, 3)
            - 1.0 * einsum("ik,kJab->iJab", hf.foo, ampl.pphh)
            - 1.0 * einsum("JK,iKab->iJab", hf.fcc, ampl.pphh)
            # 1st order
            + (
                - 2.0 * einsum("iKac,KbJc->iJab", ampl.pphh, hf.cvcv)
                + 2.0 * einsum("icka,kJbc->iJab", hf.ovov, ampl.pphh)
            ).antisymmetrise(2, 3)
            + 1.0 * einsum("iJlK,lKab->iJab", hf.ococ, ampl.pphh)
            + 0.5 * einsum("iJcd,abcd->iJab", ampl.pphh, hf.vvvv)
        ))  # fmt: skip

    return AdcBlock(apply, diagonal_pphh_pphh_1(hf))


def diagonal_ppphhh_ppphhh_1(hf, intermediates):
    dinterm_ov = (
        direct_sum("a-i->ia", hf.fvv.diagonal(), hf.foo.diagonal())
        - 3.0 * einsum("iaia->ia", hf.ovov)
    ).evaluate()
    diag_oo = einsum("ijij->ij", hf.oooo).symmetrise()
    diag_vv = einsum("abab->ab", hf.vvvv).symmetrise()
    return AmplitudeVector(ppphhh=(
        # 0th + 1st order
        3.0 * direct_sum("ij,ka,bc->ijkabc", diag_oo, dinterm_ov,
                         diag_vv).symmetrise(0, 1, 2).symmetrise(3, 4, 5)
    ))


def block_ppphhh_ppphhh_1(hf, mp, intermediates):
    def apply(ampl):
        ur3 = ampl.ppphhh
        #diag = intermediates.df3
        return AmplitudeVector(ppphhh=(
            # 0th order
            # The scaling comment is given as: [comp_scaling] / [mem_scaling]
            intermediates.df3 * ur3
            # The following two lines should be identical
            ### # N^6: O^3V^3 / N^6: O^3V^3
            ### + 3.0 * einsum("ijkabc,a->ijkabc", ur3, hf.fvv.diagonal()).antisymmetrise(3, 4, 5)
            ### # N^6: O^3V^3 / N^6: O^3V^3
            ### - 3.0 * einsum("ijkabc,i->ijkabc", ur3, hf.foo.diagonal()).antisymmetrise(0, 1, 2)
            # 1st order
            # The scaling comment is given as: [comp_scaling] / [mem_scaling]
            # N^8: O^4V^4 / N^6: O^3V^3
            + 9 * einsum("iklabd,jdlc->ijkabc", ur3, hf.ovov
            ).antisymmetrise(0, 1, 2).antisymmetrise(3, 4, 5)
            # N^8: O^5V^3 / N^6: O^3V^3
            + 1.5 * einsum("ilmabc,jklm->ijkabc", ur3, hf.oooo
            ).antisymmetrise(0, 1, 2)
            # N^8: O^3V^5 / N^6: O^3V^3
            + 1.5 * einsum("ijkade,bcde->ijkabc", ur3, hf.vvvv
            ).antisymmetrise(3, 4, 5)
        ))

    return AdcBlock(apply, diagonal_ppphhh_ppphhh_1(hf, intermediates))


#
# 1st order coupling
#
def block_ph_pphh_1(hf, mp, intermediates):
    def apply(ampl):
        return AmplitudeVector(ph=(
            + einsum("jkib,jkab->ia", hf.ooov, ampl.pphh)
            + einsum("ijbc,jabc->ia", ampl.pphh, hf.ovvv)
        ))  # fmt: skip

    return AdcBlock(apply, 0)


def block_cvs_ph_pphh_1(hf, mp, intermediates):
    def apply(ampl):
        return AmplitudeVector(ph=(
            + sqrt(2) * einsum("jKIb,jKab->Ia", hf.occv, ampl.pphh)
            - 1 / sqrt(2) * einsum("jIbc,jabc->Ia", ampl.pphh, hf.ovvv)
        ))  # fmt: skip

    return AdcBlock(apply, 0)


def block_pphh_ph_1(hf, mp, intermediates):
    def apply(ampl):
        return AmplitudeVector(pphh=(
            + einsum("ic,jcab->ijab", ampl.ph, hf.ovvv).antisymmetrise(0, 1)
            - einsum("ijka,kb->ijab", hf.ooov, ampl.ph).antisymmetrise(2, 3)
        ))  # fmt: skip

    return AdcBlock(apply, 0)


def block_cvs_pphh_ph_1(hf, mp, intermediates):
    def apply(ampl):
        return AmplitudeVector(pphh=(
            + sqrt(2) * einsum("jIKb,Ka->jIab",
                               hf.occv, ampl.ph).antisymmetrise(2, 3)
            - 1 / sqrt(2) * einsum("Ic,jcab->jIab", ampl.ph, hf.ovvv)
        ))  # fmt: skip

    return AdcBlock(apply, 0)


def block_ph_ppphhh_1(hf, mp, intermediates):
    return AdcBlock(lambda ampl: 0, 0)


def block_ppphhh_ph_1(hf, mp, intermediates):
    return AdcBlock(lambda ampl: 0, 0)


def block_pphh_ppphhh_1(hf, mp, intermediates):
    def apply(ampl):
        ur3 = ampl.ppphhh
        return AmplitudeVector(pphh=(
            2 * (
                # N^7: O^3V^4 / N^6: O^3V^3
                - 3 / 2 * einsum("ijkacd,kbcd->ijab",
                                 ur3, hf.ovvv)
            ).antisymmetrise(2, 3)
            + 2 * (
                # N^7: O^4V^3 / N^6: O^3V^3
                - 3 / 2 * einsum("iklabc,kljc->ijab",
                                 ur3, hf.ooov)
            ).antisymmetrise(0, 1)
        ))  # fmt: skip

    return AdcBlock(apply, 0)


def block_ppphhh_pphh_1(hf, mp, intermediates):
    def apply(ampl):
        ur2 = ampl.pphh
        return AmplitudeVector(ppphhh=(
            # Prefactor of 9: the antisymmetrisation generates 36 terms with
            # an overall factor of 1/36, so each term appears with weight 1/4,
            # yielding effectively 9 distinct terms.
            9 * (
                # N^7: O^4V^3 / N^6: O^3V^3
                - 1 / 3 * einsum("ilab,jklc->ijkabc", ur2, hf.ooov)
                # N^7: O^3V^4 / N^6: O^3V^3
                - 1 / 3 * einsum("ijad,kdbc->ijkabc", ur2, hf.ovvv)
            ).antisymmetrise(0, 1, 2).antisymmetrise(3, 4, 5)
        ))  # fmt: skip

    return AdcBlock(apply, 0)


#
# 2nd order main
#
def block_ph_ph_2(hf, mp, intermediates):
    i1 = intermediates.adc2_i1
    i2 = intermediates.adc2_i2
    diagonal = AmplitudeVector(ph=(
        + direct_sum("a-i->ia", i1.diagonal(), i2.diagonal())
        - einsum("IaIa->Ia", hf.ovov)
        - einsum("ikac,ikac->ia", mp.t2oo, hf.oovv)
    ))  # fmt: skip

    # Not used anywhere else, so kept as an anonymous intermediate
    term_t2_eri = (
        + einsum("ijab,jkbc->ikac", mp.t2oo, hf.oovv)
        + einsum("ijab,jkbc->ikac", hf.oovv, mp.t2oo)
    ).evaluate()  # fmt: skip

    def apply(ampl):
        return AmplitudeVector(ph=(
            + einsum("ib,ab->ia", ampl.ph, i1)
            - einsum("ij,ja->ia", i2, ampl.ph)
            - einsum("jaib,jb->ia", hf.ovov, ampl.ph)    # 1
            - 0.5 * einsum("ikac,kc->ia", term_t2_eri, ampl.ph)  # 2
        ))  # fmt: skip

    return AdcBlock(apply, diagonal)


def block_cvs_ph_ph_2(hf, mp, intermediates):
    i1 = intermediates.adc2_i1
    diagonal = AmplitudeVector(ph=(
        + direct_sum("a-i->ia", i1.diagonal(), hf.fcc.diagonal())
        - einsum("IaIa->Ia", hf.cvcv)
    ))  # fmt: skip

    def apply(ampl):
        return AmplitudeVector(ph=(
            + einsum("ib,ab->ia", ampl.ph, i1)
            - einsum("ij,ja->ia", hf.fcc, ampl.ph)
            - einsum("JaIb,Jb->Ia", hf.cvcv, ampl.ph)
        ))  # fmt: skip

    return AdcBlock(apply, diagonal)


def diagonal_pphh_pphh_2(hf, mp):
    # Fock matrix and ovov diagonal term (sometimes called "intermediate diagonal")
    dinterm_ov = (
        direct_sum("a-i->ia", hf.fvv.diagonal(), hf.foo.diagonal())
        - 2.0 * einsum("iaia->ia", hf.ovov)
    ).evaluate()

    t2_1 = mp.t2(b.oovv)

    return AmplitudeVector(pphh=(
        # 0th + 1st order
        + direct_sum("ia+Jb->iJab", dinterm_ov, dinterm_ov).symmetrise(2, 3)
        + direct_sum("iJ+ab->iJab", einsum("ijij->ij", hf.oooo).symmetrise(),
                     einsum("abab->ab", hf.vvvv).symmetrise())
        # 2nd order
        + direct_sum("i+jab->ijab", einsum("ikcd,ikcd->i", t2_1, hf.oovv),
                     einsum("jkab,jkab->jab", t2_1, hf.oovv)).symmetrise(0, 1)
        + direct_sum("ija+b->ijab", einsum("ijac,ijac->ija", t2_1, hf.oovv),
                     einsum("klbc,klbc->b", t2_1, hf.oovv)).symmetrise(2, 3)
        + 0.25 * direct_sum("ij+ab->ijab", einsum("ijcd,ijcd->ij", t2_1, hf.oovv),
                            einsum("klab,klab->ab", t2_1, hf.oovv))
        + 2 * direct_sum("ia+jb->ijab", einsum("ikac,ikac->ia", t2_1, hf.oovv),
                         einsum("ikac,ikac->ia", t2_1, hf.oovv)).symmetrise(2, 3)
    ))  # fmt: skip


def block_pphh_pphh_2(hf, mp, intermediates):
    t2_1 = mp.t2(b.oovv)

    def apply(ampl):
        ur2 = ampl.pphh
        return AmplitudeVector(pphh=(  # 0th order
            + 2 * einsum("ijac,bc->ijab", ur2, hf.fvv).antisymmetrise(2, 3)
            - 2 * einsum("ik,kjab->ijab", hf.foo, ur2).antisymmetrise(0, 1)
            # 1st order
            + (
                -4 * einsum("ikac,kbjc->ijab", ur2, hf.ovov)
            ).antisymmetrise(0, 1).antisymmetrise(2, 3)
            + 0.5 * einsum("ijkl,klab->ijab", hf.oooo, ur2)
            + 0.5 * einsum("ijcd,abcd->ijab", ur2, hf.vvvv)
            # 2nd order
            + 2 * (
                # N^5: O^3V^2 / N^4: O^2V^2
                - 0.25 * einsum("jl,ilab->ijab",
                                einsum("jkcd,klcd->jl", ur2, hf.oovv), t2_1)
                # N^5: O^3V^2 / N^4: O^2V^2
                - 0.25 * einsum("jl,ilab->ijab",
                                einsum("jkcd,klcd->jl", ur2, t2_1), hf.oovv)
                # N^5: O^3V^2 / N^4: O^2V^2
                - 0.25 * einsum("ik,jkab->ijab",
                                einsum("ilcd,klcd->ik", t2_1, hf.oovv), ur2)
                # N^5: O^3V^2 / N^4: O^2V^2
                - 0.25 * einsum("ik,jkab->ijab",
                                einsum("klcd,ilcd->ik", t2_1, hf.oovv), ur2)
            ).antisymmetrise(0, 1)
            + (
                # N^6: O^4V^2 / N^4: O^2V^2
                - 1 / 8 * einsum("ijkl,klab->ijab",
                                 einsum("ijcd,klcd->ijkl", ur2, hf.oovv), t2_1)
                # N^6:  O^4V^2 / N^4: O^2V^2
                - 1 / 8 * einsum("ijkl,klab->ijab",
                                 einsum("ijcd,klcd->ijkl", ur2, t2_1), hf.oovv)
                # N^6: O^4V^2 / N^4: O^2V^2
                - 1 / 8 * einsum("ijkl,klab->ijab",
                                 einsum("ijcd,klcd->ijkl", t2_1, hf.oovv), ur2)
                # N^6: O^4V^2 / N^4: O^2V^2
                - 1 / 8 * einsum("ijkl,klab->ijab",
                                 einsum("klcd,ijcd->ijkl", t2_1, hf.oovv), ur2)
            )
            + 2 * (
                # N^5: O^2V^3 / N^4: O^2V^2
                - 0.25 * einsum("ac,ijbc->ijab",
                                einsum("klad,klcd->ac", t2_1, hf.oovv), ur2)
                # N^5: O^2V^3 / N^4: O^2V^2
                - 0.25 * einsum("ac,ijbc->ijab",
                                einsum("klcd,klad->ac", t2_1, hf.oovv), ur2)
                # N^5: O^2V^3 / N^4: O^2V^2
                - 0.25 * einsum("bd,ijad->ijab",
                                einsum("klbc,klcd->bd", ur2, hf.oovv), t2_1)
                # N^5: O^2V^3 / N^4: O^2V^2
                - 0.25 * einsum("bd,ijad->ijab",
                                einsum("klbc,klcd->bd", ur2, t2_1), hf.oovv)
            ).antisymmetrise(2, 3)
            + 4 * (
                # N^6: O^3V^3 / N^4: O^2V^2
                + 0.5 * einsum("jlad,ilbd->ijab",
                               einsum("jkac,klcd->jlad", ur2, hf.oovv), t2_1)
                # N^6: O^3V^3 / N^4: O^2V^2
                + 0.5 * einsum("jlad,ilbd->ijab",
                               einsum("jkac,klcd->jlad", ur2, t2_1), hf.oovv)
            ).antisymmetrise(0, 1).antisymmetrise(2, 3)
        ))  # fmt: skip

    return AdcBlock(apply, diagonal_pphh_pphh_2(hf, mp))


#
# 2nd order coupling
#
def block_ph_pphh_2(hf, mp, intermediates):
    pia_ooov = intermediates.adc3_pia
    pib_ovvv = intermediates.adc3_pib

    def apply(ampl):
        return AmplitudeVector(ph=(
            + einsum("jkib,jkab->ia", pia_ooov, ampl.pphh)
            + einsum("ijbc,jabc->ia", ampl.pphh, pib_ovvv)
            + einsum("icab,jkcd,jkbd->ia", hf.ovvv, ampl.pphh, mp.t2oo)  # 2nd
            + einsum("ijka,jlbc,klbc->ia", hf.ooov, mp.t2oo, ampl.pphh)  # 2nd
        ))  # fmt: skip

    return AdcBlock(apply, 0)


def block_cvs_ph_pphh_2(hf, mp, intermediates):
    pia_occv = intermediates.cvs_adc3_pia
    pib_ovvv = intermediates.adc3_pib

    def apply(ampl):
        return AmplitudeVector(ph=(1 / sqrt(2)) * (
            + 2.0 * einsum("lKIc,lKac->Ia", pia_occv, ampl.pphh)
            - einsum("lIcd,lacd->Ia", ampl.pphh, pib_ovvv)
            - einsum("jIKa,ljcd,lKcd->Ia", hf.occv, mp.t2oo, ampl.pphh)
        ))  # fmt: skip

    return AdcBlock(apply, 0)


def block_pphh_ph_2(hf, mp, intermediates):
    pia_ooov = intermediates.adc3_pia
    pib_ovvv = intermediates.adc3_pib

    def apply(ampl):
        return AmplitudeVector(pphh=(
            (
                + einsum("ic,jcab->ijab", ampl.ph, pib_ovvv)
                + einsum("lkic,kc,jlab->ijab", hf.ooov, ampl.ph, mp.t2oo)  # 2st
            ).antisymmetrise(0, 1)
            + (
                - einsum("ijka,kb->ijab", pia_ooov, ampl.ph)
                - einsum("ijac,kbcd,kd->ijab", mp.t2oo, hf.ovvv, ampl.ph)  # 2st
            ).antisymmetrise(2, 3)
        ))  # fmt: skip

    return AdcBlock(apply, 0)


def block_cvs_pphh_ph_2(hf, mp, intermediates):
    pia_occv = intermediates.cvs_adc3_pia
    pib_ovvv = intermediates.adc3_pib

    def apply(ampl):
        return AmplitudeVector(pphh=(1 / sqrt(2)) * (
            - 2.0 * einsum("jIKa,Kb->jIab", pia_occv, ampl.ph).antisymmetrise(2, 3)
            - einsum("Ic,jcab->jIab", ampl.ph, pib_ovvv)
            - einsum("lKIc,Kc,jlab->jIab", hf.occv, ampl.ph, mp.t2oo)
        ))  # fmt: skip

    return AdcBlock(apply, 0)


def block_ph_ppphhh_2(hf, mp, intermediates):
    t2_1 = mp.t2(b.oovv)

    def apply(ampl):
        ur3 = ampl.ppphhh
        return AmplitudeVector(ph=(
            # N^7: O^4V^3 / N^6: O^3V^3
            + 3 * einsum("ijld,jald->ia",
                         einsum("ijkbcd,klbc->ijld", ur3, t2_1), hf.ovov)
            # N^7: O^3V^4 / N^6: O^3V^3
            + 3 / 2 * einsum("ibde,aebd->ia",
                             einsum("ijkbcd,jkce->ibde", ur3, t2_1), hf.vvvv)
            # N^7: O^3V^4 / N^6: O^3V^3
            + 3 * einsum("kacd,ickd->ia",
                         einsum("jklabc,jlbd->kacd", ur3, t2_1), hf.ovov)
            # N^7: O^4V^3 / N^6: O^3V^3
            + 3 / 2 * einsum("jlma,imjl->ia",
                             einsum("jklabc,kmbc->jlma", ur3, t2_1), hf.oooo)
        ))  # fmt: skip

    return AdcBlock(apply, 0)


def block_ppphhh_ph_2(hf, mp, intermediates):
    t2_1 = mp.t2(b.oovv)

    def apply(ampl):
        ur1 = ampl.ph
        return AmplitudeVector(ppphhh=(
            # Prefactor of 18: the antisymmetrisation generates 36 terms with
            # an overall factor of 1/36, so each term appears with weight 1/2,
            # yielding effectively 18 distinct terms.
            18 * (
                # N^7: O^4V^3 / N^6: O^3V^3
                - 1 / 6 * einsum("iklc,jlab->ijkabc",
                                 einsum("id,kdlc->iklc", ur1, hf.ovov), t2_1)
                # N^7: O^3V^4 / N^6: O^3V^3
                - 1 / 6 * einsum("kacd,ijbd->ijkabc",
                                 einsum("la,kdlc->kacd", ur1, hf.ovov), t2_1)
            ).antisymmetrise(0, 1, 2).antisymmetrise(3, 4, 5)
            # Prefactor of 9: the antisymmetrisation generates 36 terms with
            # an overall factor of 1/36, so each term appears with weight 1/4,
            # yielding effectively 9 distinct terms.
            + 9 * (
                # N^7: O^3V^4 / N^6: O^3V^3
                - 1 / 6 * einsum("ibce,jkae->ijkabc",
                                 einsum("id,bcde->ibce", ur1, hf.vvvv), t2_1)
                # N^7: O^4V^3 / N^6: O^3V^3
                - 1 / 6 * einsum("jkma,imbc->ijkabc",
                                 einsum("la,jklm->jkma", ur1, hf.oooo), t2_1)
            ).antisymmetrise(0, 1, 2).antisymmetrise(3, 4, 5)
        ))  # fmt: skip

    return AdcBlock(apply, 0)


def block_pphh_ppphhh_2(hf, mp, intermediates):
    t2_1 = mp.t2(b.oovv)

    def apply(ampl):
        ur3 = ampl.ppphhh
        return AmplitudeVector(pphh=(
            + 2 * ( 
                # N^7: O^4V^3 / N^6: O^3V^3
                - 3 * einsum("jklc,iklabc->ijab",
                             einsum("kmcd,jmld->jklc", t2_1, hf.ooov), ur3)
                - 3 / 4 * einsum("iklc,jklabc->ijab",
                                 einsum("klde,icde->iklc", t2_1, hf.ovvv), ur3)
            ).antisymmetrise(0, 1)
            + 2 * (
                # N^7: O^3V^4 / N^6: O^3V^3
                - 3 * einsum("kbcd,ijkacd->ijab",
                             einsum("klce,ldbe->kbcd", t2_1, hf.ovvv), ur3)
                - 3 / 4 * einsum("kacd,ijkbcd->ijab",
                                 einsum("lmcd,lmka->kacd", t2_1, hf.ooov), ur3)
            ).antisymmetrise(2, 3)
            + 4 * (
                # N^7: O^4V^3 / N^6: O^3V^3
                - 3 / 2 * einsum("ilma,jmlb->ijab",
                                 einsum("iklacd,kmcd->ilma", ur3, t2_1), hf.ooov)
                # N^7: O^3V^4 / N^6: O^3V^3
                - 3 / 2 * einsum("iade,jdbe->ijab",
                                 einsum("iklacd,klce->iade", ur3, t2_1), hf.ovvv)
            ).antisymmetrise(0, 1).antisymmetrise(2, 3)
            # N^7: O^4V^3 / N^6: O^3V^3
            + 3 / 2 * einsum("ijle,leab->ijab",
                             einsum("ijkcde,klcd->ijle", ur3, t2_1), hf.ovvv)
            # N^7: O^3V^4 / N^6: O^3V^3
            + 3 / 2 * einsum("mabd,ijmd->ijab",
                             einsum("klmabc,klcd->mabd", ur3, t2_1), hf.ooov)
        ))  # fmt: skip

    return AdcBlock(apply, 0)


def block_ppphhh_pphh_2(hf, mp, intermediates):
    t2_1 = mp.t2(b.oovv)

    def apply(ampl):
        ur2 = ampl.pphh
        return AmplitudeVector(ppphhh=(
            # Prefactor of 18: the antisymmetrisation generates 36 terms with
            # an overall factor of 1/36, so each term appears with weight 1/2,
            # yielding effectively 18 distinct terms.
            18 * (
                # N^7: O^4V^3 / N^6: O^3V^3
                - 1 / 3 * einsum("jklc,ilab->ijkabc",
                                 einsum("jmcd,lmkd->jklc", t2_1, hf.ooov), ur2)
                - 1 / 3 * einsum("ikma,jmbc->ijkabc",
                                 einsum("ilad,lmkd->ikma", ur2, hf.ooov), t2_1)
                # N^7: O^3V^4 / N^6: O^3V^3
                - 1 / 3 * einsum("kbcd,ijad->ijkabc",
                                 einsum("klbe,lcde->kbcd", t2_1, hf.ovvv), ur2)
                - 1 / 3 * einsum("iace,jkbe->ijkabc",
                                 einsum("ilad,lcde->iace", ur2, hf.ovvv), t2_1)
            ).antisymmetrise(0, 1, 2).antisymmetrise(3, 4, 5)
            # Prefactor of 9: the antisymmetrisation generates 36 terms with
            # an overall factor of 1/36, so each term appears with weight 1/4,
            # yielding effectively 9 distinct terms.
            + 9 * (
                # N^7: O^3V^4 / N^6: O^3V^3
                - 1 / 6 * einsum("jabd,ikcd->ijkabc",
                                 einsum("lmab,lmjd->jabd", ur2, hf.ooov), t2_1)
                - 1 / 6 * einsum("jbcd,ikad->ijkabc",
                                 einsum("lmbc,lmjd->jbcd", t2_1, hf.ooov), ur2)
                # N^7: O^4V^3 / N^6: O^3V^3
                - 1 / 6 * einsum("iklc,jlab->ijkabc",
                                 einsum("ikde,lcde->iklc", t2_1, hf.ovvv), ur2)
                - 1 / 6 * einsum("ijlb,klac->ijkabc",
                                 einsum("ijde,lbde->ijlb", ur2, hf.ovvv), t2_1)
            ).antisymmetrise(0, 1, 2).antisymmetrise(3, 4, 5)
        ))  # fmt: skip

    return AdcBlock(apply, 0)


#
# 3rd order main
#
def block_ph_ph_3(hf, mp, intermediates):
    if hf.has_core_occupied_space:
        m11 = intermediates.cvs_adc3_m11
    else:
        m11 = intermediates.adc3_m11
    diagonal = AmplitudeVector(ph=einsum("iaia->ia", m11))

    def apply(ampl):
        return AmplitudeVector(ph=einsum("iajb,jb->ia", m11, ampl.ph))

    return AdcBlock(apply, diagonal)


block_cvs_ph_ph_3 = block_ph_ph_3


def diagonal_pphh_pphh_3(hf, mp):
    pass
   # # Fock matrix and ovov diagonal term (sometimes called "intermediate diagonal")
   # dinterm_ov = (
   #     direct_sum("a-i->ia", hf.fvv.diagonal(), hf.foo.diagonal())
   #     - 2.0 * einsum("iaia->ia", hf.ovov)
   # ).evaluate()

   # t2_1 = mp.t2(b.oovv)

   # return AmplitudeVector(pphh=(
   #     # 0th + 1st order
   #     + direct_sum("ia+Jb->iJab", dinterm_ov, dinterm_ov).symmetrise(2, 3)
   #     + direct_sum("iJ+ab->iJab", einsum("ijij->ij", hf.oooo).symmetrise(),
   #                  einsum("abab->ab", hf.vvvv).symmetrise())
   #     # 2nd order
   #     + direct_sum("i+jab->ijab", einsum("ikcd,ikcd->i", t2_1, hf.oovv),
   #                  einsum("jkab,jkab->jab", t2_1, hf.oovv)).symmetrise(0, 1)
   #     + direct_sum("ija+b->ijab", einsum("ijac,ijac->ija", t2_1, hf.oovv),
   #                  einsum("klbc,klbc->b", t2_1, hf.oovv)).symmetrise(2, 3)
   #     + 0.25 * direct_sum("ij+ab->ijab", einsum("ijcd,ijcd->ij", t2_1, hf.oovv),
   #                         einsum("klab,klab->ab", t2_1, hf.oovv))
   #     + 2 * direct_sum("ia+jb->ijab", einsum("ikac,ikac->ia", t2_1, hf.oovv),
   #                      einsum("ikac,ikac->ia", t2_1, hf.oovv)).symmetrise(2, 3)
   #     # 3rd order
   #     
   # ))  # fmt: skip


def block_pphh_pphh_3(hf, mp, intermediates):
    t2_1 = mp.t2(b.oovv)
    t2_2 = mp.td2(b.oovv)

    p0_2 = mp.mp2_diffdm
    p0_2_oo, p0_2_vv, t1_2 = p0_2.oo, p0_2.vv, p0_2.ov

    t2sq = intermediates.t2sq
    t2eri_3 = mp.t2eri(b.oovv, b.oo).evaluate()
    t2eri_4 = mp.t2eri(b.oovv, b.ov).evaluate()
    t2eri_5 = mp.t2eri(b.oovv, b.vv).evaluate()

    def apply(ampl):
        ur2 = ampl.pphh
        return AmplitudeVector(pphh=(  # 0th order
            + 2 * einsum("ijac,bc->ijab", ur2, hf.fvv).antisymmetrise(2, 3)
            - 2 * einsum("ik,kjab->ijab", hf.foo, ur2).antisymmetrise(0, 1)
            # 1st order
            + (
                -4 * einsum("ikac,kbjc->ijab", ur2, hf.ovov)
            ).antisymmetrise(0, 1).antisymmetrise(2, 3)
            + 0.5 * einsum("ijkl,klab->ijab", hf.oooo, ur2)
            + 0.5 * einsum("ijcd,abcd->ijab", ur2, hf.vvvv)
            # 2nd order
            + 2 * (
                # N^5: O^3V^2 / N^4: O^2V^2
                - 0.25 * einsum("jl,ilab->ijab",
                                einsum("jkcd,klcd->jl", ur2, hf.oovv), t2_1)
                # N^5: O^3V^2 / N^4: O^2V^2
                - 0.25 * einsum("jl,ilab->ijab",
                                einsum("jkcd,klcd->jl", ur2, t2_1), hf.oovv)
                # N^5: O^3V^2 / N^4: O^2V^2
                - 0.25 * einsum("ik,jkab->ijab",
                                einsum("ilcd,klcd->ik", t2_1, hf.oovv), ur2)
                # N^5: O^3V^2 / N^4: O^2V^2
                - 0.25 * einsum("ik,jkab->ijab",
                                einsum("klcd,ilcd->ik", t2_1, hf.oovv), ur2)
            ).antisymmetrise(0, 1)
            + (
                # N^6: O^4V^2 / N^4: O^2V^2
                - 1 / 8 * einsum("ijkl,klab->ijab",
                                 einsum("ijcd,klcd->ijkl", ur2, hf.oovv), t2_1)
                # N^6:  O^4V^2 / N^4: O^2V^2
                - 1 / 8 * einsum("ijkl,klab->ijab",
                                 einsum("ijcd,klcd->ijkl", ur2, t2_1), hf.oovv)
                # N^6: O^4V^2 / N^4: O^2V^2
                - 1 / 8 * einsum("ijkl,klab->ijab",
                                 einsum("ijcd,klcd->ijkl", t2_1, hf.oovv), ur2)
                # N^6: O^4V^2 / N^4: O^2V^2
                - 1 / 8 * einsum("ijkl,klab->ijab",
                                 einsum("klcd,ijcd->ijkl", t2_1, hf.oovv), ur2)
            )
            + 2 * (
                # N^5: O^2V^3 / N^4: O^2V^2
                - 0.25 * einsum("ac,ijbc->ijab",
                                einsum("klad,klcd->ac", t2_1, hf.oovv), ur2)
                # N^5: O^2V^3 / N^4: O^2V^2
                - 0.25 * einsum("ac,ijbc->ijab",
                                einsum("klcd,klad->ac", t2_1, hf.oovv), ur2)
                # N^5: O^2V^3 / N^4: O^2V^2
                - 0.25 * einsum("bd,ijad->ijab",
                                einsum("klbc,klcd->bd", ur2, hf.oovv), t2_1)
                # N^5: O^2V^3 / N^4: O^2V^2
                - 0.25 * einsum("bd,ijad->ijab",
                                einsum("klbc,klcd->bd", ur2, t2_1), hf.oovv)
            ).antisymmetrise(2, 3)
            + 4 * (
                # N^6: O^3V^3 / N^4: O^2V^2
                + 0.5 * einsum("jlad,ilbd->ijab",
                               einsum("jkac,klcd->jlad", ur2, hf.oovv), t2_1)
                # N^6: O^3V^3 / N^4: O^2V^2
                + 0.5 * einsum("jlad,ilbd->ijab",
                               einsum("jkac,klcd->jlad", ur2, t2_1), hf.oovv)
            ).antisymmetrise(0, 1).antisymmetrise(2, 3)
            # 3rd order
            + (
                # N^6: O^2V^4 / N^4: V^4
                + 0.5 * einsum("ijde,abde->ijab",
                               einsum("ijcd,ce->ijde", ur2, p0_2_vv), hf.vvvv)
                + 0.25 * einsum("ijde,abde->ijab", einsum("df,ijef->ijde",
                                einsum("klcd,klcf->df", ur2, t2_1), t2_1), hf.vvvv)
                # N^6: O^3V^3 / N^4: O^1V^3
                - 1 * einsum("ijkd,kdab->ijab",
                             einsum("ijcd,kc->ijkd", ur2, t1_2), hf.ovvv)
                # N^6: O^4V^2 / N^4: O^2V^2
                - 1 * einsum("ijkl,klab->ijab",
                             einsum("kc,ijlc->ijkl", t1_2, hf.ooov), ur2)
                + 0.5 * einsum("ijkl,klab->ijab",
                               einsum("ijcd,klcd->ijkl", t2_1, t2eri_4), ur2)
                + 0.5 * einsum("ijkl,klab->ijab",
                               einsum("ijcd,klcd->ijkl", ur2, t2eri_4), t2_1)
                - 0.5 * einsum("ijkl,klab->ijab",
                               einsum("ijlm,km->ijkl", hf.oooo, p0_2_oo), ur2)
                + 0.25 * einsum("ijmn,mnab->ijab", einsum("ln,ijlm->ijmn",
                                einsum("klcd,kncd->ln", ur2, t2_1), hf.oooo), t2_1)
                - 1 / 8 * einsum("ijkl,klab->ijab",
                                 einsum("ijcd,klcd->ijkl", t2_2, hf.oovv), ur2)
                - 1 / 8 * einsum("ijkl,klab->ijab",
                                 einsum("klcd,ijcd->ijkl", t2_2, hf.oovv), ur2)
                - 1 / 8 * einsum("ijkl,klab->ijab",
                                 einsum("ijcd,klcd->ijkl", ur2, hf.oovv), t2_2)
                - 1 / 8 * einsum("ijkl,klab->ijab",
                                 einsum("ijcd,klcd->ijkl", ur2, t2_2), hf.oovv)
            )
            + 2 * (
                # N^6: O^2V^4 / N^4: V^4
                + 0.5 * einsum("ijce,abce->ijab",
                               einsum("ikcd,jekd->ijce", ur2, t2sq), hf.vvvv)
                # N^6: O^3V^3 / N^4: O^2V^2
                - 1 * einsum("ijlm,lmab->ijab", einsum("ilde,jdme->ijlm",
                             einsum("ikcd,klce->ilde", ur2, t2_1), hf.ovov), t2_1)
                - 0.5 * einsum("jm,imab->ijab", einsum("kmde,jdke->jm",
                               einsum("klcd,lmce->kmde", ur2, t2_1), hf.ovov), t2_1)
                - 0.25 * einsum("ijlm,lmab->ijab", einsum("ijce,lmce->ijlm",
                                einsum("ikcd,jdke->ijce", ur2, hf.ovov), t2_1), t2_1)
                # N^6: O^4V^2 / N^4: O^2V^2
                - 1 * einsum("ijkl,klab->ijab",
                             einsum("icld,jckd->ijkl", hf.ovov, t2sq), ur2)
                + 0.5 * einsum("ijkl,klab->ijab",
                               einsum("jc,klic->ijkl", t1_2, hf.ooov), ur2)
                - 0.5 * einsum("ijkl,klab->ijab", einsum("ikmn,jnlm->ijkl",
                               einsum("imcd,kncd->ikmn", t2_1, t2_1), hf.oooo), ur2)
                - 0.5 * einsum("ijlm,lmab->ijab", einsum("ikmn,jnkl->ijlm",
                               einsum("ikcd,mncd->ikmn", ur2, t2_1), hf.oooo), t2_1)
                - 0.25 * einsum("ijkl,klab->ijab",
                                einsum("jmkl,im->ijkl", hf.oooo, p0_2_oo), ur2)
                + 0.25 * einsum("ijkl,klab->ijab",
                                einsum("klcd,ijcd->ijkl", t2_1, t2eri_4), ur2)
                - 1 / 8 * einsum("jm,imab->ijab", einsum("klmn,jnkl->jm",
                                 einsum("klcd,mncd->klmn", ur2, t2_1), hf.oooo), t2_1)
                # N^5: O^2V^3 / N^4: O^2V^2
                - 0.5 * einsum("jm,imab->ijab", einsum("de,jdme->jm",
                               einsum("klcd,klce->de", ur2, t2_1), hf.ovov), t2_1)
                # N^5: O^3V^2 / N^4: O^2V^2
                + 1 * einsum("jl,ilab->ijab",
                             einsum("kc,kljc->jl", t1_2, hf.ooov), ur2)
                + 1 * einsum("ik,jkab->ijab",
                             einsum("ickd,cd->ik", hf.ovov, p0_2_vv), ur2)
                + 1 * einsum("ik,jkab->ijab",
                             einsum("imkl,lm->ik", hf.oooo, p0_2_oo), ur2)
                + 1 * einsum("il,jlab->ijab",
                             einsum("kc,iklc->il", t1_2, hf.ooov), ur2)
                + 0.5 * einsum("jk,ikab->ijab",
                               einsum("jlcd,kldc->jk", t2_1, t2eri_4), ur2)
                + 0.5 * einsum("jk,ikab->ijab",
                               einsum("klcd,jldc->jk", t2_1, t2eri_4), ur2)
                + 0.5 * einsum("il,jlab->ijab",
                               einsum("ikcd,lkcd->il", ur2, t2eri_4), t2_1)
                - 0.5 * einsum("jm,imab->ijab", einsum("ln,jnlm->jm",
                               einsum("klcd,kncd->ln", ur2, t2_1), hf.oooo), t2_1)
                - 0.25 * einsum("jl,ilab->ijab",
                                einsum("jkcd,klcd->jl", t2_2, hf.oovv), ur2)
                - 0.25 * einsum("jl,ilab->ijab",
                                einsum("klcd,jkcd->jl", t2_2, hf.oovv), ur2)
                - 0.25 * einsum("ik,jkab->ijab",
                                einsum("ilcd,klcd->ik", ur2, hf.oovv), t2_2)
                - 0.25 * einsum("jl,ilab->ijab",
                                einsum("jkcd,klcd->jl", ur2, t2_2), hf.oovv)
                - 1 / 8 * einsum("jk,ikab->ijab",
                                 einsum("jlcd,klcd->jk", t2_1, t2eri_3), ur2)
                - 1 / 8 * einsum("jk,ikab->ijab",
                                 einsum("klcd,jlcd->jk", t2_1, t2eri_3), ur2)
                - 1 / 8 * einsum("il,jlab->ijab",
                                 einsum("ikcd,klcd->il", ur2, t2eri_3), t2_1)
                - 1 / 8 * einsum("il,jlab->ijab",
                                 einsum("ikcd,klcd->il", ur2, t2_1), t2eri_3)
            ).antisymmetrise(0, 1)
            + 2 * (
                # N^6: V^6 / N^4: V^4
                - 0.5 * einsum("abde,ijde->ijab", einsum("acef,bfcd->abde",
                               einsum("klac,klef->acef", ur2, t2_1), hf.vvvv), t2_1)
                - 0.5 * einsum("abcd,ijcd->ijab", einsum("acef,bfde->abcd",
                               einsum("klae,klcf->acef", t2_1, t2_1), hf.vvvv), ur2)
                # N^6: O^2V^4 / N^4: V^4
                + 1 * einsum("abcd,ijcd->ijab",
                             einsum("kbld,kalc->abcd", hf.ovov, t2sq), ur2)
                - 1 * einsum("abde,ijde->ijab", einsum("lmad,lbme->abde",
                             einsum("klac,kmcd->lmad", ur2, t2_1), hf.ovov), t2_1)
                - 0.25 * einsum("ijae,be->ijab",
                                einsum("ijcd,aecd->ijae", ur2, hf.vvvv), p0_2_vv)
                - 1 / 8 * einsum("be,ijae->ijab", einsum("klbf,klef->be",
                                 einsum("klcd,bfcd->klbf", ur2, hf.vvvv), t2_1), t2_1)
                # N^6: O^3V^3 / N^4: O^2V^2
                + 0.5 * einsum("kmab,ijkm->ijab",
                               einsum("klac,lcmb->kmab", ur2, t2sq), hf.oooo)
                + 0.5 * einsum("ijka,kb->ijab",
                               einsum("ijcd,kacd->ijka", ur2, hf.ovvv), t1_2)
                - 0.5 * einsum("be,ijae->ijab", einsum("kmde,kbmd->be",
                               einsum("klcd,lmce->kmde", ur2, t2_1), hf.ovov), t2_1)
                - 0.25 * einsum("ijkm,kmab->ijab", einsum("ijde,kmde->ijkm", t2_1, t2_1),
                                einsum("klac,lbmc->kmab", ur2, hf.ovov))
                # N^6: O^4V^2 / N^4: O^2V^2
                + 0.25 * einsum("ijkl,klab->ijab",
                                einsum("ijcd,klcd->ijkl", ur2, t2_1), t2eri_4)
                # N^5: O^2V^3 / N^4: V^4
                + 1 * einsum("bc,ijac->ijab",
                             einsum("becd,de->bc", hf.vvvv, p0_2_vv), ur2)
                - 0.5 * einsum("be,ijae->ijab", einsum("df,bfde->be",
                               einsum("klcd,klcf->df", ur2, t2_1), hf.vvvv), t2_1)
                # N^5: O^2V^3 / N^4: O^1V^3
                + 1 * einsum("bd,ijad->ijab",
                             einsum("kc,kbcd->bd", t1_2, hf.ovvv), ur2)
                + 1 * einsum("ad,ijbd->ijab",
                             einsum("kc,kdac->ad", t1_2, hf.ovvv), ur2)
                # N^5: O^2V^3 / N^4: O^2V^2
                + 1 * einsum("bc,ijac->ijab",
                             einsum("kblc,kl->bc", hf.ovov, p0_2_oo), ur2)
                + 0.5 * einsum("bc,ijac->ijab",
                               einsum("klbd,kldc->bc", t2_1, t2eri_4), ur2)
                + 0.5 * einsum("bc,ijac->ijab",
                               einsum("klcd,kldb->bc", t2_1, t2eri_4), ur2)
                + 0.5 * einsum("ad,ijbd->ijab",
                               einsum("klac,lkcd->ad", ur2, t2eri_4), t2_1)
                - 0.5 * einsum("be,ijae->ijab", einsum("lm,lbme->be",
                               einsum("klcd,kmcd->lm", ur2, t2_1), hf.ovov), t2_1)
                - 0.25 * einsum("bd,ijad->ijab",
                                einsum("klbc,klcd->bd", t2_2, hf.oovv), ur2)
                - 0.25 * einsum("bd,ijad->ijab",
                                einsum("klcd,klbc->bd", t2_2, hf.oovv), ur2)
                - 0.25 * einsum("ac,ijbc->ijab",
                                einsum("klad,klcd->ac", ur2, hf.oovv), t2_2)
                - 0.25 * einsum("bd,ijad->ijab",
                                einsum("klbc,klcd->bd", ur2, t2_2), hf.oovv)
                - 1 / 8 * einsum("bc,ijac->ijab",
                                 einsum("klbd,klcd->bc", t2_1, t2eri_5), ur2)
                - 1 / 8 * einsum("bc,ijac->ijab",
                                 einsum("klcd,klbd->bc", t2_1, t2eri_5), ur2)
                - 1 / 8 * einsum("ad,ijbd->ijab",
                                 einsum("klac,klcd->ad", ur2, t2eri_5), t2_1)
                - 1 / 8 * einsum("ad,ijbd->ijab",
                                 einsum("klac,klcd->ad", ur2, t2_1), t2eri_5)
            ).antisymmetrise(2, 3)
            + 4 * (
                # N^6: O^2V^4 / N^4: V^4
                + 1 * einsum("ikbc,jkac->ijab",
                             einsum("becd,idke->ikbc", hf.vvvv, t2sq), ur2)
                + 1 * einsum("ilbe,jlae->ijab", einsum("ildf,bfde->ilbe",
                             einsum("ikcd,klcf->ildf", ur2, t2_1), hf.vvvv), t2_1)
                + 0.5 * einsum("jkbc,ikac->ijab", einsum("bcde,jdke->jkbc",
                               einsum("lmbd,lmce->bcde", t2_1, t2_1), hf.ovov), ur2)
                + 0.5 * einsum("imad,jmbd->ijab", einsum("acde,icme->imad",
                               einsum("klac,klde->acde", ur2, t2_1), hf.ovov), t2_1)
                - 0.25 * einsum("ikbe,jake->ijab",
                                einsum("ikcd,becd->ikbe", ur2, hf.vvvv), t2sq)
                # N^6: O^3V^3 / N^4: O^2V^2
                + 1 * einsum("ikbc,jkac->ijab",
                             einsum("icld,kdlb->ikbc", hf.ovov, t2sq), ur2)
                + 1 * einsum("ikbc,jkac->ijab",
                             einsum("imkl,lbmc->ikbc", hf.oooo, t2sq), ur2)
                + 1 * einsum("ikbc,jkac->ijab",
                             einsum("kbld,idlc->ikbc", hf.ovov, t2sq), ur2)
                + 1 * einsum("jkbd,ikad->ijab",
                             einsum("kc,jdbc->jkbd", t1_2, hf.ovvv), ur2)
                + 1 * einsum("ikbd,jkad->ijab",
                             einsum("ic,kbcd->ikbd", t1_2, hf.ovvv), ur2)
                + 1 * einsum("jmad,imbd->ijab", einsum("lnad,jnlm->jmad",
                             einsum("klac,kncd->lnad", ur2, t2_1), hf.oooo), t2_1)
                + 1 * einsum("jmad,imbd->ijab", einsum("kmae,jdke->jmad",
                             einsum("klac,lmce->kmae", ur2, t2_1), hf.ovov), t2_1)
                + 1 * einsum("ilbe,jlae->ijab", einsum("imce,lbmc->ilbe",
                             einsum("ikcd,kmde->imce", ur2, t2_1), hf.ovov), t2_1)
                + 0.5 * einsum("ilad,jlbd->ijab",
                               einsum("ikac,klcd->ilad", ur2, t2eri_4), t2_1)
                + 0.5 * einsum("ilad,jlbd->ijab",
                               einsum("ikac,lkdc->ilad", ur2, t2eri_4), t2_1)
                + 0.5 * einsum("ilad,jlbd->ijab",
                               einsum("ikac,klcd->ilad", ur2, t2_1), t2eri_4)
                + 0.5 * einsum("ilad,ljdb->ijab",
                               einsum("ikac,klcd->ilad", ur2, t2_1), t2eri_4)
                + 0.5 * einsum("ijad,bd->ijab",
                               einsum("ikac,jckd->ijad", ur2, hf.ovov), p0_2_vv)
                + 0.5 * einsum("ikad,jdkb->ijab",
                               einsum("ikac,cd->ikad", ur2, p0_2_vv), hf.ovov)
                + 0.5 * einsum("jlac,iclb->ijab",
                               einsum("jkac,kl->jlac", ur2, p0_2_oo), hf.ovov)
                + 0.5 * einsum("jlab,il->ijab",
                               einsum("jkac,kblc->jlab", ur2, hf.ovov), p0_2_oo)
                + 0.5 * einsum("ikad,jbkd->ijab",
                               einsum("klac,icld->ikad", ur2, hf.ovov), t2sq)
                + 0.5 * einsum("jlad,idlb->ijab",
                               einsum("klac,jdkc->jlad", ur2, t2sq), hf.ovov)
                + 0.5 * einsum("jkac,ikbc->ijab",
                               einsum("jlad,klcd->jkac", ur2, hf.oovv), t2_2)
                + 0.5 * einsum("jkac,ikbc->ijab",
                               einsum("jlad,klcd->jkac", ur2, t2_2), hf.oovv)
                + 0.5 * einsum("ilac,jclb->ijab",
                               einsum("ikcd,kdla->ilac", ur2, t2sq), hf.ovov)
                + 0.5 * einsum("ilac,jblc->ijab",
                               einsum("ikcd,kald->ilac", ur2, hf.ovov), t2sq)
                + 0.5 * einsum("jkbc,ikac->ijab", einsum("jklm,lbmc->jkbc",
                               einsum("jlde,kmde->jklm", t2_1, t2_1), hf.ovov), ur2)
                + 0.5 * einsum("ilae,jlbe->ijab", einsum("iklm,kame->ilae",
                               einsum("ikcd,lmcd->iklm", ur2, t2_1), hf.ovov), t2_1)
                - 0.25 * einsum("jlad,ilbd->ijab",
                                einsum("jkac,klcd->jlad", ur2, t2eri_3), t2_1)
                - 0.25 * einsum("jlad,ilbd->ijab",
                                einsum("jkac,klcd->jlad", ur2, t2eri_5), t2_1)
                - 0.25 * einsum("jlad,ilbd->ijab",
                                einsum("jkac,klcd->jlad", ur2, t2_1), t2eri_3)
                - 0.25 * einsum("jlad,ilbd->ijab",
                                einsum("jkac,klcd->jlad", ur2, t2_1), t2eri_5)
                - 0.25 * einsum("jmac,ibmc->ijab",
                                einsum("klac,jmkl->jmac", ur2, hf.oooo), t2sq)
                - 0.25 * einsum("ikae,jekb->ijab", einsum("km,imae->ikae",
                                einsum("klcd,lmcd->km", ur2, t2_1), t2_1), hf.ovov)
                - 0.25 * einsum("imac,jcmb->ijab", einsum("ce,imae->imac",
                                einsum("klcd,klde->ce", ur2, t2_1), t2_1), hf.ovov)
                - 1 / 8 * einsum("imac,jcmb->ijab", einsum("iklm,klac->imac",
                                 einsum("imde,klde->iklm", t2_1, t2_1), ur2), hf.ovov)
                - 1 / 8 * einsum("ikae,jekb->ijab", einsum("iklm,lmae->ikae",
                                 einsum("ikcd,lmcd->iklm", ur2, t2_1), t2_1), hf.ovov)
                # N^6: O^4V^2 / N^4: O^2V^2
                + 1 * einsum("ikla,jklb->ijab",
                             einsum("ilac,kc->ikla", ur2, t1_2), hf.ooov)
                + 1 * einsum("ijka,kb->ijab",
                             einsum("jlac,klic->ijka", ur2, hf.ooov), t1_2)
                # N^5: O^2V^3 / N^4: O^2V^2
                - 0.25 * einsum("ad,ijbd->ijab",
                                einsum("klac,klcd->ad", ur2, t2_1), t2eri_4)
                # N^5: O^3V^2 / N^4: O^2V^2
                - 0.25 * einsum("il,ljba->ijab",
                                einsum("ikcd,klcd->il", ur2, t2_1), t2eri_4)
            ).antisymmetrise(0, 1).antisymmetrise(2, 3)
        ))  # fmt: skip

   # TO DO: Implement diagonal_pphh_pphh_3
   # return AdcBlock(apply, diagonal_pphh_pphh_3(hf, mp))
    return AdcBlock(apply, diagonal_pphh_pphh_2(hf, mp))


#
# 3rd order coupling
#
def block_ph_pphh_3(hf, mp, intermediates):
    pia_ooov = intermediates.adc3_pia
    pib_ovvv = intermediates.adc3_pib

    p0_2 = mp.diffdm(level=2)
    p0_2_oo, p0_2_vv, t1_2 = p0_2.oo, p0_2.vv, p0_2.ov

    t2sq = intermediates.t2sq
    t2_1 = mp.t2(b.oovv)
    t2_2 = mp.td2(b.oovv)
    t3_2 = mp.tt2(b.ooovvv)

    t2eri_1 = mp.t2eri(b.ooov, b.vv).evaluate()
    t2eri_2 = mp.t2eri(b.ooov, b.ov).evaluate()
    t2eri_6 = mp.t2eri(b.ovvv, b.oo).evaluate()
    t2eri_7 = mp.t2eri(b.ovvv, b.ov).evaluate()

    def apply(ampl):
        ur2 = ampl.pphh
        return AmplitudeVector(ph=(
            + einsum("jkib,jkab->ia", pia_ooov, ur2)
            + einsum("ijbc,jabc->ia", ur2, pib_ovvv)
            + einsum("icab,jkcd,jkbd->ia", hf.ovvv, ur2, t2_1)  # 2nd
            + einsum("ijka,jlbc,klbc->ia", hf.ooov, t2_1, ur2)  # 2nd
            # 3rd
            # N^6: O^4V^2 / N^4: O^2V^2
            + 1 * einsum("iklb,kbla->ia",
                         einsum("jkbc,jlic->iklb", ur2, hf.ooov), t2sq)
            # N^5: O^3V^2 / N^4: O^2V^2
            + 1 * einsum("kl,ikla->ia",
                         einsum("jlbc,jkbc->kl", ur2, t2_2), hf.ooov)
            # N^5: O^2V^3 / N^4: O^1V^3
            + 1 * einsum("cd,idac->ia",
                         einsum("jkbd,jkbc->cd", ur2, t2_2), hf.ovvv)
            # N^5: O^3V^2 / N^4: O^2V^2
            + 0.5 * einsum("kl,lika->ia",
                           einsum("jkbc,jlbc->kl", ur2, t2_1), t2eri_2)
            # N^5: O^2V^3 / N^4: O^1V^3
            + 0.5 * einsum("bd,ibda->ia",
                           einsum("jkbc,jkcd->bd", ur2, t2_1), t2eri_7)
            # N^6: O^3V^3 / N^6: O^3V^3
            + 0.5 * einsum("lc,ilac->ia",
                           einsum("jkbd,jklbcd->lc", ur2, t3_2), hf.oovv)
            # N^6: O^3V^3 / N^4: O^1V^3
            - 1 * einsum("ijbd,jabd->ia",
                         einsum("jkbc,idkc->ijbd", ur2, t2sq), hf.ovvv)
            # N^5: O^3V^2 / N^4: O^2V^2
            - 0.5 * einsum("lb,ialb->ia",
                           einsum("jkbc,jklc->lb", ur2, hf.ooov), t2sq)
            # N^5: O^2V^3 / N^4: O^1V^3
            - 0.5 * einsum("jd,iajd->ia",
                           einsum("jkbc,kdbc->jd", ur2, hf.ovvv), t2sq)
            # N^5: O^3V^2 / N^4: O^2V^2
            - 0.25 * einsum("kl,ilka->ia",
                            einsum("jkbc,jlbc->kl", ur2, t2_1), t2eri_1)
            # N^5: O^2V^3 / N^4: O^1V^3
            - 0.25 * einsum("cd,icad->ia",
                            einsum("jkbc,jkbd->cd", ur2, t2_1), t2eri_6)
            # N^6: O^4V^2 / N^4: O^2V^2
            + 1 / 8 * einsum("ilmd,lmad->ia",
                             einsum("jklm,jkid->ilmd",
                                    einsum("jkbc,lmbc->jklm", ur2, t2_1),
                                    hf.ooov), t2_1)
            # N^6: O^4V^2 / N^4: O^1V^3
            + 1 / 8 * einsum("ilbc,labc->ia",
                             einsum("ijkl,jkbc->ilbc",
                                    einsum("ilde,jkde->ijkl", t2_1, t2_1),
                                    ur2), hf.ovvv)
            # N^6: O^3V^3 / N^4: O^1V^3
            + 1 * einsum("ikbd,kbda->ia",
                         einsum("ijbc,jkcd->ikbd", ur2, t2_1), t2eri_7)
            # N^5: O^2V^3 / N^4: O^1V^3
            + 0.5 * einsum("id,ad->ia",
                           einsum("ijbc,jdbc->id", ur2, hf.ovvv), p0_2_vv)
            # N^5: O^2V^3 / N^4: O^1V^3
            + 0.5 * einsum("ikbc,kabc->ia",
                           einsum("ijbc,jk->ikbc", ur2, p0_2_oo), hf.ovvv)
            # N^7: O^4V^3 / N^6: O^3V^3
            + 0.5 * einsum("iklc,klac->ia",
                           einsum("ijbd,jklbcd->iklc", ur2, t3_2), hf.oovv)
            # N^5: O^2V^3 / N^4: O^1V^3
            - 1 * einsum("ijbd,jabd->ia",
                         einsum("ijbc,cd->ijbd", ur2, p0_2_vv), hf.ovvv)
            # N^5: O^1V^4 / N^4: V^4
            - 1 * einsum("ibcd,abcd->ia",
                         einsum("ijcd,jb->ibcd", ur2, t1_2), hf.vvvv)
            # N^5: O^3V^2 / N^4: O^2V^2
            - 2 * einsum("ijkc,jcka->ia",
                         einsum("ikbc,jb->ijkc", ur2, t1_2), hf.ovov)
            # N^6: O^3V^3 / N^4: O^1V^3
            + 2 * einsum("ikcd,kdac->ia",
                         einsum("ijbd,jkbc->ikcd", ur2, t2_2), hf.ovvv)
            # N^6: O^4V^2 / N^4: O^2V^2
            - 0.5 * einsum("ijkl,jkla->ia",
                           einsum("ilbc,jkbc->ijkl", ur2, t2_2), hf.ooov)
            # N^6: O^4V^2 / N^4: O^2V^2
            + 1 / 8 * einsum("ijkl,klja->ia",
                             einsum("ijbc,klbc->ijkl", ur2, t2_1), t2eri_1)
            # N^5: O^2V^3 / N^4: O^1V^3
            + 0.25 * einsum("ka,ik->ia",
                            einsum("klde,lade->ka", t2_1, hf.ovvv),
                            einsum("ijbc,jkbc->ik", ur2, t2_1))
            # N^7: O^4V^3 / N^6: O^3V^3
            + 0.5 * einsum("ijkc,jkac->ia",
                           einsum("jklbcd,ilbd->ijkc", t3_2, hf.oovv), ur2)
            # N^6: O^4V^2 / N^4: O^2V^2
            - 1 * einsum("ijkb,jkab->ia",
                         einsum("klbc,lijc->ijkb", t2_1, t2eri_2), ur2)
            # N^5: O^3V^2 / N^4: O^2V^2
            - 1 * einsum("ijkb,jkab->ia",
                         einsum("klib,jl->ijkb", hf.ooov, p0_2_oo), ur2)
            # N^5: O^3V^2 / N^4: O^2V^2
            - 1 * einsum("jkla,ijkl->ia",
                         einsum("klab,jb->jkla", ur2, t1_2), hf.oooo)
            # N^5: O^3V^2 / N^4: O^2V^2
            - 2 * einsum("ijkc,jkac->ia",
                         einsum("jb,ickb->ijkc", t1_2, hf.ovov), ur2)
            # N^6: O^4V^2 / N^4: O^2V^2
            + 2 * einsum("ijlb,jlab->ia",
                         einsum("jkbc,iklc->ijlb", t2_2, hf.ooov), ur2)
            # N^5: O^3V^2 / N^4: O^2V^2
            - 0.5 * einsum("ijkb,jkab->ia",
                           einsum("jkic,bc->ijkb", hf.ooov, p0_2_vv), ur2)
            # N^5: O^3V^2 / N^4: O^2V^2
            - 0.5 * einsum("la,il->ia",
                           einsum("jkab,jklb->la", ur2, hf.ooov), p0_2_oo)
            # N^6: O^3V^3 / N^4: O^1V^3
            - 0.5 * einsum("ijkd,jkad->ia",
                           einsum("jkbc,idbc->ijkd", t2_2, hf.ovvv), ur2)
            # N^6: O^3V^3 / N^4: O^1V^3
            + 1 / 8 * einsum("ijkb,jkab->ia",
                             einsum("jkcd,ibcd->ijkb", t2_1, t2eri_6), ur2)
            # N^5: O^3V^2 / N^4: O^2V^2
            + 0.25 * einsum("ijkb,jkab->ia",
                            einsum("ic,jkbc->ijkb",
                                   einsum("lmcd,lmid->ic", t2_1, hf.ooov),
                                   t2_1), ur2)
        ))  # fmt: skip

    return AdcBlock(apply, 0)


def block_pphh_ph_3(hf, mp, intermediates):
    pia_ooov = intermediates.adc3_pia
    pib_ovvv = intermediates.adc3_pib
    p0_2 = mp.diffdm(level=2)
    p0_2_oo, p0_2_vv, t1_2 = p0_2.oo, p0_2.vv, p0_2.ov

    t2_1 = mp.t2(b.oovv)
    t2_2 = mp.td2(b.oovv)
    t3_2 = mp.tt2(b.ooovvv)

    t2sq = intermediates.t2sq
    t2eri_1 = mp.t2eri(b.ooov, b.vv).evaluate()
    t2eri_2 = mp.t2eri(b.ooov, b.ov).evaluate()
    t2eri_6 = mp.t2eri(b.ovvv, b.oo).evaluate()
    t2eri_7 = mp.t2eri(b.ovvv, b.ov).evaluate()

    def apply(ampl):
        ur1 = ampl.ph
        return AmplitudeVector(pphh=(
            (
                + einsum("ic,jcab->ijab", ur1, pib_ovvv)
                + einsum("lkic,kc,jlab->ijab", hf.ooov, ur1, t2_1)  # 2st
            ).antisymmetrise(0, 1)
            + (
                - einsum("ijka,kb->ijab", pia_ooov, ur1)
                - einsum("ijac,kbcd,kd->ijab", t2_1, hf.ovvv, ur1)  # 2st
            ).antisymmetrise(2, 3)
            # 3rd order
            + 2 * (
                # N^5: O^3V^2 / N^4: O^2V^2
                + 0.5 * einsum("ik,jkab->ijab",
                               einsum("lc,klic->ik", ur1, hf.ooov), t2_2)
                # N^5: O^3V^2 / N^4: O^2V^2
                - 0.25 * einsum("il,jlab->ijab",
                                einsum("kc,lkic->il", ur1, t2eri_2), t2_1)
                # N^5: O^2V^3 / N^4: O^1V^3
                - 0.25 * einsum("id,jdab->ijab",
                                einsum("kc,idkc->id", ur1, t2sq), hf.ovvv)
                # N^5: O^3V^2 / N^4: O^2V^2
                - 1 / 8 * einsum("jl,ilab->ijab",
                                 einsum("kc,kljc->jl", ur1, t2eri_1), t2_1)
                # N^5: O^1V^4 / N^4: V^4
                + 0.5 * einsum("iabc,jc->ijab",
                               einsum("id,abcd->iabc", ur1, hf.vvvv), t1_2)
                # N^6: O^4V^2 / N^4: O^2V^2
                - 0.25 * einsum("ijkl,klab->ijab",
                                einsum("ic,kljc->ijkl", ur1, hf.ooov), t2_2)
                # N^5: O^2V^3 / N^4: O^1V^3
                - 0.25 * einsum("jd,idab->ijab",
                                einsum("jc,cd->jd", ur1, p0_2_vv), hf.ovvv)
                # N^5: O^2V^3 / N^4: O^1V^3
                - 0.25 * einsum("jkab,ik->ijab",
                                einsum("jc,kcab->jkab", ur1, hf.ovvv), p0_2_oo)
                # N^7: O^4V^3 / N^6: O^3V^3
                - 0.25 * einsum("jklc,iklabc->ijab",
                                einsum("jd,klcd->jklc", ur1, hf.oovv), t3_2)
                # N^6: O^4V^2 / N^4: O^2V^2
                - 1 / 16 * einsum("ijkl,klab->ijab",
                                  einsum("jc,klic->ijkl", ur1, t2eri_1), t2_1)
                # N^5: O^2V^3 / N^4: O^1V^3
                - 1 / 8 * einsum("jk,ikab->ijab",
                                 einsum("kc,jc->jk",
                                        einsum("klde,lcde->kc", t2_1, hf.ovvv),
                                        ur1), t2_1)
            ).antisymmetrise(0, 1)
            + 2 * (
                # N^5: O^2V^3 / N^4: O^1V^3
                + 0.5 * einsum("ac,ijbc->ijab",
                               einsum("kd,kacd->ac", ur1, hf.ovvv), t2_2)
                # N^5: O^2V^3 / N^4: O^1V^3
                - 0.25 * einsum("bd,ijad->ijab",
                                einsum("kc,kbdc->bd", ur1, t2eri_7), t2_1)
                # N^5: O^3V^2 / N^4: O^2V^2
                - 0.25 * einsum("la,ijlb->ijab",
                                einsum("kc,kcla->la", ur1, t2sq), hf.ooov)
                # N^5: O^2V^3 / N^4: O^1V^3
                - 1 / 8 * einsum("bd,ijad->ijab",
                                 einsum("kc,kbcd->bd", ur1, t2eri_6), t2_1)
                # N^5: O^3V^2 / N^4: O^2V^2
                + 0.5 * einsum("ijka,kb->ijab",
                               einsum("la,ijkl->ijka", ur1, hf.oooo), t1_2)
                # N^6: O^3V^3 / N^4: O^1V^3
                - 0.25 * einsum("ijkb,ka->ijab",
                                einsum("ijcd,kbcd->ijkb", t2_2, hf.ovvv), ur1)
                # N^5: O^3V^2 / N^4: O^2V^2
                - 0.25 * einsum("ijkb,ka->ijab",
                                einsum("ijkc,bc->ijkb", hf.ooov, p0_2_vv), ur1)
                # N^5: O^3V^2 / N^4: O^2V^2
                - 0.25 * einsum("la,ijlb->ijab",
                                einsum("ka,kl->la", ur1, p0_2_oo), hf.ooov)
                # N^7: O^4V^3 / N^6: O^3V^3
                - 0.25 * einsum("ijla,lb->ijab",
                                einsum("ijkacd,klcd->ijla", t3_2, hf.oovv), ur1)
                # N^6: O^3V^3 / N^4: O^1V^3
                - 1 / 16 * einsum("ijka,kb->ijab",
                                  einsum("ijcd,kacd->ijka", t2_1, t2eri_6), ur1)
                # N^5: O^3V^2 / N^4: O^2V^2
                - 1 / 8 * einsum("ijka,kb->ijab",
                                 einsum("kc,ijac->ijka",
                                        einsum("lmcd,lmkd->kc", t2_1, hf.ooov),
                                        t2_1), ur1)
            ).antisymmetrise(2, 3)
            + (
                # N^6: O^3V^3 / N^6: O^3V^3
                - 0.5 * einsum("kc,ijkabc->ijab",
                               einsum("ld,klcd->kc", ur1, hf.oovv), t3_2)
                # N^6: O^4V^2 / N^4: O^2V^2
                + 1 / 8 * einsum("ijlm,lmab->ijab",
                                 einsum("klmd,ijkd->ijlm",
                                        einsum("kc,lmcd->klmd", ur1, t2_1),
                                        hf.ooov), t2_1)
                # N^6: O^4V^2 / N^4: O^1V^3
                + 1 / 8 * einsum("ijkl,klab->ijab",
                                 einsum("ijde,klde->ijkl", t2_1, t2_1),
                                 einsum("kc,lcab->klab", ur1, hf.ovvv))
            )
            + 4 * (
                # N^6: O^3V^3 / N^4: O^1V^3
                - 0.25 * einsum("ikad,jbkd->ijab",
                                einsum("kc,icad->ikad", ur1, hf.ovvv), t2sq)
                # N^6: O^4V^2 / N^4: O^2V^2
                - 0.25 * einsum("jklb,ilka->ijab",
                                einsum("kc,jblc->jklb", ur1, t2sq), hf.ooov)
                # N^5: O^3V^2 / N^4: O^2V^2
                + 0.5 * einsum("ijka,kb->ijab",
                               einsum("ic,jcka->ijka", ur1, hf.ovov), t1_2)
                # N^6: O^3V^3 / N^4: O^1V^3
                + 0.5 * einsum("ikac,jkbc->ijab",
                               einsum("id,kacd->ikac", ur1, hf.ovvv), t2_2)
                # N^6: O^3V^3 / N^4: O^1V^3
                - 0.25 * einsum("ikbd,jkad->ijab",
                                einsum("ic,kbdc->ikbd", ur1, t2eri_7), t2_1)
                # N^5: O^2V^3 / N^4: O^1V^3
                - 0.25 * einsum("ijad,bd->ijab",
                                einsum("ic,jcad->ijad", ur1, hf.ovvv), p0_2_vv)
                # N^5: O^3V^2 / N^4: O^2V^2
                + 0.5 * einsum("ijkb,ka->ijab",
                               einsum("jc,ickb->ijkb", t1_2, hf.ovov), ur1)
                # N^6: O^4V^2 / N^4: O^2V^2
                + 0.5 * einsum("ijlb,la->ijab",
                               einsum("jkbc,klic->ijlb", t2_2, hf.ooov), ur1)
                # N^6: O^4V^2 / N^4: O^2V^2
                - 0.25 * einsum("ijkb,ka->ijab",
                                einsum("jlbc,lkic->ijkb", t2_1, t2eri_2), ur1)
                # N^5: O^3V^2 / N^4: O^2V^2
                - 0.25 * einsum("ijkb,ka->ijab",
                                einsum("jlkb,il->ijkb", hf.ooov, p0_2_oo), ur1)
            ).antisymmetrise(0, 1).antisymmetrise(2, 3)
        ))  # fmt: skip

    return AdcBlock(apply, 0)


def block_ph_ppphhh_3(hf, mp, intermediates):
    t2_1 = mp.t2(b.oovv)
    t2_2 = mp.td2(b.oovv)
    t3_2 = mp.tt2(b.ooovvv)

    def apply(ampl):
        ur3 = ampl.ppphhh
        return AmplitudeVector(ph=(
            # 2nd order
            # N^7: O^4V^3 / N^6: O^3V^3
            + 3 * einsum("ijld,jald->ia",
                         einsum("ijkbcd,klbc->ijld", ur3, t2_1), hf.ovov)
            # N^7: O^3V^4 / N^6: O^3V^3
            + 3 / 2 * einsum("ibde,aebd->ia",
                             einsum("ijkbcd,jkce->ibde", ur3, t2_1), hf.vvvv)
            # N^7: O^3V^4 / N^6: O^3V^3
            + 3 * einsum("kacd,ickd->ia",
                         einsum("jklabc,jlbd->kacd", ur3, t2_1), hf.ovov)
            # N^7: O^4V^3 / N^6: O^3V^3
            + 3 / 2 * einsum("jlma,imjl->ia",
                             einsum("jklabc,kmbc->jlma", ur3, t2_1), hf.oooo)
            # 3rd order
            # N^8: O^4V^4 / N^6: O^3V^3
            - 3 / 2 * einsum("ikde,kead->ia",
                             einsum("ijlbce,jklbcd->ikde", ur3, t3_2), hf.ovvv)
            - 0.5 * einsum("ijklbe,jklabe->ia",
                           einsum("jklbcd,iecd->ijklbe", t3_2, hf.ovvv), ur3)
            # N^8: O^5V^3 / N^6: O^3V^3
            - 3 / 2 * einsum("ijlmbc,jlmabc->ia",
                             einsum("jklbcd,ikmd->ijlmbc", t3_2, hf.ooov), ur3)
            - 0.5 * einsum("iklm,klma->ia",
                           einsum("ijmbcd,jklbcd->iklm", ur3, t3_2), hf.ooov)
            # N^7: O^3V^4 / N^6: O^3V^3
            - 3 / 2 * einsum("icde,acde->ia",
                             einsum("ijkbde,jkbc->icde", ur3, t2_2), hf.vvvv)
            + 3 / 4 * einsum("ilme,lmae->ia",
                             einsum("ibce,lmbc->ilme",
                                    einsum("ijkbcd,jkde->ibce", ur3, t2_1),
                                    t2_1), hf.oovv)
            + 0.5 * einsum("ce,ieac->ia",
                           einsum("jklbde,jklbcd->ce", ur3, t3_2), hf.ovvv)
            # N^7: O^4V^3 / N^6: O^3V^3
            - 3 * einsum("ijklbd,jklabd->ia",
                         einsum("jkbc,idlc->ijklbd", t2_2, hf.ovov), ur3)
            + 3 * einsum("ijld,jdla->ia",
                         einsum("iklbcd,jkbc->ijld", ur3, t2_2), hf.ovov)
            + 3 * einsum("ijklbc,jklabc->ia",
                         einsum("ilbd,jkcd->ijklbc",
                                einsum("lmbe,imde->ilbd", t2_1, hf.oovv),
                                t2_1), ur3)
            + 3 * einsum("ilme,lmae->ia",
                         einsum("ijmc,jlce->ilme",
                                einsum("ijkbcd,kmbd->ijmc", ur3, t2_1),
                                t2_1), hf.oovv)
            + 3 / 2 * einsum("jlma,ijlm->ia",
                             einsum("klmabc,jkbc->jlma", ur3, t2_2), hf.oooo)
            + 3 / 2 * einsum("me,imae->ia",
                             einsum("jkmc,jkce->me",
                                    einsum("jklbcd,lmbd->jkmc", ur3, t2_1),
                                    t2_1), hf.oovv)
            - 0.5 * einsum("lm,ilma->ia",
                           einsum("jkmbcd,jklbcd->lm", ur3, t3_2), hf.ooov)
            + 3 / 4 * einsum("ijkm,jkma->ia",
                             einsum("jkde,imde->ijkm", t2_1, hf.oovv),
                                    einsum("jklabc,lmbc->jkma", ur3, t2_1))
        ))  # fmt: skip

    return AdcBlock(apply, 0)


#
# 4th order main
#
def block_ph_ph_4(hf, mp, intermediates):
    m11 = intermediates.adc4_m11
    diagonal = AmplitudeVector(ph=einsum("iaia->ia", m11))

    def apply(ampl):
        return AmplitudeVector(ph=einsum("iajb,jb->ia", m11, ampl.ph))

    return AdcBlock(apply, diagonal)


#
# Intermediates
#


@register_as_intermediate
def adc2_i1(hf, mp, intermediates):
    # This definition differs from libadc. It additionally has the hf.fvv term.
    return hf.fvv + 0.5 * einsum("ijac,ijbc->ab", mp.t2oo, hf.oovv).symmetrise()


@register_as_intermediate
def adc2_i2(hf, mp, intermediates):
    # This definition differs from libadc. It additionally has the hf.foo term.
    return hf.foo - 0.5 * einsum("ikab,jkab->ij", mp.t2oo, hf.oovv).symmetrise()


def adc3_i1(hf, mp, intermediates):
    # Used for both CVS and general
    td2 = mp.td2(b.oovv)
    p0 = mp.second_order_dm_correction(apply_cvs=hf.has_core_occupied_space)

    t2eri_sum = (
        + einsum("jicb->ijcb", mp.t2eri(b.oovv, b.ov))  # t2eri4
        - 0.25 * mp.t2eri(b.oovv, b.vv)                 # t2eri5
    )  # fmt: skip
    return (
        (  # symmetrise a<>b
            + 0.5 * einsum("ijac,ijbc->ab", mp.t2oo + td2, hf.oovv)
            - 1.0 * einsum("ijac,ijcb->ab", mp.t2oo, t2eri_sum)
            - 2.0 * einsum("iabc,ic->ab", hf.ovvv, p0.ov)
        ).symmetrise()
        + einsum("iajb,ij->ab", hf.ovov, p0.oo)
        + einsum("acbd,cd->ab", hf.vvvv, p0.vv)
    )  # fmt: skip


def adc3_i2(hf, mp, intermediates):
    # Used only for general
    td2 = mp.td2(b.oovv)
    p0 = mp.second_order_dm_correction()

    # t2eri4 + t2eri3 / 4
    t2eri_sum = mp.t2eri(b.oovv, b.ov) + 0.25 * mp.t2eri(b.oovv, b.oo)
    return (
        (  # symmetrise i<>j
            + 0.5 * einsum("ikab,jkab->ij", mp.t2oo + td2, hf.oovv)
            - 1.0 * einsum("ikab,jkab->ij", mp.t2oo, t2eri_sum)
            + 2.0 * einsum("kija,ka->ij", hf.ooov, p0.ov)
        ).symmetrise()
        - einsum("ikjl,kl->ij", hf.oooo, p0.oo)
        - einsum("iajb,ab->ij", hf.ovov, p0.vv)
    )  # fmt: skip


def cvs_adc3_i2(hf, mp, intermediates):
    cvs_p0 = mp.second_order_dm_correction(apply_cvs=True)
    return (
        + 2.0 * einsum("kIJa,ka->IJ", hf.occv, cvs_p0.ov).symmetrise()
        - 1.0 * einsum("kIlJ,kl->IJ", hf.ococ, cvs_p0.oo)
        - 1.0 * einsum("IaJb,ab->IJ", hf.cvcv, cvs_p0.vv)
    )  # fmt: skip


@register_as_intermediate
def adc3_m11(hf, mp, intermediates):
    td2 = mp.td2(b.oovv)
    p0 = mp.second_order_dm_correction()

    i1 = adc3_i1(hf, mp, intermediates).evaluate()
    i2 = adc3_i2(hf, mp, intermediates).evaluate()
    t2sq = intermediates.t2sq

    # Build two Kronecker deltas
    d_oo = zeros_like(hf.foo)
    d_vv = zeros_like(hf.fvv)
    d_oo.set_mask("ii", 1.0)
    d_vv.set_mask("aa", 1.0)

    t2eri_sum = (
        + 2.0 * mp.t2eri(b.oovv, b.ov).symmetrise((0, 1), (2, 3))  # t2eri4
        + 0.5 * mp.t2eri(b.oovv, b.vv)                             # t2eri5
        + 0.5 * mp.t2eri(b.oovv, b.oo)                             # t2eri3
    )  # fmt: skip
    return (
        + einsum("ij,ab->iajb", d_oo, hf.fvv + i1)
        - einsum("ij,ab->iajb", hf.foo - i2, d_vv)
        - einsum("jaib->iajb", hf.ovov)
        - (  # symmetrise i<>j and a<>b
            + einsum("jkbc,ikac->iajb", hf.oovv, mp.t2oo + td2)
            - einsum("jkbc,ikac->iajb", mp.t2oo, t2eri_sum)
            - einsum("ibac,jc->iajb", hf.ovvv, 2.0 * p0.ov)
            - einsum("ikja,kb->iajb", hf.ooov, 2.0 * p0.ov)
            - einsum("jaic,bc->iajb", hf.ovov, p0.vv)
            + einsum("ik,jakb->iajb", p0.oo, hf.ovov)
            + einsum("ibkc,kajc->iajb", hf.ovov, 2.0 * t2sq)
        ).symmetrise((0, 2), (1, 3))
        # TODO This hack is done to avoid opt_einsum being smart and instantiating
        #      a tensor of dimension 6 (to avoid the vvvv tensor) in some cases,
        #      which is the right thing to do, but not yet supported.
        # + 0.5 * einsum("icjd,klac,klbd->iajb", hf.ovov, mp.t2oo, mp.t2oo)
        + 0.5 * einsum("icjd,acbd->iajb", hf.ovov,
                       einsum("klac,klbd->acbd", mp.t2oo, mp.t2oo))
        # + 0.5 * einsum("ikcd,jlcd,kalb->iajb", mp.t2oo, mp.t2oo, hf.ovov)
        + 0.5 * einsum("ikjl,kalb->iajb",
                       einsum("ikcd,jlcd->ikjl", mp.t2oo, mp.t2oo), hf.ovov)
        - einsum("iljk,kalb->iajb", hf.oooo, t2sq)
        - einsum("idjc,acbd->iajb", t2sq, hf.vvvv)
    )  # fmt: skip


@register_as_intermediate
def cvs_adc3_m11(hf, mp, intermediates):
    i1 = adc3_i1(hf, mp, intermediates).evaluate()
    i2 = cvs_adc3_i2(hf, mp, intermediates).evaluate()
    t2sq = intermediates.t2sq
    cvs_p0 = mp.second_order_dm_correction(apply_cvs=True)

    # Build two Kronecker deltas
    d_cc = zeros_like(hf.fcc)
    d_vv = zeros_like(hf.fvv)
    d_cc.set_mask("II", 1.0)
    d_vv.set_mask("aa", 1.0)

    return (
        + einsum("IJ,ab->IaJb", d_cc, hf.fvv + i1)
        - einsum("IJ,ab->IaJb", hf.fcc - i2, d_vv)
        - einsum("JaIb->IaJb", hf.cvcv)
        + (  # symmetrise I<>J and a<>b
            + einsum("JaIc,bc->IaJb", hf.cvcv, cvs_p0.vv)
            - einsum("kIJa,kb->IaJb", hf.occv, 2.0 * cvs_p0.ov)
        ).symmetrise((0, 2), (1, 3))
        # TODO This hack is done to avoid opt_einsum being smart and instantiating
        #      a tensor of dimension 6 (to avoid the vvvv tensor) in some cases,
        #      which is the right thing to do, but not yet supported.
        # + 0.5 * einsum("IcJd,klac,klbd->IaJb", hf.cvcv, mp.t2oo, mp.t2oo)
        + 0.5 * einsum("IcJd,acbd->IaJb", hf.cvcv,
                       einsum("klac,klbd->acbd", mp.t2oo, mp.t2oo))
        - einsum("lIkJ,kalb->IaJb", hf.ococ, t2sq)
    )  # fmt: skip


@register_as_intermediate
def adc3_pia(hf, mp, intermediates):
    # This definition differs from libadc. It additionally has the hf.ooov term.
    return (                          # Perturbation theory in ADC coupling block
        + hf.ooov                                            # 1st order
        - 2.0 * mp.t2eri(b.ooov, b.ov).antisymmetrise(0, 1)  # 2nd order
        - 0.5 * mp.t2eri(b.ooov, b.vv)                       # 2nd order
    )  # fmt: skip


@register_as_intermediate
def cvs_adc3_pia(hf, mp, intermediates):
    # Perturbation theory in CVS-ADC coupling block:
    #       1st                     2nd
    return hf.occv - einsum("jlac,lKIc->jIKa", mp.t2oo, hf.occv)


@register_as_intermediate
def adc3_pib(hf, mp, intermediates):
    # This definition differs from libadc. It additionally has the hf.ovvv term.
    return (                          # Perturbation theory in ADC coupling block
        + hf.ovvv                                            # 1st order
        + 2.0 * mp.t2eri(b.ovvv, b.ov).antisymmetrise(2, 3)  # 2nd order
        - 0.5 * mp.t2eri(b.ovvv, b.oo)                       # 2nd order
    )  # fmt: skip


@register_as_intermediate
def adc4_m11(hf, mp, intermediates):
    p0_2 = mp.mp2_diffdm
    p0_2_oo, p0_2_vv, t1_2 = p0_2.oo, p0_2.vv, p0_2.ov

    t2_1 = mp.t2(b.oovv)
    t2_2 = mp.td2(b.oovv)
    t2_3 = mp.td3(b.oovv)
    t3_2 = mp.tt2(b.ooovvv)

    # MP3 diffdm
    p0 = mp.diffdm(3)
    p0_vv, p0_ov, p0_oo = p0.vv, p0.ov, p0.oo

    t2sq = intermediates.t2sq
    t2eri_1 = mp.t2eri(b.ooov, b.vv).evaluate()
    t2eri_2 = mp.t2eri(b.ooov, b.ov).evaluate()
    t2eri_3 = mp.t2eri(b.oovv, b.oo).evaluate()
    t2eri_4 = mp.t2eri(b.oovv, b.ov).evaluate()
    t2eri_5 = mp.t2eri(b.oovv, b.vv).evaluate()
    t2eri_6 = mp.t2eri(b.ovvv, b.oo).evaluate()
    t2eri_7 = mp.t2eri(b.ovvv, b.ov).evaluate()

    # Build two Kronecker deltas
    d_oo = zeros_like(hf.foo)
    d_vv = zeros_like(hf.fvv)
    d_oo.set_mask("ii", 1.0)
    d_vv.set_mask("aa", 1.0)
    return (
        # 0th order
        # N^4: O^2V^2 / N^4: O^2V^2
        - 1 * einsum("ij,ab->iajb", hf.foo, d_vv)
        # N^4: O^2V^2 / N^4: O^2V^2
        + 1 * einsum("ab,ij->iajb", hf.fvv, d_oo)
        # 1st order
        # N^4: O^2V^2 / N^4: O^2V^2
        - 1 * einsum("ibja->iajb", hf.ovov)
        # 2nd order
        + 2 * (
            # N^5: O^3V^2 / N^4: O^2V^2
            + 0.25 * einsum("ij,ab->iajb",
                            einsum("ikcd,jkcd->ij", t2_1, hf.oovv), d_vv)
            # N^5: O^2V^3 / N^4: O^2V^2
            + 0.25 * einsum("ab,ij->iajb",
                            einsum("klac,klbc->ab", t2_1, hf.oovv), d_oo)
            # N^6: O^3V^3 / N^4: O^2V^2
            - 0.5 * einsum("ikac,jkbc->iajb", t2_1, hf.oovv)
        ).symmetrise((0, 2), (1, 3))
        # 3rd order
        # N^6: O^2V^4 / N^4: V^4
        - 1 * einsum("adbc,icjd->iajb", hf.vvvv, t2sq)
        # N^6: O^4V^2 / N^4: O^2V^2
        - 1 * einsum("iljk,kalb->iajb", hf.oooo, t2sq)
        # N^6: O^2V^4 / N^4: V^4
        + 0.5 * einsum("abcd,icjd->iajb",
                       einsum("klac,klbd->abcd", t2_1, t2_1), hf.ovov)
        # N^6: O^4V^2 / N^4: O^2V^2
        + 0.5 * einsum("ijkl,kalb->iajb",
                       einsum("ikcd,jlcd->ijkl", t2_1, t2_1), hf.ovov)
        + 2 * (
            # N^5: O^3V^2 / N^4: O^2V^2
            + 0.5 * einsum("ij,ab->iajb",
                           einsum("ikcd,jkdc->ij", t2_1, t2eri_4), d_vv)
            # N^5: O^3V^2 / N^4: O^2V^2
            - 1 / 8 * einsum("ij,ab->iajb",
                             einsum("ikcd,jkcd->ij", t2_1, t2eri_3), d_vv)
            # N^5: O^3V^2 / N^4: O^2V^2
            + 0.25 * einsum("ij,ab->iajb",
                            einsum("ikcd,jkcd->ij", t2_2, hf.oovv), d_vv)
            # N^5: O^2V^3 / N^4: O^2V^2
            + 0.5 * einsum("ab,ij->iajb",
                           einsum("klac,klcb->ab", t2_1, t2eri_4), d_oo)
            # N^5: O^2V^3 / N^4: O^2V^2
            - 1 / 8 * einsum("ab,ij->iajb",
                             einsum("klac,klbc->ab", t2_1, t2eri_5), d_oo)
            # N^5: O^2V^3 / N^4: O^2V^2
            + 0.25 * einsum("ab,ij->iajb",
                            einsum("klac,klbc->ab", t2_2, hf.oovv), d_oo)
            # N^6: O^3V^3 / N^4: O^2V^2
            + 0.5 * einsum("ikac,jkbc->iajb", t2_1, t2eri_4)
            # N^6: O^3V^3 / N^4: O^2V^2
            + 0.5 * einsum("ikac,kjcb->iajb", t2_1, t2eri_4)
            # N^6: O^3V^3 / N^4: O^2V^2
            - 1 * einsum("ibkc,jcka->iajb", hf.ovov, t2sq)
            # N^6: O^3V^3 / N^4: O^2V^2
            - 0.5 * einsum("ikac,jkbc->iajb", t2_2, hf.oovv)
            # N^6: O^3V^3 / N^4: O^2V^2
            + 0.25 * einsum("ikac,jkbc->iajb", t2_1, t2eri_3)
            # N^6: O^3V^3 / N^4: O^2V^2
            + 0.25 * einsum("ikac,jkbc->iajb", t2_1, t2eri_5)
        ).symmetrise((0, 2), (1, 3))
        # 4th order
        + 2.0 * (
            # N^4: O^2V^2 / N^4: O^2V^2
            + 0.5 * einsum("ij,ab->iajb",
                           einsum("kc,ikjc->ij", t1_2, t2eri_2), d_vv)
            # N^4: O^2V^2 / N^4: O^2V^2
            - 1 * einsum("ij,ab->iajb",
                         einsum("ikjc,kc->ij", hf.ooov, p0_ov), d_vv)
            # N^5: O^3V^2 / N^4: O^2V^2
            - 0.5 * einsum("ij,ab->iajb",
                           einsum("ikcd,jkcd->ij", t2_2, t2eri_4), d_vv)
            # N^5: O^3V^2 / N^4: O^2V^2
            - 1 / 8 * einsum("ij,ab->iajb",
                             einsum("ikcd,jkcd->ij", t2_2, t2eri_3), d_vv)
            # N^4: O^2V^2 / N^4: O^2V^2
            + 0.25 * einsum("ij,ab->iajb",
                            einsum("kc,ikjc->ij", t1_2, t2eri_1), d_vv)
            # N^5: O^3V^2 / N^4: O^2V^2
            + 0.25 * einsum("ij,ab->iajb",
                            einsum("ikcd,jkcd->ij", t2_3, hf.oovv), d_vv)
            # N^5: O^2V^3 / N^4: O^2V^2
            + 0.5 * einsum("ij,ab->iajb",
                           einsum("ikde,jkde->ij",
                                  einsum("ikcd,ce->ikde", t2_1, p0_2_vv),
                                  hf.oovv), d_vv)
            # N^6: O^3V^3 / N^4: O^2V^2
            - 0.5 * einsum("ij,ab->iajb",
                           einsum("ilde,jdle->ij",
                                  einsum("ikce,klcd->ilde", t2_1, t2_2),
                                  hf.ovov), d_vv)
            # N^6: O^3V^3 / N^4: O^2V^2
            - 0.5 * einsum("ij,ab->iajb",
                           einsum("ilde,jdle->ij",
                                  einsum("klce,ikcd->ilde", t2_1, hf.oovv),
                                  t2sq), d_vv)
            # N^5: O^3V^2 / N^4: O^2V^2
            - 0.25 * einsum("ij,ab->iajb",
                            einsum("jc,ic->ij",
                                   einsum("klcd,kljd->jc", t2_1, hf.ooov),
                                   t1_2), d_vv)
            # N^7: O^4V^3 / N^6: O^3V^3
            - 0.25 * einsum("ij,ab->iajb",
                            einsum("jlmd,ilmd->ij",
                                   einsum("kmce,jklcde->jlmd", t2_1, t3_2),
                                   hf.ooov), d_vv)
            # N^6: O^4V^2 / N^4: O^2V^2
            - 1 / 8 * einsum("ij,ab->iajb",
                             einsum("iklm,jmkl->ij",
                                    einsum("imcd,klcd->iklm", t2_1, t2_2),
                                    hf.oooo), d_vv)
            # N^7: O^4V^3 / N^6: O^3V^3
            - 1 / 8 * einsum("ij,ab->iajb",
                             einsum("ilmd,lmjd->ij",
                                    einsum("ikce,klmcde->ilmd", t2_1, t3_2),
                                    hf.ooov), d_vv)
            # N^7: O^3V^4 / N^6: O^3V^3
            - 1 / 8 * einsum("ij,ab->iajb",
                             einsum("idef,jfde->ij",
                                    einsum("klcf,iklcde->idef", t2_1, t3_2),
                                    hf.ovvv), d_vv)
            # N^5: O^3V^2 / N^4: O^2V^2
            - 1 / 16 * einsum("ij,ab->iajb",
                              einsum("jk,ik->ij",
                                     einsum("jlcd,klcd->jk", t2_1, hf.oovv),
                                     p0_2_oo), d_vv)
            # N^5: O^3V^2 / N^4: O^2V^2
            - 1 / 16 * einsum("ij,ab->iajb",
                              einsum("il,jl->ij",
                                     einsum("klcd,ikcd->il", t2_1, hf.oovv),
                                     p0_2_oo), d_vv)
            # N^5: O^3V^2 / N^4: O^2V^2
            + 0.25 * einsum("ij,ab->iajb",
                            einsum("ilcd,jlcd->ij",
                                   einsum("ikcd,kl->ilcd", t2_1, p0_2_oo),
                                   hf.oovv), d_vv)
            # N^6: O^4V^2 / N^4: O^2V^2
            + 1 / 16 * einsum("ij,ab->iajb",
                              einsum("jklm,iklm->ij",
                                     einsum("lmef,jkef->jklm", t2_1, hf.oovv),
                                     einsum("ikcd,lmcd->iklm", t2_1, t2_1)), d_vv)
            # N^4: O^1V^3 / N^4: O^1V^3
            - 1 * einsum("ab,ij->iajb",
                         einsum("kabc,kc->ab", hf.ovvv, p0_ov), d_oo)
            # N^4: O^1V^3 / N^4: O^1V^3
            - 0.5 * einsum("ab,ij->iajb",
                           einsum("kc,kabc->ab", t1_2, t2eri_7), d_oo)
            # N^5: O^2V^3 / N^4: O^2V^2
            - 0.5 * einsum("ab,ij->iajb",
                           einsum("klac,lkcb->ab", t2_2, t2eri_4), d_oo)
            # N^5: O^2V^3 / N^4: O^2V^2
            - 1 / 8 * einsum("ab,ij->iajb",
                             einsum("klac,klbc->ab", t2_2, t2eri_5), d_oo)
            # N^4: O^1V^3 / N^4: O^1V^3
            + 0.25 * einsum("ab,ij->iajb",
                            einsum("kc,kabc->ab", t1_2, t2eri_6), d_oo)
            # N^5: O^2V^3 / N^4: O^2V^2
            + 0.25 * einsum("ab,ij->iajb",
                            einsum("klac,klbc->ab", t2_3, hf.oovv), d_oo)
            # N^5: O^2V^3 / N^4: O^2V^2
            - 0.5 * einsum("ab,ij->iajb",
                           einsum("lmac,lmbc->ab",
                                  einsum("klac,km->lmac", t2_1, p0_2_oo),
                                  hf.oovv), d_oo)
            # N^6: O^3V^3 / N^4: O^2V^2
            - 0.5 * einsum("ab,ij->iajb",
                           einsum("lmad,lmbd->ab",
                                  einsum("klac,kcmd->lmad", t2_1, t2sq),
                                  hf.oovv), d_oo)
            # N^6: O^3V^3 / N^4: O^2V^2
            - 0.5 * einsum("ab,ij->iajb",
                           einsum("lmad,lbmd->ab",
                                  einsum("kmac,klcd->lmad", t2_1, t2_2),
                                  hf.ovov), d_oo)
            # N^5: O^2V^3 / N^4: O^2V^2
            - 0.25 * einsum("ab,ij->iajb",
                            einsum("klad,klbd->ab",
                                   einsum("klac,cd->klad", t2_1, p0_2_vv),
                                   hf.oovv), d_oo)
            # N^5: O^2V^3 / N^4: O^1V^3
            - 0.25 * einsum("ab,ij->iajb",
                            einsum("kb,ka->ab",
                                   einsum("klcd,lbcd->kb", t2_1, hf.ovvv),
                                   t1_2), d_oo)
            # N^7: O^3V^4 / N^6: O^3V^3
            - 1 / 8 * einsum("ab,ij->iajb",
                             einsum("klbd,klad->ab",
                                    einsum("klmcde,mbce->klbd", t3_2, hf.ovvv),
                                    t2_1), d_oo)
            # N^6: O^2V^4 / N^4: V^4
            - 1 / 8 * einsum("ab,ij->iajb",
                             einsum("klbe,klae->ab",
                                    einsum("klcd,becd->klbe", t2_2, hf.vvvv),
                                    t2_1), d_oo)
            # N^7: O^4V^3 / N^6: O^3V^3
            - 1 / 8 * einsum("ab,ij->iajb",
                             einsum("klna,klnb->ab",
                                    einsum("mncd,klmacd->klna", t2_1, t3_2),
                                    hf.ooov), d_oo)
            # N^5: O^2V^3 / N^4: O^2V^2
            - 1 / 16 * einsum("ab,ij->iajb",
                              einsum("ad,bd->ab",
                                     einsum("klac,klcd->ad", t2_1, hf.oovv),
                                     p0_2_vv), d_oo)
            # N^5: O^2V^3 / N^4: O^2V^2
            - 1 / 16 * einsum("ab,ij->iajb",
                              einsum("bc,ac->ab",
                                     einsum("klcd,klbd->bc", t2_1, hf.oovv),
                                     p0_2_vv), d_oo)
            # N^7: O^3V^4 / N^6: O^3V^3
            + 0.25 * einsum("ab,ij->iajb",
                            einsum("kade,kebd->ab",
                                   einsum("lmce,klmacd->kade", t2_1, t3_2),
                                   hf.ovvv), d_oo)
            # N^6: O^4V^2 / N^4: O^2V^2
            + 1 / 16 * einsum("ab,ij->iajb",
                              einsum("mnac,mnbc->ab",
                                     einsum("klmn,klac->mnac",
                                            einsum("klde,mnde->klmn",
                                                   t2_1, t2_1), t2_1),
                                     hf.oovv), d_oo)
            # N^5: O^3V^2 / N^4: O^2V^2
            + 1 * einsum("ka,kjib->iajb", t1_2, t2eri_2)
            # N^5: O^2V^3 / N^4: O^1V^3
            + 1 * einsum("ibac,jc->iajb", hf.ovvv, p0_ov)
            # N^5: O^3V^2 / N^4: O^2V^2
            + 1 * einsum("ikja,kb->iajb", hf.ooov, p0_ov)
            # N^5: O^2V^3 / N^4: O^1V^3
            + 0.5 * einsum("ic,jabc->iajb", t1_2, t2eri_7)
            # N^6: O^3V^3 / N^4: O^2V^2
            + 0.5 * einsum("ikac,jkbc->iajb", t2_2, t2eri_4)
            # N^6: O^3V^3 / N^4: O^2V^2
            + 0.5 * einsum("ikac,kjcb->iajb", t2_2, t2eri_4)
            # N^5: O^2V^3 / N^4: O^2V^2
            + 0.5 * einsum("ibjc,ac->iajb", hf.ovov, p0_vv)
            # N^5: O^2V^3 / N^4: O^1V^3
            - 1 * einsum("ic,jacb->iajb", t1_2, t2eri_7)
            # N^5: O^3V^2 / N^4: O^2V^2
            - 0.5 * einsum("ka,jkib->iajb", t1_2, t2eri_1)
            # N^5: O^3V^2 / N^4: O^2V^2
            - 0.5 * einsum("ka,jkib->iajb", t1_2, t2eri_2)
            # N^5: O^2V^3 / N^4: O^1V^3
            - 0.5 * einsum("ic,jabc->iajb", t1_2, t2eri_6)
            # N^6: O^3V^3 / N^4: O^2V^2
            - 0.5 * einsum("ikac,jkbc->iajb", t2_3, hf.oovv)
            # N^5: O^3V^2 / N^4: O^2V^2
            - 0.5 * einsum("ibka,jk->iajb", hf.ovov, p0_oo)
            # N^6: O^3V^3 / N^4: O^2V^2
            + 0.25 * einsum("ikac,jkbc->iajb", t2_2, t2eri_3)
            # N^6: O^3V^3 / N^4: O^2V^2
            + 0.25 * einsum("ikac,jkbc->iajb", t2_2, t2eri_5)
            # N^6: O^3V^3 / N^4: O^2V^2
            + 1 * einsum("kmab,ikjm->iajb",
                         einsum("lmac,klbc->kmab", t2_1, t2_2), hf.oooo)
            # N^6: O^3V^3 / N^4: O^2V^2
            + 1 * einsum("jkad,ibkd->iajb",
                         einsum("jlcd,klac->jkad", t2_1, t2_2), hf.ovov)
            # N^6: O^2V^4 / N^4: V^4
            + 1 * einsum("ijce,aebc->iajb",
                         einsum("jkde,ikcd->ijce", t2_1, t2_2), hf.vvvv)
            # N^6: O^3V^3 / N^4: O^2V^2
            + 0.5 * einsum("ikad,jkbd->iajb",
                           einsum("ikac,cd->ikad", t2_1, p0_2_vv), hf.oovv)
            # N^6: O^3V^3 / N^4: O^2V^2
            + 0.5 * einsum("jkbc,ikac->iajb",
                           einsum("jlcd,kbld->jkbc", hf.oovv, t2sq), t2_1)
            # N^6: O^3V^3 / N^4: O^2V^2
            + 0.5 * einsum("jkbc,ikac->iajb",
                           einsum("klbd,jcld->jkbc", hf.oovv, t2sq), t2_1)
            # N^6: O^3V^3 / N^4: O^2V^2
            + 0.5 * einsum("jkbd,ikad->iajb",
                           einsum("klbc,jcld->jkbd", t2_2, hf.ovov), t2_1)
            # N^6: O^2V^4 / N^4: V^4
            + 0.5 * einsum("abcd,idjc->iajb",
                           einsum("klad,klbc->abcd", t2_1, t2_2), hf.ovov)
            # N^6: O^3V^3 / N^4: O^2V^2
            + 0.5 * einsum("ikac,jkbc->iajb",
                           einsum("klad,icld->ikac", t2_1, t2sq), hf.oovv)
            # N^8: O^5V^3 / N^6: O^3V^3
            + 0.5 * einsum("ijkmbd,kmad->iajb",
                           einsum("jklbcd,ilmc->ijkmbd", t3_2, hf.ooov), t2_1)
            # N^6: O^3V^3 / N^4: O^2V^2
            + 0.5 * einsum("ikbd,jdka->iajb",
                           einsum("klbc,ilcd->ikbd", t2_1, hf.oovv), t2sq)
            # N^7: O^3V^4 / N^6: O^3V^3
            + 0.5 * einsum("kabc,jkic->iajb",
                           einsum("lmbd,klmacd->kabc", t2_1, t3_2), hf.ooov)
            # N^6: O^4V^2 / N^4: O^2V^2
            + 0.5 * einsum("ijkl,kbla->iajb",
                           einsum("ilcd,jkcd->ijkl", t2_1, t2_2), hf.ovov)
            # N^6: O^3V^3 / N^4: O^2V^2
            + 0.5 * einsum("ikbd,jdka->iajb",
                           einsum("ilcd,klbc->ikbd", t2_1, hf.oovv), t2sq)
            # N^7: O^4V^3 / N^6: O^3V^3
            + 0.5 * einsum("ijld,lbad->iajb",
                           einsum("ikce,jklcde->ijld", t2_1, t3_2), hf.ovvv)
            # N^8: O^4V^4 / N^6: O^3V^3
            + 0.5 * einsum("ijlbce,leac->iajb",
                           einsum("ikde,jklbcd->ijlbce", t2_1, t3_2), hf.ovvv)
            # N^6: O^3V^3 / N^4: O^2V^2
            - 1 * einsum("jlad,ibld->iajb",
                         einsum("klac,jkcd->jlad", t2_1, t2_2), hf.ovov)
            # N^6: O^3V^3 / N^4: O^2V^2
            - 3 / 8 * einsum("ijad,bd->iajb",
                             einsum("ikac,jkcd->ijad", t2_1, hf.oovv), p0_2_vv)
            # N^6: O^3V^3 / N^4: O^2V^2
            - 0.5 * einsum("ilac,jlbc->iajb",
                           einsum("ikac,kl->ilac", t2_1, p0_2_oo), hf.oovv)
            # N^6: O^4V^2 / N^4: O^2V^2
            - 0.5 * einsum("ikla,jlkb->iajb",
                           einsum("ilac,kc->ikla", t2_1, t1_2), hf.ooov)
            # N^6: O^3V^3 / N^4: O^1V^3
            - 0.5 * einsum("jkbd,ikad->iajb",
                           einsum("kc,jcbd->jkbd", t1_2, hf.ovvv), t2_1)
            # N^6: O^3V^3 / N^4: O^2V^2
            - 0.5 * einsum("jlbd,ilad->iajb",
                           einsum("jkcd,kblc->jlbd", t2_2, hf.ovov), t2_1)
            # N^7: O^4V^3 / N^6: O^3V^3
            - 0.25 * einsum("jkbc,ikac->iajb",
                            einsum("klmbcd,lmjd->jkbc", t3_2, hf.ooov), t2_1)
            # N^7: O^4V^3 / N^6: O^3V^3
            - 0.25 * einsum("iklb,klja->iajb",
                            einsum("imcd,klmbcd->iklb", t2_1, t3_2), hf.ooov)
            # N^7: O^4V^3 / N^6: O^3V^3
            - 0.25 * einsum("ilma,jlmb->iajb",
                            einsum("kmcd,iklacd->ilma", t2_1, t3_2), hf.ooov)
            # N^6: O^3V^3 / N^4: O^2V^2
            - 1 / 8 * einsum("jkbd,iakd->iajb",
                             einsum("jlbc,klcd->jkbd", t2_1, hf.oovv), t2sq)
            # N^6: O^3V^3 / N^4: O^2V^2
            - 1 / 8 * einsum("ijbd,ad->iajb",
                             einsum("ikcd,jkbc->ijbd", t2_1, hf.oovv), p0_2_vv)
            # N^6: O^3V^3 / N^4: O^2V^2
            - 1 / 8 * einsum("ilad,jbld->iajb",
                             einsum("klcd,ikac->ilad", t2_1, hf.oovv), t2sq)
            # N^5: O^3V^2 / N^4: O^2V^2
            - 1 / 16 * einsum("il,jbla->iajb",
                              einsum("klcd,ikcd->il", t2_1, hf.oovv), t2sq)
            # N^5: O^2V^3 / N^4: O^2V^2
            - 1 / 16 * einsum("ad,idjb->iajb",
                              einsum("klcd,klac->ad", t2_1, hf.oovv), t2sq)
            # N^6: O^3V^3 / N^4: O^2V^2
            + 0.25 * einsum("jmbc,imac->iajb",
                            einsum("klbc,jmkl->jmbc", t2_2, hf.oooo), t2_1)
            # N^7: O^3V^4 / N^6: O^3V^3
            + 0.25 * einsum("jkbd,ikad->iajb",
                            einsum("jklcde,lbce->jkbd", t3_2, hf.ovvv), t2_1)
            # N^7: O^3V^4 / N^6: O^3V^3
            + 0.25 * einsum("jace,ibce->iajb",
                            einsum("klad,jklcde->jace", t2_1, t3_2), hf.ovvv)
            # N^6: O^2V^4 / N^4: V^4
            + 0.25 * einsum("abcd,icjd->iajb",
                            einsum("klad,klbc->abcd", t2_1, hf.oovv), t2sq)
            # N^6: O^2V^4 / N^4: V^4
            + 0.25 * einsum("jkbe,ikae->iajb",
                            einsum("jkcd,becd->jkbe", t2_2, hf.vvvv), t2_1)
            # N^6: O^4V^2 / N^4: O^2V^2
            + 0.25 * einsum("ijkl,kalb->iajb",
                            einsum("ilcd,jkcd->ijkl", t2_1, hf.oovv), t2sq)
            # N^7: O^3V^4 / N^6: O^3V^3
            + 0.25 * einsum("iace,jebc->iajb",
                            einsum("klde,iklacd->iace", t2_1, t3_2), hf.ovvv)
            # N^6: O^3V^3 / N^4: O^2V^2
            + 1 / 8 * einsum("jlab,il->iajb",
                             einsum("klac,jkbc->jlab", t2_1, hf.oovv), p0_2_oo)
            # N^8: O^4V^4 / N^6: O^3V^3
            + 1 / 8 * einsum("ijklbe,klae->iajb",
                             einsum("jklbcd,iecd->ijklbe", t3_2, hf.ovvv), t2_1)
            # N^8: O^5V^3 / N^6: O^3V^3
            + 1 / 8 * einsum("ijklmb,klma->iajb",
                             einsum("imcd,jklbcd->ijklmb", t2_1, t3_2), hf.ooov)
            # N^5: O^2V^3 / N^4: O^2V^2
            + 1 / 16 * einsum("ad,idjb->iajb",
                              einsum("klac,klcd->ad", t2_1, hf.oovv), t2sq)
            # N^5: O^3V^2 / N^4: O^2V^2
            + 1 / 16 * einsum("il,jbla->iajb",
                              einsum("ikcd,klcd->il", t2_1, hf.oovv), t2sq)
            # N^6: O^3V^3 / N^4: O^2V^2
            + 3 / 8 * einsum("ilab,jl->iajb",
                             einsum("ikac,klbc->ilab", t2_1, hf.oovv), p0_2_oo)
            # N^6: O^3V^3 / N^4: O^2V^2
            - 1 / 8 * einsum("jkbc,ikac->iajb",
                             einsum("jklm,lmbc->jkbc",
                                    einsum("lmde,jkde->jklm", t2_1, hf.oovv),
                                    t2_1), t2_1)
            # N^6: O^3V^3 / N^4: O^2V^2
            - 1 / 8 * einsum("jkbc,ikac->iajb",
                             einsum("jklm,lmbc->jkbc",
                                    einsum("jkde,lmde->jklm", t2_1, t2_1),
                                    hf.oovv), t2_1)
            # N^6: O^3V^3 / N^4: O^2V^2
            - 1 / 8 * einsum("ikac,jkbc->iajb",
                             einsum("iklm,lmac->ikac",
                                    einsum("ikde,lmde->iklm", t2_1, t2_1),
                                    t2_1), hf.oovv)
            # N^6: O^2V^4 / N^4: V^4
            + 0.25 * einsum("ijde,abde->iajb",
                            einsum("imcd,jmce->ijde", t2_1, hf.oovv),
                            einsum("klae,klbd->abde", t2_1, t2_1))
            # N^6: O^3V^3 / N^4: O^2V^2
            + 0.25 * einsum("ijlm,lmab->iajb",
                            einsum("imcd,jlcd->ijlm", t2_1, t2_1),
                            einsum("klae,kmbe->lmab", t2_1, hf.oovv))
        ).symmetrise((0, 2), (1, 3))
        + (
            # N^4: O^2V^2 / N^4: O^2V^2
            - 1 * einsum("ij,ab->iajb",
                         einsum("idjc,cd->ij", hf.ovov, p0_vv), d_vv)
            # N^4: O^2V^2 / N^4: O^2V^2
            - 1 * einsum("ij,ab->iajb",
                         einsum("iljk,kl->ij", hf.oooo, p0_oo), d_vv)
            # N^4: V^4 / N^4: V^4
            + 1 * einsum("ab,ij->iajb",
                         einsum("adbc,cd->ab", hf.vvvv, p0_vv), d_oo)
            # N^4: O^2V^2 / N^4: O^2V^2
            + 1 * einsum("ab,ij->iajb",
                         einsum("kbla,kl->ab", hf.ovov, p0_oo), d_oo)
        )
    )  # fmt: skip


@register_as_intermediate
def t2sq(hf, mp, intermediates):
    return einsum("ikac,jkbc->iajb", mp.t2oo, mp.t2oo)


@register_as_intermediate
def df3(hf, mp, intermediates):
    return direct_sum(
        "-i-j-k+a+b+c->ijkabc",
        hf.foo.diagonal(), hf.foo.diagonal(), hf.foo.diagonal(),
        hf.fvv.diagonal(), hf.fvv.diagonal(), hf.fvv.diagonal()
    ).symmetrise(0, 1, 2).symmetrise(3, 4, 5)  # fmt: skip
