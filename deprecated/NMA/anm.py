# **************************************************************************
# *
# * Authors:  David Herreros Calero (dherreros@cnb.csic.es)
# *
# * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# * This program is distributed in the hope that it will be useful,
# * but WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# * GNU General Public License for more details.
# *
# * You should have received a copy of the GNU General Public License
# * along with this program; if not, write to the Free Software
# * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# * 02111-1307  USA
# *
# *  All comments concerning this program package may be sent to the
# *  e-mail address 'scipion@cnb.csic.es'
# *
# **************************************************************************


from prody.dynamics.anm import ANMBase
from prody.utilities.logger import LOGGER
from .eigtools import solveEig


class ANMBaseFlex(ANMBase):

    def calcModes(self, n_modes=20, zeros=False, turbo=True, mode="cupy"):
        """Calculate normal modes.  This method uses :func:`scipy.linalg.eigh`
        function to diagonalize the Hessian matrix. When Scipy is not found,
        :func:`numpy.linalg.eigh` is used.

        :arg n_modes: number of non-zero eigenvalues/vectors to calculate.
            If **None** or ``'all'`` is given, all modes will be calculated.
        :type n_modes: int or None, default is 20

        :arg zeros: If **True**, modes with zero eigenvalues will be kept.
        :type zeros: bool, default is **True**

        :arg turbo: Use a memory intensive, but faster way to calculate modes.
        :type turbo: bool, default is **True**
        """

        if self._hessian is None:
            raise ValueError('Hessian matrix is not built or set')
        if str(n_modes).lower() == 'all':
            n_modes = None
        assert n_modes is None or isinstance(n_modes, int) and n_modes > 0, \
            'n_modes must be a positive integer'
        assert isinstance(zeros, bool), 'zeros must be a boolean'
        assert isinstance(turbo, bool), 'turbo must be a boolean'
        self._clear()
        LOGGER.timeit('_anm_calc_modes')
        values, vectors, vars = solveEig(self._hessian, n_modes=n_modes, zeros=zeros,
                                         turbo=turbo, expct_n_zeros=6, mode=mode)
        self._eigvals = values
        self._array = vectors
        self._vars = vars
        self._trace = self._vars.sum()

        self._n_modes = len(self._eigvals)
        if self._n_modes > 1:
            LOGGER.report('{0} modes were calculated in %.2fs.'
                          .format(self._n_modes), label='_anm_calc_modes')
        else:
            LOGGER.report('{0} mode was calculated in %.2fs.'
                          .format(self._n_modes), label='_anm_calc_modes')


class ANM(ANMBaseFlex):

    def __init__(self, name='Unknown'):
        super(ANM, self).__init__(name)
