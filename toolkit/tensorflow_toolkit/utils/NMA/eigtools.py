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


"""This module defines utility functions for solving eigenvalues."""
import numpy as np
import re
from prody.utilities.misctools import div0
from prody.utilities.logger import LOGGER
from .misctools import importLA

__all__ = ['solveEig', 'ZERO']

ZERO = 1e-6

def solveEig(M, n_modes=None, zeros=False, turbo=True, expct_n_zeros=None, reverse=False, mode="cupy"):
    linalg = importLA(mode=mode)
    dof = M.shape[0]

    if expct_n_zeros is None:
        expct_n_zeros = 0
        warn_zeros = False
    else:
        warn_zeros = True

    if n_modes is None:
        eigvals = None
        n_modes = dof
    else:
        if n_modes >= dof:
            eigvals = None
            n_modes = dof
        else:
            if reverse:
                eigvals = (dof-expct_n_zeros-n_modes, dof-1)
            else:
                eigvals = (0, n_modes+expct_n_zeros-1)

    def _eigh(M, eigvals=None, turbo=True):
        if linalg.__package__.startswith('scipy'):
            from scipy.sparse import issparse

            if eigvals:
                turbo = False
            if not issparse(M):
                values, vectors = linalg.eigh(M, turbo=turbo, eigvals=eigvals)
            else:
                try:
                    from scipy.sparse import linalg as scipy_sparse_la
                except ImportError:
                    raise ImportError('failed to import scipy.sparse.linalg, '
                                      'which is required for sparse matrix '
                                      'decomposition')
                if eigvals:
                    j = eigvals[0]
                    k = eigvals[-1] + 1
                else:
                    j = 0
                    k = dof

                if k >= dof:
                    k -= 1
                    LOGGER.warning('Cannot calculate all eigenvalues for sparse matrices, thus '
                                   'the last eigenvalue is omitted. See scipy.sparse.linalg.eigsh '
                                   'for more information')
                values, vectors = scipy_sparse_la.eigsh(M, k=k, which='SA')
                values = values[j:k]
                vectors = vectors[:, j:k]
        elif linalg.__package__.startswith('skcuda'):
            import pycuda.gpuarray as gpuarray
            import pycuda.autoinit
            linalg.init()
            vectors, values = linalg.eig(gpuarray.to_gpu(M), 'N', 'V')
            values, vectors = values.get(), vectors.get()
        elif linalg.__package__.startswith('cupy'):
            import cupy as cp
            dtype = re.findall(r'\d+', mode)
            dtype = getattr(cp, 'float' + dtype[0]) if dtype else cp.float32
            values, vectors = linalg.eigh(cp.asarray(M, dtype=dtype))
            values, vectors = cp.asnumpy(values).astype(np.float32), cp.asnumpy(vectors).astype(np.float32)
        else:
            if n_modes is not None:
                LOGGER.info('Scipy is not found, all modes were calculated.')
            else:
                n_modes = dof
            values, vectors = linalg.eigh(M)
        return values, vectors

    def _calc_n_zero_modes(M):
        from scipy.sparse import issparse

        if not issparse(M):
            w = linalg.eigvalsh(M)
        else:
            try:
                from scipy.sparse import linalg as scipy_sparse_la
            except ImportError:
                raise ImportError('failed to import scipy.sparse.linalg, '
                                    'which is required for sparse matrix '
                                    'decomposition')
            w, _ = scipy_sparse_la.eigsh(M, k=dof-1, which='SA')
        n_zeros = sum(w < ZERO)
        return n_zeros

    values, vectors = _eigh(M, eigvals, turbo)
    n_zeros = sum(values < ZERO)

    if warn_zeros:
        if n_zeros < n_modes + expct_n_zeros:
            if n_zeros < expct_n_zeros:
                LOGGER.warning('Fewer than %d (%d) zero eigenvalues were calculated.'%(expct_n_zeros, n_zeros))
            elif n_zeros > expct_n_zeros:
                LOGGER.warning('More than %d (%d) zero eigenvalues were calculated.'%(expct_n_zeros, n_zeros))
        else:
            LOGGER.warning('More than %d zero eigenvalues were detected.'%expct_n_zeros)

    if not zeros:
        if n_zeros > expct_n_zeros:
            if n_zeros == n_modes + expct_n_zeros and n_modes < dof:
                LOGGER.debug('Determing the number of zero eigenvalues...')
                # find the actual number of zero modes
                n_zeros = _calc_n_zero_modes(M)
                LOGGER.debug('%d zero eigenvalues detected.'%n_zeros)
            LOGGER.debug('Solving for additional eigenvalues...')

            if n_modes < dof:
                start = min(n_modes+expct_n_zeros, dof-1); end = min(n_modes+n_zeros-1, dof-1)
                values_, vectors_ = _eigh(M, eigvals=(start, end))
                values = np.concatenate((values, values_))
                vectors = np.hstack((vectors, vectors_))

        # final_n_modes may exceed len(eigvals) - no need to fix for the sake of the simplicity of the code
        final_n_modes = n_zeros + n_modes
        eigvals = values[n_zeros:final_n_modes]
        eigvecs = vectors[:, n_zeros:final_n_modes]
        invvals = 1 / eigvals
    else:
        eigvals = values[:n_modes]
        eigvecs = vectors[:, :n_modes]
        invvals = div0(1, values)
        invvals[:n_zeros] = 0.
        invvals = invvals[:n_modes]

    if reverse:
        invvals = invvals[::-1]
        eigvals = eigvals[::-1]
        eigvecs = eigvecs[:, ::-1]

    return eigvals, eigvecs, invvals
