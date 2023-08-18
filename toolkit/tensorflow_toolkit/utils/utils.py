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


import math
import numpy as np

import tensorflow as tf
# from tensorflow.python.ops.numpy_ops import deg2rad


# axis sequences for Euler angles
_NEXT_AXIS = [1, 2, 0, 1]

# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}

_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())

def euler_matrix(alpha, beta, gamma):
    A = []

    ca = tf.cos(alpha * np.pi / 180.0)
    sa = tf.sin(alpha * np.pi / 180.0)
    cb = tf.cos(beta * np.pi / 180.0)
    sb = tf.sin(beta * np.pi / 180.0)
    cg = tf.cos(gamma * np.pi / 180.0)
    sg = tf.sin(gamma * np.pi / 180.0)

    cc = cb * ca
    cs = cb * sa
    sc = sb * ca
    ss = sb * sa

    A.append([cg * cc - sg * sa, cg * cs + sg * ca, -cg * sb])
    A.append([-sg * cc - cg * sa, -sg * cs + cg * ca, sg * sb])
    A.append([sc, ss, cb])
    return tf.stack(A)

def getXmippOrigin(boxsize):
    return np.asarray([int(0.5 * boxsize),
                       int(0.5 * boxsize),
                       int(0.5 * boxsize)])

def euler_from_matrix(matrix):
    # Only valid for Xmipp axes szyz
    firstaxis, parity, repetition, frame = (2, 1, 1, 0)
    _EPS = np.finfo(float).eps * 4.0

    i = firstaxis
    j = _NEXT_AXIS[i + parity]
    k = _NEXT_AXIS[i - parity + 1]

    M = np.array(matrix, dtype=np.float64, copy=False)[:3, :3]
    if repetition:
        sy = math.sqrt(M[i, j] * M[i, j] + M[i, k] * M[i, k])
        if sy > _EPS:
            ax = math.atan2(M[i, j], M[i, k])
            ay = math.atan2(sy, M[i, i])
            az = math.atan2(M[j, i], -M[k, i])
        else:
            ax = math.atan2(-M[j, k], M[j, j])
            ay = math.atan2(sy, M[i, i])
            az = 0.0
    else:
        cy = math.sqrt(M[i, i] * M[i, i] + M[j, i] * M[j, i])
        if cy > _EPS:
            ax = math.atan2(M[k, j], M[k, k])
            ay = math.atan2(-M[k, i], cy)
            az = math.atan2(M[j, i], M[i, i])
        else:
            ax = math.atan2(-M[j, k], M[j, j])
            ay = math.atan2(-M[k, i], cy)
            az = 0.0

    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    return ax, ay, az

def xmippEulerFromMatrix(matrix):
    return -np.rad2deg(euler_from_matrix(matrix))

def euler_matrix_row(alpha, beta, gamma, row, batch_size):
    A = []

    for idx in range(batch_size):
        ca = tf.cos(tf.gather(alpha, idx, axis=0) * (np.pi / 180.0))
        sa = tf.sin(tf.gather(alpha, idx, axis=0) * (np.pi / 180.0))
        cb = tf.cos(tf.gather(beta, idx, axis=0) * (np.pi / 180.0))
        sb = tf.sin(tf.gather(beta, idx, axis=0) * (np.pi / 180.0))
        cg = tf.cos(tf.gather(gamma, idx, axis=0) * (np.pi / 180.0))
        sg = tf.sin(tf.gather(gamma, idx, axis=0) * (np.pi / 180.0))

        cc = cb * ca
        cs = cb * sa
        sc = sb * ca
        ss = sb * sa

        if row == 1:
            A.append([cg * cc - sg * sa, cg * cs + sg * ca, -cg * sb])
            # A.append([cg * cc - sg * sa, -sg * cc - cg, sc])
        elif row == 2:
            A.append([-sg * cc - cg * sa, -sg * cs + cg * ca, sg * sb])
            # A.append([cg * cs + sg * ca, -sg * cs + cg * ca, sg * ss])
        elif row == 3:
            A.append([sc, ss, cb])
            # A.append([-cg * sb, sg * ss, cb])

    return tf.stack(A)

def euler_matrix_batch(alpha, beta, gamma):

    ca = tf.cos(alpha * (np.pi / 180.0))[:, None]
    sa = tf.sin(alpha * (np.pi / 180.0))[:, None]
    cb = tf.cos(beta * (np.pi / 180.0))[:, None]
    sb = tf.sin(beta * (np.pi / 180.0))[:, None]
    cg = tf.cos(gamma * (np.pi / 180.0))[:, None]
    sg = tf.sin(gamma * (np.pi / 180.0))[:, None]

    cc = cb * ca
    cs = cb * sa
    sc = sb * ca
    ss = sb * sa

    row_1 = tf.concat([cg * cc - sg * sa, cg * cs + sg * ca, -cg * sb], axis=1)
    # A.append([cg * cc - sg * sa, -sg * cc - cg, sc])

    row_2 = tf.concat([-sg * cc - cg * sa, -sg * cs + cg * ca, sg * sb], axis=1)
    # A.append([cg * cs + sg * ca, -sg * cs + cg * ca, sg * ss])

    row_3 = tf.concat([sc, ss, cb], axis=1)
    # A.append([-cg * sb, sg * ss, cb])

    return row_1, row_2, row_3

def ctf_freqs(shape, d=1.0, full=False):
    """
    :param shape: Shape tuple.
    :param d: Frequency spacing in inverse Å (1 / pixel size).
    :param full: When false, return only unique Fourier half-space for real data.
    """
    if full:
        xfrq = tf.constant(np.fft.fftfreq(shape[1]), dtype=tf.float32)
    else:
        xfrq = tf.constant(np.fft.rfftfreq(shape[1]), dtype=tf.float32)
    x, y = tf.meshgrid(xfrq, tf.constant(np.fft.fftfreq(shape[0]), dtype=tf.float32))
    rho = tf.sqrt(x ** 2. + y ** 2.)
    a = tf.atan2(y, x)
    s = rho * d
    return s, a


# @numba.jit(cache=True, nopython=True, nogil=True)
def eval_ctf(s, a, def1, def2, angast=0, phase=0, kv=300, ac=0.1, cs=2.0, bf=0, lp=0):
    """
    :param s: Precomputed frequency grid for CTF evaluation.
    :param a: Precomputed frequency grid angles.
    :param def1: 1st prinicipal underfocus distance (Å).
    :param def2: 2nd principal underfocus distance (Å).
    :param angast: Angle of astigmatism (deg) from x-axis to azimuth.
    :param phase: Phase shift (deg).
    :param kv:  Microscope acceleration potential (kV).
    :param ac:  Amplitude contrast in [0, 1.0].
    :param cs:  Spherical aberration (mm).
    :param bf:  B-factor, divided by 4 in exponential, lowpass positive.
    :param lp:  Hard low-pass filter (Å), should usually be Nyquist.
    """
    angast = angast * (np.pi / 180.0)
    kv = kv * 1e3
    cs = cs * 1e7
    lamb = 12.2643247 / tf.sqrt(kv * (1. + kv * 0.978466e-6))
    def_avg = -(def1 + def2) * 0.5
    def_dev = -(def1 - def2) * 0.5
    k1 = np.pi / 2. * 2. * lamb
    k2 = np.pi / 2. * cs * lamb ** 3.
    k3 = tf.sqrt(1. - ac ** 2.)
    k4 = bf / 4.  # B-factor, follows RELION convention.
    k5 = phase * (np.pi / 180.0)  # Phase shift.
    if lp != 0:  # Hard low- or high-pass.
        s *= s <= (1. / lp)
    s_2 = s ** 2.
    s_4 = s_2 ** 2.
    dZ = def_avg[:, None, None] + def_dev[:, None, None] * (tf.cos(2. * (a - angast[:, None, None])))
    gamma = (k1 * dZ * s_2) + (k2[:, None, None] * s_4) - k5
    # dZ = def_avg + def_dev * (tf.cos(2. * (a - angast)))
    # gamma = (k1 * dZ * s_2) + (k2 * s_4) - k5
    ctf = -(k3 * tf.sin(gamma) - ac * tf.cos(gamma))
    if bf != 0:  # Enforce envelope.
        ctf *= tf.exp(-k4 * s_2)
    return ctf

def computeCTF(defocusU, defocusV, defocusAngle, cs, kv, sr, pad_factor, img_shape, batch_size, applyCTF):
    if applyCTF == 1:
        # s, a = ctf_freqs([img_shape[0], img_shape[0]], 1 / sr)
        # ctf = []
        # for idx in range(batch_size):
        #     def1, def2, angast, cs_var = defocusU[idx], defocusV[idx], defocusAngle[idx], cs[idx]
        #     ctf_img = eval_ctf(s, a, def1, def2, angast=angast, cs=cs_var, kv=kv)
        #     ctf.append(tf.signal.fftshift(ctf_img[:, :img_shape[1]]))
        # return tf.stack(ctf)

        s, a = ctf_freqs([pad_factor * img_shape[0], pad_factor * img_shape[0]], 1 / sr)
        s, a = tf.tile(s[None, :, :], [batch_size, 1, 1]), tf.tile(a[None, :, :], [batch_size, 1, 1])
        ctf = eval_ctf(s, a, defocusU, defocusV, angast=defocusAngle, cs=cs, kv=kv)
        ctf = tf.signal.fftshift(ctf)
        return ctf

    else:
        # size_aux = int(0.5 * pad_factor * img_shape[0] + 1)
        return tf.ones([batch_size, pad_factor * img_shape[0], pad_factor * img_shape[1] - (pad_factor - 1)], dtype=tf.float32)
        # return tf.ones([batch_size, img_shape[0], img_shape[1]], dtype=tf.float32)

def fft_pad(imgs, size_x, size_y):
    padded_imgs = tf.image.resize_with_crop_or_pad(imgs, size_x, size_y)
    ft_images = tf.signal.fftshift(tf.signal.rfft2d(padded_imgs[:, :, :, 0]))
    return ft_images

def ifft_pad(ft_imgs, size_x, size_y):
    padded_imgs = tf.signal.irfft2d(tf.signal.ifftshift(ft_imgs))[..., None]
    imgs = tf.image.resize_with_crop_or_pad(padded_imgs, size_x, size_y)
    return imgs

def gramSchmidt(r):
    c1 = tf.nn.l2_normalize(r[:, :3], axis=-1)
    c2 = tf.nn.l2_normalize(r[:, 3:] - dot(c1, r[:, 3:]) * c1, axis=-1)
    c3 = tf.linalg.cross(c1, c2)
    c = tf.stack([c1, c2, c3], axis=2)
    return c[:, 0, :], c[:, 1, :], c[:, 2, :]

def dot(a, b):
    return tf.reduce_sum(a * b, axis=-1, keepdims=True)

def epochs_from_iterations(total_samples_seen, n_samples, batch_size):
    # Get the total number of batch iterations needed
    batch_iterations = np.ceil(total_samples_seen / batch_size)

    # Get number of batches in one iteration
    steps = np.ceil(n_samples / batch_size)

    # Get number of epochs to reach batch_iterations
    return int(batch_iterations / steps)

