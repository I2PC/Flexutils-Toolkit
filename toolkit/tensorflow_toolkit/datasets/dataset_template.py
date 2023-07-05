# **************************************************************************
# *
# * Authors:     David Herreros (dherreros@cnb.csic.es)
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


import types
import tensorflow as tf


def sequence_to_data_pipeline(sequence):
    ''' Load sequence on tf.data pipeline '''

    # Public getitem method
    def getitem(self, index):
        return self.__getitem__(index)

    # Prepare sequence
    sequence.shuffle = False
    sequence.getitem = types.MethodType(getitem, sequence)
    sequence.len = sequence.__len__()

    def gen_data_generator():
        for i in range(sequence.len):
            yield sequence.getitem(i)
    return gen_data_generator, sequence


def create_dataset(data_generator, sequence, prefetch=True, shuffle=True, cache=True, interleave=False):
    ''' Create Dataset for tf.data pipeline '''
    dataset = tf.data.Dataset.from_generator(data_generator, output_signature=(
        tf.TensorSpec(shape=(None, sequence.xsize, sequence.xsize, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.int32)))

    if prefetch:
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

    if shuffle:
        dataset = dataset.shuffle(sequence.len, reshuffle_each_iteration=True)

    if interleave:
        dataset = tf.data.Dataset.range(1).interleave(
            lambda _: dataset,
            num_parallel_calls=tf.data.AUTOTUNE
        )

    if cache:
        dataset = dataset.cache()

    # Assign cardinality beforehand for progress bar
    dataset = tf.data.experimental.assert_cardinality(sequence.len)(dataset)

    return dataset

