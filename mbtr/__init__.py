import os
import subprocess
import tempfile
import sys
import scipy.sparse
import logging

import numpy

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))


def update_mbtr_exe():
    subprocess.call(['make', '-C', SCRIPT_PATH])


def export_periodic_xyz(samples, fn, prop):
    f = open(fn, 'w')
    for sample in samples:
        f.write('%d\n' % len(sample['atom_labels']))
        f.write('%r\n' % sample[prop])
        f.write('%f %f %f %f %f %f %f %f %f\n' % tuple(numpy.array(sample['basis_matrix']).flatten().tolist()))

        atom_positions = numpy.array(sample['atom_positions_fractional'])
        atom_positions = numpy.dot(numpy.array(sample['basis_matrix']).T, atom_positions.T).T
        for i in range(len(sample['atom_labels'])):
            f.write('%s %f %f %f\n' % (sample['atom_labels'][i],
                                       atom_positions[i][0], atom_positions[i][1], atom_positions[i][2]))


def get_mbtr(output_fn, dense=True):
    if dense:
        x = numpy.load(output_fn)
    else:
        data = numpy.load(output_fn)
        data_idx = numpy.load(output_fn+'-indices')
        x = scipy.sparse.csr_matrix((data, (data_idx[:, 0], data_idx[:, 1])))

    return x


def get_new_temp_file():
    fid, tmp = tempfile.mkstemp()
    os.close(fid)
    return tmp


def clean_temp_file(fn):
    os.remove(fn)
    if os.path.exists(fn + '-indices'):
        os.remove(fn + '-indices')


def obtain_mbtr(samples, periodic, prop, sigma, D, grid_size, start=None, end=None, other_args=None):
    tmpxyz_fn = get_new_temp_file()
    tmpnpy_fn = get_new_temp_file()

    try:
        if isinstance(samples, list):
            export_periodic_xyz(samples, tmpxyz_fn, prop)
            ds = tmpxyz_fn
        else:
            ds = samples

        cmd = [os.path.join(SCRIPT_PATH, 'mbtr_cpp'),
               '-periodic' if periodic else '-molecular',
               '-dataset', ds,
               '-output', tmpnpy_fn,
               '-sigma', '%f' % sigma,
               '-D', '%f' % D,
               '-grid', '%d' % grid_size]
        if other_args is not None:
            cmd += other_args
        if start:
            cmd += ['-start', '%f' % start]
        if end:
            cmd += ['-end', '%f' % end]
        logging.debug(' '.join(cmd))

        update_mbtr_exe()
        subprocess.call(cmd)

        x = get_mbtr(tmpnpy_fn, '-sparse' not in cmd)
        if len(x.shape) > 2 and '-sparse' not in cmd:
            x = x.reshape((x.shape[0], x.size//x.shape[0]))

        if isinstance(samples, list):
            y = numpy.array([_[prop] for _ in samples])
            y = y.reshape((y.size, 1))
        else:
            y = None
    except Exception as e:
        clean_temp_file(tmpxyz_fn)
        clean_temp_file(tmpnpy_fn)
        raise e

    return x, y
