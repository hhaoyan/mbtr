#ifndef _UTILITY_H_
#define _UTILITY_H_

#include <vector>
#include <limits>
#include "mbtr.h"

template<uint rank, typename g_func, typename w_func, typename d_func, typename corr_func>
bool FitTensorLastDimRange(
        std::vector<System> &systems, std::pair<double, double> &output,
        g_func g, w_func w, d_func d, corr_func corr) {

    output.first = std::numeric_limits<double>::max();
    output.second = std::numeric_limits<double>::lowest();

    for (auto system:systems) {
        MBTR<rank, g_func, w_func, d_func, corr_func> mbtr(system, 0.0, 1.0, 2, g, w, d, corr);

        size_t sz;
        double *array = mbtr.WriteMBTRToArray(mbtr.NotNullAtomNumbers(), sz);
        if (array == nullptr) {
            return false;
        }
        PyMem_Free(reinterpret_cast<void *>(array));

        auto _minmax = mbtr.g().get_minmax();
        output.first = Py_MIN(output.first, _minmax.first);
        output.second = Py_MAX(output.second, _minmax.second);
    }

    // Consider the "smearing" effect.
    double cutoff = d[5e-4];
    output.first -= cutoff;
    output.second += cutoff;

    return true;
}

struct MBTRResult {
    double *array;
    size_t array_size;
    std::vector<uint> atom_number;
};

template<uint rank, typename g_func, typename w_func, typename d_func, typename corr_func>
bool ComputerMBTR(
        std::vector<System> &systems, std::vector<MBTRResult> &output,
        std::pair<double, double> tensor_range, uint grid_size,
        g_func g, w_func w, d_func d, corr_func corr) {

    for (auto system:systems) {
        MBTR<rank, g_func, w_func, d_func, corr_func> mbtr(system,
                                                           tensor_range.first, tensor_range.second, grid_size, g, w, d,
                                                           corr);
        MBTRResult result;

        result.atom_number = mbtr.NotNullAtomNumbers();
        result.array = mbtr.WriteMBTRToArray(result.atom_number, result.array_size);
        if (result.array == nullptr) {
            goto fail;
        }

        output.push_back(result);
    }

    return true;

    fail:
    // Only fails when no memory... No other reasons could possibly cause
    // the program to fail.
    for (auto result: output) {
        double *array = result.array;
        PyMem_Free(reinterpret_cast<void *>(array));
    }
    output.clear();
    return false;
}

#endif
