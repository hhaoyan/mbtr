#ifndef _PRESETS_H_
#define _PRESETS_H_

#include "mbtr.h"
#include "weighting_funcs.h"
#include "correlation_funcs.h"
#include "density_funcs.h"
#include "geometry_funcs.h"
#include "utility.h"

#define USE_DISCRETE_ERF

#ifdef USE_DISCRETE_ERF
typedef DiscreteErfDensity<10000> ErfDensity;
#else
typedef AccurateErfDensity ErfDensity;
#endif

struct PresetInfo {
    std::vector<System> systems;
    std::vector<uint> z;

    int preset;
    uint grid_size, rank;
    double sigma, D;

    bool is_periodic, is_range_set;
    std::pair<double, double> tensor_range;
};

const char *PresetErrorMsg =
        "You must choose from the following presets:"
        "  101. Molecular\trank=1\tg=1\t\tw=none\t\td=gaussian\tcorr=delta(z1, z2).\n"
        "  102. Molecular\trank=2\tg=1/r\t\tw=none\t\td=gaussian\tcorr=delta(z1, z2).\n"
        "  103. Molecular\trank=2\tg=1/r\t\tw=1/r^2\t\td=gaussian\tcorr=delta(z1, z2).\n"
        "  104. Molecular\trank=2\tg=1/r\t\tw=r^2\t\td=gaussian\tcorr=delta(z1, z2).\n"
        "  105. Molecular\trank=3\tg=cos(angle)\tw=none\t\td=gaussian\tcorr=delta(z1, z2).\n"
        "  151. Periodic\trank=1\tg=1\t\tw=none\t\td=gaussian\tcorr=delta(z1, z2).\n"
        "  152. Periodic\trank=2\tg=1/r\t\tw=exp(-r/D)\td=gaussian\tcorr=delta(z1, z2).\n"
        "  153. Periodic\trank=2\tg=1/r\t\tw=exp(-r^2/D)\td=gaussian\tcorr=delta(z1, z2).\n"
        "  154. Periodic\trank=3\tg=cos(angle)\tw=exp(-(r1+r2+r3)/D)\td=gaussian\tcorr=delta(z1, z2).\n";

#define RequirePeriodic(preset_id, info) if ((info).is_periodic) { \
    PyErr_SetString(PyExc_ValueError, \
            "Preset " #preset_id " only accepts crystal systems"); \
    return false;\
}
#define RequireMolecular(preset_id, info) if (!(info).is_periodic) { \
    PyErr_SetString(PyExc_ValueError, \
            "Preset " #preset_id " only accepts molecular systems"); \
    return false;\
}
#define ReturnMBTR(rank, info, arrays, g, w, d, c) {return \
    ComputerMBTR<rank>((info).systems, arrays, (info).tensor_range, (info).grid_size, \
    g, w, d, c);}
#define FitMBTR(rank, info, output, g, w, d, c) {return \
    FitTensorLastDimRange<rank>((info).systems, output, g, w, d, c);}

bool PresetsFitMBTR(PresetInfo &info, std::pair<double, double> &output) {
    InverseTripletCosineAngleG g_3body_cos_angle;
    InverseDoubletDistanceG g_2body_inv_r;
    OneBodyCountingG g_1body_1;

    NoWeightingW w_none;
    DoubletDistanceInvQuadraticWeightingW w_2body_inv_r2;
    DoubletDistanceQuadraticWeightingW w_2body_r2;
    DoubletDistanceExponentialWeightingW w_2body_exp_nr(info.D);
    DoubletDistanceSquaredExponentialWeightingW w_2body_exp_nr2(info.D);
    TripletDistanceExponentialWeightingW w_3body_exp_r_sum(info.D);


    ErfDensity d_gaussian(info.sigma);

    DeltaCorrelation corr_delta;

    switch (info.preset) {
        case 101:
            RequireMolecular(101, info);
            FitMBTR(1, info, output, g_1body_1, w_none, d_gaussian, corr_delta);

        case 102:
            RequireMolecular(102, info);
            FitMBTR(2, info, output, g_2body_inv_r, w_none, d_gaussian, corr_delta);

        case 103:
            RequireMolecular(103, info);
            FitMBTR(2, info, output, g_2body_inv_r, w_2body_inv_r2, d_gaussian, corr_delta);

        case 104:
            RequireMolecular(104, info);
            FitMBTR(2, info, output, g_2body_inv_r, w_2body_r2, d_gaussian, corr_delta);

        case 105:
            RequireMolecular(105, info);
            FitMBTR(3, info, output, g_3body_cos_angle, w_none, d_gaussian, corr_delta);

        case 151:
            RequirePeriodic(151, info);
            FitMBTR(1, info, output, g_1body_1, w_none, d_gaussian, corr_delta);

        case 152:
            RequirePeriodic(152, info);
            FitMBTR(2, info, output, g_2body_inv_r, w_2body_exp_nr, d_gaussian, corr_delta);

        case 153:
            RequirePeriodic(153, info);
            FitMBTR(2, info, output, g_2body_inv_r, w_2body_exp_nr2, d_gaussian, corr_delta);

        case 154:
            RequirePeriodic(154, info);
            FitMBTR(3, info, output, g_3body_cos_angle, w_3body_exp_r_sum, d_gaussian, corr_delta);

        default:
            PyErr_SetString(PyExc_ValueError, PresetErrorMsg);
            return false;
    }
}

bool PresetsComputeMBTR(PresetInfo &info, std::vector<MBTRResult> &arrays) {
    InverseTripletCosineAngleG g_3body_cos_angle;
    InverseDoubletDistanceG g_2body_inv_r;
    OneBodyCountingG g_1body_1;

    NoWeightingW w_none;
    DoubletDistanceInvQuadraticWeightingW w_2body_inv_r2;
    DoubletDistanceQuadraticWeightingW w_2body_r2;
    DoubletDistanceExponentialWeightingW w_2body_exp_nr(info.D);
    DoubletDistanceSquaredExponentialWeightingW w_2body_exp_nr2(info.D);
    TripletDistanceExponentialWeightingW w_3body_exp_r_sum(info.D);

    ErfDensity d_gaussian(info.sigma);

    DeltaCorrelation corr_delta;

    switch (info.preset) {
        case 101:
            RequireMolecular(101, info);
            ReturnMBTR(1, info, arrays, g_1body_1, w_none, d_gaussian, corr_delta);

        case 102:
            RequireMolecular(102, info);
            ReturnMBTR(2, info, arrays, g_2body_inv_r, w_none, d_gaussian, corr_delta);

        case 103:
            RequireMolecular(103, info);
            ReturnMBTR(2, info, arrays, g_2body_inv_r, w_2body_inv_r2, d_gaussian, corr_delta);

        case 104:
            RequireMolecular(104, info);
            ReturnMBTR(2, info, arrays, g_2body_inv_r, w_2body_r2, d_gaussian, corr_delta);

        case 105:
            RequireMolecular(105, info);
            ReturnMBTR(3, info, arrays, g_3body_cos_angle, w_none, d_gaussian, corr_delta);

        case 151:
            RequirePeriodic(151, info);
            ReturnMBTR(1, info, arrays, g_1body_1, w_none, d_gaussian, corr_delta);

        case 152:
            RequirePeriodic(152, info);
            ReturnMBTR(2, info, arrays, g_2body_inv_r, w_2body_exp_nr, d_gaussian, corr_delta);

        case 153:
            RequirePeriodic(153, info);
            ReturnMBTR(2, info, arrays, g_2body_inv_r, w_2body_exp_nr2, d_gaussian, corr_delta);

        case 154:
            RequirePeriodic(154, info);
            ReturnMBTR(3, info, arrays, g_3body_cos_angle, w_3body_exp_r_sum, d_gaussian, corr_delta);

        default:
            PyErr_SetString(PyExc_ValueError, PresetErrorMsg);
            return false;
    }
}

#endif // _PRESETS_H_
