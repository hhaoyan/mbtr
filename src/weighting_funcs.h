#ifndef _WEIGHTING_FUNCS_H_
#define _WEIGHTING_FUNCS_H_

#include "mbtr.h"

class NoWeightingW {
public:
    double operator()(const System::Atom *_[]) {
        return 1.0;
    }
};

class DoubletDistanceExponentialWeightingW {
    double D_;
public:
    explicit DoubletDistanceExponentialWeightingW(double D) {
        D_ = D;
    }

    double operator()(const System::Atom *atoms[]) {
        double distance = sqrt(
                (atoms[0]->position[0] - atoms[1]->position[0]) * (atoms[0]->position[0] - atoms[1]->position[0]) +
                (atoms[0]->position[1] - atoms[1]->position[1]) * (atoms[0]->position[1] - atoms[1]->position[1]) +
                (atoms[0]->position[2] - atoms[1]->position[2]) * (atoms[0]->position[2] - atoms[1]->position[2])
        );

        return exp(-distance / D_);
    }
};

class DoubletDistanceSquaredExponentialWeightingW {
    double _D;
public:
    explicit DoubletDistanceSquaredExponentialWeightingW(double D) {
        _D = D;
    }

    double operator()(const System::Atom *atoms[]) {
        double distance = sqrt(
                (atoms[0]->position[0] - atoms[1]->position[0]) * (atoms[0]->position[0] - atoms[1]->position[0]) +
                (atoms[0]->position[1] - atoms[1]->position[1]) * (atoms[0]->position[1] - atoms[1]->position[1]) +
                (atoms[0]->position[2] - atoms[1]->position[2]) * (atoms[0]->position[2] - atoms[1]->position[2])
        );

        return exp(-distance * distance / _D);
    }
};

class DoubletDistanceInvQuadraticWeightingW {
public:
    double operator()(const System::Atom *atoms[]) {
        double distance = sqrt(
                (atoms[0]->position[0] - atoms[1]->position[0]) * (atoms[0]->position[0] - atoms[1]->position[0]) +
                (atoms[0]->position[1] - atoms[1]->position[1]) * (atoms[0]->position[1] - atoms[1]->position[1]) +
                (atoms[0]->position[2] - atoms[1]->position[2]) * (atoms[0]->position[2] - atoms[1]->position[2])
        );

        return 1.0 / (distance * distance);
    }
};

class DoubletDistanceQuadraticWeightingW {
public:
    double operator()(const System::Atom *atoms[]) {
        double distance = sqrt(
                (atoms[0]->position[0] - atoms[1]->position[0]) * (atoms[0]->position[0] - atoms[1]->position[0]) +
                (atoms[0]->position[1] - atoms[1]->position[1]) * (atoms[0]->position[1] - atoms[1]->position[1]) +
                (atoms[0]->position[2] - atoms[1]->position[2]) * (atoms[0]->position[2] - atoms[1]->position[2])
        );

        return (distance * distance);
    }
};

class TripletDistanceExponentialWeightingW {
    double D_;
public:
    explicit TripletDistanceExponentialWeightingW(double D) {
        D_ = D;
    }

    double operator()(const System::Atom *atoms[]) {
        double distance1 = sqrt(
                (atoms[0]->position[0] - atoms[1]->position[0]) * (atoms[0]->position[0] - atoms[1]->position[0]) +
                (atoms[0]->position[1] - atoms[1]->position[1]) * (atoms[0]->position[1] - atoms[1]->position[1]) +
                (atoms[0]->position[2] - atoms[1]->position[2]) * (atoms[0]->position[2] - atoms[1]->position[2])
        );
        double distance2 = sqrt(
                (atoms[0]->position[0] - atoms[2]->position[0]) * (atoms[0]->position[0] - atoms[2]->position[0]) +
                (atoms[0]->position[1] - atoms[2]->position[1]) * (atoms[0]->position[1] - atoms[2]->position[1]) +
                (atoms[0]->position[2] - atoms[2]->position[2]) * (atoms[0]->position[2] - atoms[2]->position[2])
        );
        double distance3 = sqrt(
                (atoms[1]->position[0] - atoms[2]->position[0]) * (atoms[1]->position[0] - atoms[2]->position[0]) +
                (atoms[1]->position[1] - atoms[2]->position[1]) * (atoms[1]->position[1] - atoms[2]->position[1]) +
                (atoms[1]->position[2] - atoms[2]->position[2]) * (atoms[1]->position[2] - atoms[2]->position[2])
        );

        return exp(-(distance1 + distance2 + distance3) / D_);
    }
};

#endif // _WEIGHTING_FUNCS_H_