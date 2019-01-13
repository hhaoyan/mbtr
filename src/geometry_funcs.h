#ifndef _GEOMETRY_FUNCS_H_
#define _GEOMETRY_FUNCS_H_

#include <limits>
#include "mbtr.h"

class OneBodyCountingG {
    double _min, _max;
public:
    OneBodyCountingG() {
        _min = std::numeric_limits<double>::max();
        _max = std::numeric_limits<double>::min();
    }

    void reset_minmax() {
        _min = std::numeric_limits<double>::max();
        _max = std::numeric_limits<double>::min();
    }

    const std::pair<double, double> get_minmax() const {
        return std::make_pair(_min, _max);
    }

    double operator()(const System::Atom *atoms[]) {
        auto v = static_cast<double>(atoms[0]->atom_number);

        _min = std::min(_min, v);
        _max = std::max(_max, v);

        return v;
    }
};

class InverseDoubletDistanceG {
    double _min, _max;
public:
    InverseDoubletDistanceG() {
        _min = std::numeric_limits<double>::max();
        _max = std::numeric_limits<double>::min();
    }

    void reset_minmax() {
        _min = std::numeric_limits<double>::max();
        _max = std::numeric_limits<double>::min();
    }

    const std::pair<double, double> get_minmax() const {
        return std::make_pair(_min, _max);
    }

    double operator()(const System::Atom *atoms[]) {
        double distance = sqrt(
                (atoms[0]->position[0] - atoms[1]->position[0]) * (atoms[0]->position[0] - atoms[1]->position[0]) +
                (atoms[0]->position[1] - atoms[1]->position[1]) * (atoms[0]->position[1] - atoms[1]->position[1]) +
                (atoms[0]->position[2] - atoms[1]->position[2]) * (atoms[0]->position[2] - atoms[1]->position[2])
        );

        double v = 1.0f / distance;

        _min = std::min(_min, v);
        _max = std::max(_max, v);

        return v;
    }
};

class InverseTripletCosineAngleG {
    double _min, _max;
public:
    InverseTripletCosineAngleG() {
        _min = std::numeric_limits<double>::max();
        _max = std::numeric_limits<double>::min();
    }

    void reset_minmax() {
        _min = std::numeric_limits<double>::max();
        _max = std::numeric_limits<double>::min();
    }

    const std::pair<double, double> get_minmax() const {
        return std::make_pair(_min, _max);
    }

    double operator()(const System::Atom *atoms[]) {
        double distance1_squared = (
                (atoms[0]->position[0] - atoms[1]->position[0]) * (atoms[0]->position[0] - atoms[1]->position[0]) +
                (atoms[0]->position[1] - atoms[1]->position[1]) * (atoms[0]->position[1] - atoms[1]->position[1]) +
                (atoms[0]->position[2] - atoms[1]->position[2]) * (atoms[0]->position[2] - atoms[1]->position[2])
        );
        double distance2_squared = (
                (atoms[0]->position[0] - atoms[2]->position[0]) * (atoms[0]->position[0] - atoms[2]->position[0]) +
                (atoms[0]->position[1] - atoms[2]->position[1]) * (atoms[0]->position[1] - atoms[2]->position[1]) +
                (atoms[0]->position[2] - atoms[2]->position[2]) * (atoms[0]->position[2] - atoms[2]->position[2])
        );
        double distance3_squared = (
                (atoms[1]->position[0] - atoms[2]->position[0]) * (atoms[1]->position[0] - atoms[2]->position[0]) +
                (atoms[1]->position[1] - atoms[2]->position[1]) * (atoms[1]->position[1] - atoms[2]->position[1]) +
                (atoms[1]->position[2] - atoms[2]->position[2]) * (atoms[1]->position[2] - atoms[2]->position[2])
        );
        double cosine = (distance1_squared + distance3_squared - distance2_squared)
                        / sqrt(4.0f * distance1_squared * distance3_squared);

        _min = std::min(_min, cosine);
        _max = std::max(_max, cosine);

        return cosine;
    }
};

class TripletAngleG {
    double _min, _max;
public:
    TripletAngleG() {
        _min = std::numeric_limits<double>::max();
        _max = std::numeric_limits<double>::min();
    }

    void reset_minmax() {
        _min = std::numeric_limits<double>::max();
        _max = std::numeric_limits<double>::min();
    }

    const std::pair<double, double> get_minmax() const {
        return std::make_pair(_min, _max);
    }

    double operator()(const System::Atom *atoms[]) {
        double distance1_squared = (
                (atoms[0]->position[0] - atoms[1]->position[0]) * (atoms[0]->position[0] - atoms[1]->position[0]) +
                (atoms[0]->position[1] - atoms[1]->position[1]) * (atoms[0]->position[1] - atoms[1]->position[1]) +
                (atoms[0]->position[2] - atoms[1]->position[2]) * (atoms[0]->position[2] - atoms[1]->position[2])
        );
        double distance2_squared = (
                (atoms[0]->position[0] - atoms[2]->position[0]) * (atoms[0]->position[0] - atoms[2]->position[0]) +
                (atoms[0]->position[1] - atoms[2]->position[1]) * (atoms[0]->position[1] - atoms[2]->position[1]) +
                (atoms[0]->position[2] - atoms[2]->position[2]) * (atoms[0]->position[2] - atoms[2]->position[2])
        );
        double distance3_squared = (
                (atoms[1]->position[0] - atoms[2]->position[0]) * (atoms[1]->position[0] - atoms[2]->position[0]) +
                (atoms[1]->position[1] - atoms[2]->position[1]) * (atoms[1]->position[1] - atoms[2]->position[1]) +
                (atoms[1]->position[2] - atoms[2]->position[2]) * (atoms[1]->position[2] - atoms[2]->position[2])
        );
        double cosine = (distance1_squared + distance3_squared - distance2_squared)
                        / sqrt(4.0f * distance1_squared * distance3_squared);
        cosine = cosine > 1.0 ? 1.0 : (cosine < -1.0 ? -1.0 : cosine);
        double angle = std::acos(cosine);

        _min = std::min(_min, angle);
        _max = std::max(_max, angle);

        return angle;
    }
};

#endif // _GEOMETRY_FUNCS_H_
