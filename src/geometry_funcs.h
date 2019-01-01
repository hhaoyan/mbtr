#include "mbtr.h"

class OneBodyCountingG {
    double _min, _max;
public:
    OneBodyCountingG() {
        _min = std::numeric_limits<double>::max();
        _max = std::numeric_limits<double>::lowest();
    }

    void reset_minmax() {
        _min = std::numeric_limits<double>::max();
        _max = std::numeric_limits<double>::lowest();
    }

    const std::pair<double, double> get_minmax() const {
        return std::make_pair(_min, _max);
    }

    double operator()(const System::Atom *atoms[]) {
        auto v = static_cast<double>(atoms[0]->atom_number);

        _min = Py_MIN(_min, v);
        _max = Py_MAX(_max, v);

        return v;
    }
};

class InverseDoubletDistanceG {
    double _min, _max;
public:
    InverseDoubletDistanceG() {
        _min = std::numeric_limits<double>::max();
        _max = std::numeric_limits<double>::lowest();
    }

    void reset_minmax() {
        _min = std::numeric_limits<double>::max();
        _max = std::numeric_limits<double>::lowest();
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

        _min = Py_MIN(_min, v);
        _max = Py_MAX(_max, v);

        return v;
    }
};

class InverseTripletCosineAngleG {
    double _min, _max;
public:
    InverseTripletCosineAngleG() {
        _min = std::numeric_limits<double>::max();
        _max = std::numeric_limits<double>::lowest();
    }

    void reset_minmax() {
        _min = std::numeric_limits<double>::max();
        _max = std::numeric_limits<double>::lowest();
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

        _min = Py_MIN(_min, cosine);
        _max = Py_MAX(_max, cosine);

        return cosine;
    }
};
