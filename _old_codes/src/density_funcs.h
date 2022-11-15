#ifndef _DENSITY_FUNCS_H_
#define _DENSITY_FUNCS_H_

#include <cmath>

class AccurateErfDensity {
    double sigma_;
public:
    explicit AccurateErfDensity(double sigma) {
        sigma_ = sigma;
    }

    inline double operator()(std::pair<double, double> &bound, float g) {
        double q0 = erf((bound.first - g) / sigma_ / sqrt(2.0)) / 2.0;
        double q1 = erf((bound.second - g) / sigma_ / sqrt(2.0)) / 2.0;
        return q1 - q0;
    }

    double operator()(double x) {
        return exp(-x * x / 2.0 / sigma_ / sigma_) / sigma_ / sqrt(2.0 * acos(-1.0));
    }

    // invfx
    double operator[](double y) {
        return sqrt(log(sigma_ * sqrt(2.0 * acos(-1.0)) * y) * -2.0 * sigma_ * sigma_);
    }
};

template<uint bins>
class DiscreteErfDensity {
    const double starting = -3.0, ending = 3.0;
    double sigma_;
    double sigma_mul_sqrt2_, delta_;

    // for speed up the operator()(pair, float)
    double cached_v1_, cached_v2_;
    double left_most_, right_most_;
    // 

    static std::vector<double> values_;
    static std::vector<double> bins_left_;
public:
    explicit DiscreteErfDensity(double sigma) {
        sigma_ = sigma;

        if (values_.size() == 0)
            for (uint i = 0; i < bins; i++) {
                bins_left_.push_back(starting + (ending - starting) * i / bins);
                values_.push_back(erf(bins_left_.back()) / 2.0);
            }

        sigma_mul_sqrt2_ = sigma_ * sqrtf(2.0);
        delta_ = (ending - starting) / bins;

        cached_v1_ = starting / delta_;
        cached_v2_ = 1.0 / (sigma_mul_sqrt2_ * delta_);
        left_most_ = starting * sigma_mul_sqrt2_;
        right_most_ = ending * sigma_mul_sqrt2_;
    }

    inline double operator()(std::pair<double, double> &bound, double g) {
        // early termination seems to work great to reduce computation time
        if (bound.second - g < left_most_)
            return 0.0;
        if (bound.first - g > right_most_)
            return 0.0;
        // float left = ((bound.first - g)/_sigma_mul_sqrt2 - starting) / _delta;
        // float right = ((bound.second - g)/_sigma_mul_sqrt2 - starting) / _delta;
        double left = (bound.first - g) * cached_v2_ - cached_v1_;
        double right = (bound.second - g) * cached_v2_ - cached_v1_;
        int left_idx = (int) left;
        int right_idx = (int) right;

        uint capped_left_idx = left_idx < 0 ? 0 : left_idx;
        capped_left_idx = capped_left_idx >= bins ? bins - 1 : capped_left_idx;
        uint capped_right_idx = right_idx < 0 ? 0 : right_idx;
        capped_right_idx = capped_right_idx >= bins ? bins - 1 : capped_right_idx;

        // we should divide 2.0 here, but to speed up, we have already did this in the constructor
        return (values_[capped_right_idx] - values_[capped_left_idx]);
    }

    double operator()(double x) {
        return exp(-x * x / 2 / sigma_ / sigma_) / sigma_ / sqrt(2.0 * acos(-1));
    }

    // invfx
    double operator[](double y) {
        return sqrt(log(sigma_ * sqrt(2.0 * acos(-1)) * y) * -2.0 * sigma_ * sigma_);
    }
};

template<uint bins>
std::vector<double> DiscreteErfDensity<bins>::values_;
template<uint bins>
std::vector<double> DiscreteErfDensity<bins>::bins_left_;

#endif // _DENSITY_FUNCS_H_
