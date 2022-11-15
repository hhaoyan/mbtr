#ifndef _MBTR_H_
#define _MBTR_H_

#include <vector>
#include <cmath>
#include <exception>
#include <algorithm>

typedef unsigned int uint;

class MBTRTooManyReflections : std::exception {
    const char *
    what() const noexcept { return "Too many reflections in MBTR, try using another weighting function.\n"; }
};

class System {
public:
    class Atom {
    public:
        Atom(uint an, double px, double py, double pz) : atom_number(an), position{px, py, pz} {}

        uint atom_number;
        double position[3];
    };

    bool is_periodic;
    std::vector <Atom> atoms;
    double basis_vector[3][3];

    void SetBasis(const double *v) {
        basis_vector[0][0] = v[0];
        basis_vector[0][1] = v[1];
        basis_vector[0][2] = v[2];
        basis_vector[1][0] = v[3];
        basis_vector[1][1] = v[4];
        basis_vector[1][2] = v[5];
        basis_vector[2][0] = v[6];
        basis_vector[2][1] = v[7];
        basis_vector[2][2] = v[8];
    }

    void SetBasis(double _1, double _2, double _3, double _4, double _5, double _6, double _7, double _8, double _9) {
        basis_vector[0][0] = _1;
        basis_vector[0][1] = _2;
        basis_vector[0][2] = _3;
        basis_vector[1][0] = _4;
        basis_vector[1][1] = _5;
        basis_vector[1][2] = _6;
        basis_vector[2][0] = _7;
        basis_vector[2][1] = _8;
        basis_vector[2][2] = _9;
    }
};

template<uint rank, typename g_func, typename w_func, typename d_func, typename corr_func>
class MBTR {
    // definitions of the MBTR
    const System &system_;
    std::vector <std::pair<double, double>> bounds_;

    // inner states for performing the summation
    const System::Atom *selected_atoms_[rank];

    // periodic systems
    std::vector <System::Atom> reflected_cells_;
    size_t firstcell_atoms_index_begin_, firstcell_atoms_index_end_;

    // functions
    g_func g_func_;
    w_func w_func_;
    d_func d_func_;
    corr_func corr_func_;

    void GenerateReflectedImagesPeriodic() {
        // FIXME: we still have big problems here.

        int reflectx = 0, reflecty = 0, reflectz = 0;

        if (rank > 1) { // not needed for rank 1 (think!)
            // find out how many reflections are needed in three directions
            System::Atom atom0(system_.atoms[0].atom_number,
                               system_.atoms[0].position[0],
                               system_.atoms[0].position[1],
                               system_.atoms[0].position[2]);

            for (;; reflectx++) {
                System::Atom atom1(system_.atoms[0].atom_number,
                                   system_.atoms[0].position[0] + system_.basis_vector[0][0] * reflectx,
                                   system_.atoms[0].position[1] + system_.basis_vector[0][1] * reflectx,
                                   system_.atoms[0].position[2] + system_.basis_vector[0][2] * reflectx);
                const System::Atom *atoms[3] = {&atom0, &atom1, &atom0};
                double w = w_func_(atoms);
                if (w < 1e-4)
                    break;
            }
            for (;; reflecty++) {
                System::Atom atom1(system_.atoms[0].atom_number,
                                   system_.atoms[0].position[0] + system_.basis_vector[1][0] * reflecty,
                                   system_.atoms[0].position[1] + system_.basis_vector[1][1] * reflecty,
                                   system_.atoms[0].position[2] + system_.basis_vector[1][2] * reflecty);
                const System::Atom *atoms[3] = {&atom0, &atom1, &atom0};
                double w = w_func_(atoms);
                if (w < 1e-4)
                    break;
            }
            for (;; reflectz++) {
                System::Atom atom1(system_.atoms[0].atom_number,
                                   system_.atoms[0].position[0] + system_.basis_vector[2][0] * reflectz,
                                   system_.atoms[0].position[1] + system_.basis_vector[2][1] * reflectz,
                                   system_.atoms[0].position[2] + system_.basis_vector[2][2] * reflectz);
                const System::Atom *atoms[3] = {&atom0, &atom1, &atom0};
                double w = w_func_(atoms);
                if (w < 1e-4)
                    break;
            }
        }

        for (int x = -reflectx; x <= reflectx; x++)
            for (int y = -reflecty; y <= reflecty; y++)
                for (int z = -reflectz; z <= reflectz; z++) {

                    double pos[3];
                    pos[0] = system_.basis_vector[0][0] * x +
                             system_.basis_vector[1][0] * y +
                             system_.basis_vector[2][0] * z;
                    pos[1] = system_.basis_vector[0][1] * x +
                             system_.basis_vector[1][1] * y +
                             system_.basis_vector[2][1] * z;
                    pos[2] = system_.basis_vector[0][2] * x +
                             system_.basis_vector[1][2] * y +
                             system_.basis_vector[2][2] * z;

                    if (x == 0 && y == 0 && z == 0)
                        firstcell_atoms_index_begin_ = reflected_cells_.size();

                    for (System::Atom a:system_.atoms) {
                        reflected_cells_.push_back(System::Atom(a.atom_number,
                                                                a.position[0] + pos[0],
                                                                a.position[1] + pos[1],
                                                                a.position[2] + pos[2]));
                    }

                    if (x == 0 && y == 0 && z == 0)
                        firstcell_atoms_index_end_ = reflected_cells_.size();

                }

    }

    void CalculateMBTRInnerLoop(const std::vector <System::Atom> &atoms,
                                uint z[], std::vector<double> &v,
                                uint current_rank) {
        for (size_t j = 0; j < atoms.size(); ++j) {
            selected_atoms_[current_rank] = &(atoms[j]);

            for (uint i = 0; i < current_rank; ++i) {
                if (selected_atoms_[i] == selected_atoms_[current_rank])
                    goto done;
            }
            if (current_rank == rank - 1) {
                double g = g_func_(selected_atoms_);
                double w = w_func_(selected_atoms_);

                if (w > 1e-7) {
                    double corr_coef = 1.0;
                    for (uint i = 0; i < rank; ++i) {
                        corr_coef *= corr_func_(z[i], selected_atoms_[i]->atom_number);
                    }

                    for (size_t i = 0; i < v.size(); ++i) {
                        v[i] += w * d_func_(bounds_[i], g) * corr_coef;
                    }
                }
            } else {
                CalculateMBTRInnerLoop(atoms, z, v, current_rank + 1);
            }

            done:;
        }
    }

    void CalculateMBTRFirstLoopPeriodic(const std::vector <System::Atom> &atoms,
                                        uint z[], std::vector<double> &v) {

        for (size_t j = firstcell_atoms_index_begin_; j < firstcell_atoms_index_end_; ++j) {
            selected_atoms_[0] = &(atoms[j]);

            if (rank == 1) {
                double g = g_func_(selected_atoms_);
                double w = w_func_(selected_atoms_);

                double corr_coef = 1.0;
                for (uint i = 0; i < rank; ++i) {
                    corr_coef *= corr_func_(z[i], selected_atoms_[i]->atom_number);
                }

                if (std::abs(corr_coef) > 1e-6)
                    for (size_t i = 0; i < v.size(); ++i) {
                        v[i] += w * d_func_(bounds_[i], g) * corr_coef;
                    }
            } else {
                CalculateMBTRInnerLoop(atoms, z, v, 1);
            }
        }
    }

    void WriteMBTRToArray(double *array, uint current_rank,
                          std::vector <uint> &an, std::vector <uint> &idx) {

        double *current_column = array;
        size_t stride = bounds_.size();
        for (auto i = current_rank; i < rank; i++) {
            stride *= an.size();
        }

        for (auto i:an) {
            idx.push_back(i);

            if (rank == current_rank) {
                std::vector<double> v;
                v = operator()(&idx[0]);

                memcpy(current_column, &v[0], v.size() * sizeof(double));
            } else {
                WriteMBTRToArray(current_column, current_rank + 1, an, idx);
            }

            idx.pop_back();
            current_column += stride;
        }
    }

public:
    MBTR(const System &system, double start, double end, uint bins,
         g_func __g_func, w_func __w_func, d_func __d_func, corr_func __corr_func) :
            system_(system), g_func_(__g_func), w_func_(__w_func), d_func_(__d_func), corr_func_(__corr_func) {
        double delta = (end - start) / (bins - 1);
        for (uint i = 0; i < bins; ++i) {
            double a = start - delta / 2 + delta * i;
            double b = a + delta;
            bounds_.push_back(std::pair<double, double>(a, b));
        }
    }

    const g_func &g() const {
        return g_func_;
    }

    const w_func &w() const {
        return w_func_;
    }

    const d_func &d() const {
        return d_func_;
    }

    const corr_func &corr() const {
        return corr_func_;
    }

    std::vector<double> operator()(uint z[]) {
        std::vector<double> v(bounds_.size(), 0.0);

        // early termination
        for (uint i = 0; i < rank; i++) {
            bool found = false;
            for (size_t j = 0; j < system_.atoms.size(); j++) {
                if (std::abs(corr_func_(system_.atoms[j].atom_number, z[i])) > 1e-3) {
                    found = true;
                    break;
                }
            }
            if (!found)
                return v;
        }

        if (!system_.is_periodic) {
            CalculateMBTRInnerLoop(system_.atoms, z, v, 0);
        } else {
            if (reflected_cells_.size() == 0)
                GenerateReflectedImagesPeriodic();

            CalculateMBTRFirstLoopPeriodic(reflected_cells_, z, v);
        }

        return v;
    }

    std::vector <uint> NotNullAtomNumbers() {
        std::vector <uint> an;
        for (auto i:system_.atoms) {
            if (an.end() == std::find(an.begin(), an.end(), i.atom_number)) {
                an.push_back(i.atom_number);
            }
        }
        std::sort(an.begin(), an.end());
        return an;
    }

    double *WriteMBTRToArray(std::vector <uint> &an, size_t &array_size) {
        // FIXME: potential overflow.
        size_t buffer_total_size = bounds_.size();
        for (uint i = 0; i < rank; i++) {
            buffer_total_size *= an.size();
        }
        array_size = buffer_total_size;

        buffer_total_size *= sizeof(double);

        double *array = reinterpret_cast<double *>(PyMem_Malloc(buffer_total_size));
        if (array == nullptr)
            return nullptr;

        std::vector <uint> idx;
        WriteMBTRToArray(array, 1, an, idx);

        return array;
    }
};

#endif // #ifndef _MBTR_H_
