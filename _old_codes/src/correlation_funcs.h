#ifndef _CORRELATION_FUNCS_H_
#define _CORRELATION_FUNCS_H_

class DeltaCorrelation {
public:
    inline double operator()(int z1, int z2) {
        return z1 == z2 ? 1.0 : 0.0;
    }
};

#endif // _CORRELATION_FUNCS_H_
