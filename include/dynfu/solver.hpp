#ifndef SOLVER_H
#define SOLVER_H

#include <dynfu/warp_field.hpp>

#include "ceres/ceres.h"

class Solver {
public:
    Solver(ceres::Solver::Options options);
    ~Solver();

    template <typename T>
    void calculateWarpToLive(WarpField::WarpField warpField, T LiveFrame);

private:
    ceres::Solver::Options options;
};

#endif
