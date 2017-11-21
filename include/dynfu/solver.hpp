#ifndef DYNFU_SOLVER_H
#define DYNFU_SOLVER_H

/* Dynfu Includes */
#include <dynfu/utils/frame.hpp>
#include <dynfu/warp_field.hpp>

/* Ceres Includes */
#include <ceres/ceres.h>

/* */
class Solver {
public:
    Solver(ceres::Solver::Options options);
    ~Solver();

    void calculateWarpToLive(Warpfield warpField, std::shared_ptr<Frame> LiveFrame);

private:
    ceres::Solver::Options options;
};

/* DYNFU_SOLVER_H */
#endif
