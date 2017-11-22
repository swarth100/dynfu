#ifndef DYNFU_SOLVER_HPP
#define DYNFU_SOLVER_HPP

/* dynfu includes */
#include "dynfu/warp_field.hpp"

/* ceres includes */
#include <ceres/ceres.h>

/* */
template<typename T>
class Solver {
public:
    Solver(ceres::Solver::Options options);
    ~Solver();

    static ceres::CostFunction* Create(WarpField& warpField, std::shared_ptr<Frame> canonicalFrame,
                                       std::shared_ptr<Frame> liveFrame);

    void calculateWarpToLive(WarpField& warpField, std::shared_ptr<Frame> canonicalFrame,
                             std::shared_ptr<Frame> liveFrame);

private:
    ceres::Solver::Options options;
};

/* DYNFU_SOLVER_HPP */
#endif
