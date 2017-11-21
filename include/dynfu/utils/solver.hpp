#ifndef DYNFU_SOLVER_H
#define DYNFU_SOLVER_H

/* dynfu includes */
#include <dynfu/utils/cost_functor.hpp>
#include <dynfu/utils/frame.hpp>
#include <dynfu/warp_field.hpp>

/* ceres includes */
#include <ceres/ceres.h>

/* */
class Solver {
public:
    Solver(ceres::Solver::Options options);
    ~Solver();

    static ceres::CostFunction* Create(WarpField warpField, std::shared_ptr<Frame> canonicalFrame,
                                       std::shared_ptr<Frame> liveFrame);

    void calculateWarpToLive(WarpField& warpField, std::shared_ptr<Frame> canonicalFrame,
                             std::shared_ptr<Frame> liveFrame);

private:
    ceres::Solver::Options options;
};

/* DYNFU_SOLVER_H */
#endif
