#ifndef DYNFU_SOLVER_HPP
#define DYNFU_SOLVER_HPP

/* dynfu includes */
#include <dynfu/utils/cost_functor.hpp>
#include <dynfu/warp_field.hpp>

/* ceres includes */
#include <ceres/ceres.h>

#define KNN 8
/* */
template <class T>
class Solver {
private:
    static T const* const* parameters;

public:
    Solver();
    ~Solver();

    static ceres::CostFunction* Create(std::shared_ptr<Warpfield> warpfield, std::shared_ptr<Frame> canonicalFrame,
                                       std::shared_ptr<Frame> liveFrame) {
        auto costFunctor   = new CostFunctor<float>(warpfield, canonicalFrame, liveFrame);
        Solver::parameters = costFunctor->getParameters();
        auto costFunction  = new ceres::DynamicAutoDiffCostFunction<CostFunctor<float>, 3>(costFunctor);

        int numResiduals = 3;
        int noParameters = KNN * 6;

        for (int i = 0; i < KNN; i++) {
            costFunction->AddParameterBlock(noParameters);
        }

        costFunction->SetNumResiduals(numResiduals);

        return costFunction;
    }

    void calculateWarpToLive(std::shared_ptr<Warpfield> warpfield, std::shared_ptr<Frame> canonicalFrame,
                             std::shared_ptr<Frame> liveFrame) {
        ceres::CostFunction* costFunction = this->Create(warpfield, canonicalFrame, liveFrame);

        ceres::Problem problem;
        problem.AddResidualBlock(costFunction, NULL, parameters);

        ceres::Solver::Summary summary;

        ceres::Solve(this.options, &problem, &summary);

        std::cout << summary.FullReport() << "\n";
    }
};

/* DYNFU_SOLVER_HPP */
#endif
