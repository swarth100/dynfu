#ifndef DYNFU_SOLVER_HPP
#define DYNFU_SOLVER_HPP

/* dynfu includes */
#include <dynfu/utils/cost_functor.hpp>
#include <dynfu/warp_field.hpp>

/* ceres includes */
#include <ceres/ceres.h>

#include <iostream>

#define KNN 8
/* */
template <class T>
class Solver {
private:
    T* parameters;

public:
    Solver()  = default;
    ~Solver() = default;

    static ceres::CostFunction* Create(Warpfield warpfield, std::shared_ptr<Frame> canonicalFrame,
                                       std::shared_ptr<Frame> liveFrame) {
        auto costFunction = new ceres::DynamicAutoDiffCostFunction<CostFunctor, 3>(new CostFunctor);

        int numResiduals = 3;
        int noParameters = KNN * 6;

        for (int i = 0; i < KNN; i++) {
            costFunction->AddParameterBlock(noParameters);
        }

        costFunction->SetNumResiduals(numResiduals);

        return costFunction;
    }

    void calculateWarpToLive(Warpfield warpfield, std::shared_ptr<Frame> canonicalFrame,
                             std::shared_ptr<Frame> liveFrame) {
        ceres::Problem problem;

        double x[] = {double(parameters[0]), double(parameters[1]), double(parameters[2])};
        problem.AddResidualBlock(Create(warpfield, canonicalFrame, liveFrame), NULL, x);

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        ceres::Solver::Summary summary;

        ceres::Solve(options, &problem, &summary);

        // set the transformations to the parametres obtained from ceres
        auto nodes = warpfield.getNodes();

        int i = 0;
        for (auto node : nodes) {
            auto transformation = std::make_shared<DualQuaternion<float>>(0.f, 0.f, 0.f, 0.f, 0.f, 0.f);
            node->setTransformation(transformation);

            i++;
        }
    }

    // std::cout << summary.FullReport() << "\n";
};

/* DYNFU_SOLVER_HPP */
#endif
