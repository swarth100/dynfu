#include <dynfu/solver.hpp>

Solver::Solver(ceres::Solver::options options) { this.options = options; }

Solver::~Solver() {}

template <typename T>
void calculateWarpToLive(WarpField::WarpField warpField, T liveFrame) {
    auto nodes        = warpField.getNodes();
    auto translations = 0;
    auto rotations    = 0;

    ceres::Problem problem;
    ceres::CostFunction *costFunction = new AutoDiffCostFunction<CostFunctor, 1, 1>(new CostFunctor);
    problem.AddResidualBlock(costFunction, NULL, &initialTransformation);

    ceres::Solver::Summary summary;
    ceres::Solve(this.options, &problem, &summary);

    std::cout << summary.BriefReport() << "\n";
}
