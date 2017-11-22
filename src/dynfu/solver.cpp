#include <dynfu/solver.hpp>

/* TODO: Add comment */
Solver::Solver(ceres::Solver::Options options) {
    /* */
    this->options = options;
}

/* TODO: Add comment */
Solver::~Solver() = default;

/* TODO: Add comment */
void Solver::calculateWarpToLive(std::shared_ptr<Warpfield> warpField, std::shared_ptr<Frame> /*LiveFrame*/) {
    auto nodes        = warpField->getNodes();
    auto translations = 0;
    auto rotations    = 0;
    /*
    ceres::Problem problem;
    ceres::CostFunction *costFunction = new ceres::AutoDiffCostFunction<ceres::CostFunctor, 1, 1>(new CostFunctor);
    problem.AddResidualBlock(costFunction, NULL, &initialTransformation);

    ceres::Solver::Summary summary;
    ceres::Solve(this->options, &problem, &summary);

    std::cout << summary.BriefReport() << "\n";
    */
}
