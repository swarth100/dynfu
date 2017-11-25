#include <gtest/gtest.h>

#include <dynfu/utils/frame.hpp>
#include <dynfu/utils/node.hpp>
// #include <dynfu/utils/solver.hpp>

#include <dynfu/warp_field.hpp>

#include <ceres/ceres.h>

#include <tgmath.h>
#include <cmath>
#include <iostream>
#include <memory>

#define KNN 8

/*
 *
 */
struct WarpResidual {
    WarpResidual(std::shared_ptr<Node> node, cv::Vec3f sourceVertex, cv::Vec3f targetVertex) {
        x_      = sourceVertex[0];
        x_live_ = targetVertex[0];

        y_      = sourceVertex[1];
        y_live_ = targetVertex[1];

        z_      = sourceVertex[2];
        z_live_ = targetVertex[2];

        cv::Vec3f nodePosition =
            node->getPosition();  // get position of deformation node in order to calculate the extent of transformation
    }

    /*
     * calculates the residual of the linear system after applying weighted translation
     */
    template <typename T>
    bool operator()(const T* const translation, const T* const radial_basis_weight, T* residual) const {
        float exponent =
            pow(nodePosition[0] - x_live_, 2) + pow(nodePosition[1] - y_live_, 2) + pow(nodePosition[2] - z_live_, 2);

        T weight = T(exp(-T(exponent) / (T(2) * (*radial_basis_weight) * (*radial_basis_weight))));

        T predicted_x = x_live_ + translation[0] * weight;
        T predicted_y = y_live_ + translation[1] * weight;
        T predicted_z = z_live_ + translation[2] * weight;

        residual[0] = T(x_) - predicted_x;
        residual[1] = T(y_) - predicted_y;
        residual[2] = T(z_) - predicted_z;

        return true;
    }

private:
    double x_;
    double x_live_;
    double y_;
    double y_live_;
    double z_;
    double z_live_;

    cv::Vec3f nodePosition;
};

/* The fixture for testing class Quaternion. */
class SolverTest : public ::testing::Test {
protected:
    /* You can remove any or all of the following functions if its body
     * is empty. */

    /* You can do set-up work for each test here. */
    SolverTest() = default;

    /* You can do clean-up work that doesn't throw exceptions here. */
    ~SolverTest() override = default;

    /* If the constructor and destructor are not enough for setting up
     * and cleaning up each test, you can define the following methods: */

    /* Code here will be called immediately after the constructor (right
     * before each test). */
    void SetUp() override {}

    /* Code here will be called immediately after each test (right
     * before the destructor). */
    void TearDown() override {}

    /* Objects declared here can be used by all tests in the test case for Solver. */
    float max_error = 1e-3;

    std::vector<std::shared_ptr<Node>> nodes;

    std::shared_ptr<DualQuaternion<float>> dg_se3 =
        std::make_shared<DualQuaternion<float>>(0.f, 0.f, 0.f, 0.f, 0.f, 0.f);
    float dg_w = 2.f;

    Warpfield warpfield;

    std::shared_ptr<Frame> canonicalFrameWarpedToLive;
    std::shared_ptr<Frame> liveFrame;

    ceres::Problem problem;
    ceres::Solver::Options options;
    ceres::Solver::Summary summary;
};

/* */
TEST_F(SolverTest, SingleVertexTest) {
    cv::Vec3f dg_v = {1.f, 1.f, 1.f};

    auto node = std::make_shared<Node>(dg_v, dg_se3, dg_w);
    nodes.emplace_back(node);

    dg_v = {1.f, 1.f, -1.f};
    node = std::make_shared<Node>(dg_v, dg_se3, dg_w);
    nodes.emplace_back(node);

    dg_v = {1.f, -1.f, 1.f};
    node = std::make_shared<Node>(dg_v, dg_se3, dg_w);
    nodes.emplace_back(node);

    dg_v = {-1.f, 1.f, 1.f};
    node = std::make_shared<Node>(dg_v, dg_se3, dg_w);
    nodes.emplace_back(node);

    dg_v = {-1.f, -1.f, 1.f};
    node = std::make_shared<Node>(dg_v, dg_se3, dg_w);
    nodes.emplace_back(node);

    dg_v = {-1.f, -1.f, -1.f};
    node = std::make_shared<Node>(dg_v, dg_se3, dg_w);
    nodes.emplace_back(node);

    dg_v = {1.f, -1.f, -1.f};
    node = std::make_shared<Node>(dg_v, dg_se3, dg_w);
    nodes.emplace_back(node);

    dg_v = {-1, 1, -1};
    node = std::make_shared<Node>(dg_v, dg_se3, dg_w);
    nodes.emplace_back(node);

    warpfield.init(nodes);

    cv::Vec3f sourceVertex(0, 0, 0);
    std::vector<cv::Vec3f> sourceVertices;
    sourceVertices.emplace_back(sourceVertex);

    canonicalFrameWarpedToLive = std::make_shared<Frame>(0, sourceVertices, sourceVertices);

    cv::Vec3f targetVertex(0.05, 0.05, 0.05);
    std::vector<cv::Vec3f> targetVertices;
    targetVertices.emplace_back(targetVertex);

    liveFrame = std::make_shared<Frame>(1, targetVertices, targetVertices);

    double translation[3];
    double weight;

    for (auto vertex : liveFrame->getVertices()) {
        auto nearestNeighbours = warpfield.findNeighbors(KNN, vertex);

        for (auto node : nearestNeighbours) {
            translation[0] = (double) node->getTransformation()->getTranslation()[0];
            translation[1] = (double) node->getTransformation()->getTranslation()[1];
            translation[2] = (double) node->getTransformation()->getTranslation()[2];

            double radialBasisWeight = (double) node->getWeight();
            weight                   = radialBasisWeight;

            ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<WarpResidual, 3, 3, 1>(
                new WarpResidual(node, sourceVertex, targetVertex));
            problem.AddResidualBlock(cost_function, NULL, &translation[0], &weight);
        }
    }

    options.linear_solver_type           = ceres::SPARSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = true;

    Solve(options, &problem, &summary);

    std::cout << summary.BriefReport() << "\n";

    std::cout << translation[0] << std::endl;
    std::cout << translation[1] << std::endl;
    std::cout << translation[2] << std::endl;

    std::cout << weight << std::endl;

    for (auto vertex : sourceVertices) {
        ASSERT_NEAR(vertex[0], targetVertex[0], max_error);
        ASSERT_NEAR(vertex[1], targetVertex[1], max_error);
        ASSERT_NEAR(vertex[2], targetVertex[2], max_error);
    }
}

/* */
TEST_F(SolverTest, MultipleVerticesTest) {
    cv::Vec3f dg_v = {1.f, 1.f, 1.f};

    auto node = std::make_shared<Node>(dg_v, dg_se3, dg_w);
    nodes.emplace_back(node);

    dg_v = {1.f, 1.f, -1.f};
    node = std::make_shared<Node>(dg_v, dg_se3, dg_w);
    nodes.emplace_back(node);

    dg_v = {1.f, -1.f, 1.f};
    node = std::make_shared<Node>(dg_v, dg_se3, dg_w);
    nodes.emplace_back(node);

    dg_v = {-1.f, 1.f, 1.f};
    node = std::make_shared<Node>(dg_v, dg_se3, dg_w);
    nodes.emplace_back(node);

    dg_v = {-1.f, -1.f, 1.f};
    node = std::make_shared<Node>(dg_v, dg_se3, dg_w);
    nodes.emplace_back(node);

    dg_v = {-1.f, -1.f, -1.f};
    node = std::make_shared<Node>(dg_v, dg_se3, dg_w);
    nodes.emplace_back(node);

    dg_v = {1.f, -1.f, -1.f};
    node = std::make_shared<Node>(dg_v, dg_se3, dg_w);
    nodes.emplace_back(node);

    dg_v = {-1, 1, -1};
    node = std::make_shared<Node>(dg_v, dg_se3, dg_w);
    nodes.emplace_back(node);

    warpfield.init(nodes);

    std::vector<cv::Vec3f> sourceVertices;
    sourceVertices.emplace_back(cv::Vec3f(-3, -3, -3));
    sourceVertices.emplace_back(cv::Vec3f(-2, -2, -2));
    sourceVertices.emplace_back(cv::Vec3f(0, 0, 0));
    sourceVertices.emplace_back(cv::Vec3f(2, 2, 2));
    sourceVertices.emplace_back(cv::Vec3f(3, 3, 3));

    canonicalFrameWarpedToLive = std::make_shared<Frame>(0, sourceVertices, sourceVertices);

    std::vector<cv::Vec3f> targetVertices;
    targetVertices.emplace_back(cv::Vec3f(-2.95f, -2.95f, -2.95f));
    targetVertices.emplace_back(cv::Vec3f(-1.95f, -1.95f, -1.95f));
    targetVertices.emplace_back(cv::Vec3f(0.05, 0.05, 0.05));
    targetVertices.emplace_back(cv::Vec3f(2.05, 2.05, 2.05));
    targetVertices.emplace_back(cv::Vec3f(3.05, 3.05, 3.05));

    liveFrame = std::make_shared<Frame>(1, targetVertices, targetVertices);

    double translation[3];
    double weight;

    int i = 0;
    for (auto vertex : liveFrame->getVertices()) {
        auto nearestNeighbours = warpfield.findNeighbors(KNN, vertex);

        for (auto node : nearestNeighbours) {
            translation[0] = (double) node->getTransformation()->getTranslation()[0];
            translation[1] = (double) node->getTransformation()->getTranslation()[1];
            translation[2] = (double) node->getTransformation()->getTranslation()[2];

            double radialBasisWeight = (double) node->getWeight();
            weight                   = radialBasisWeight;

            ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<WarpResidual, 3, 3, 1>(
                new WarpResidual(node, sourceVertices[i], targetVertices[i]));
            problem.AddResidualBlock(cost_function, NULL, &translation[0], &weight);
        }

        i++;
    }

    options.linear_solver_type           = ceres::SPARSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = true;

    Solve(options, &problem, &summary);

    std::cout << summary.BriefReport() << "\n";

    std::cout << translation[0] << std::endl;
    std::cout << translation[1] << std::endl;
    std::cout << translation[2] << std::endl;

    std::cout << weight << std::endl;

    for (auto vertex : sourceVertices) {
        // ASSERT_NEAR(vertex[0], targetVertex[0], max_error);
        // ASSERT_NEAR(vertex[1], targetVertex[1], max_error);
        // ASSERT_NEAR(vertex[2], targetVertex[2], max_error);
    }
}
