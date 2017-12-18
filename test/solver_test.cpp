/* gtest includes */
#include <gtest/gtest.h>

/* dynfu includes */
#include <dynfu/utils/ceres_solver.hpp>
#include <dynfu/utils/frame.hpp>
#include <dynfu/utils/node.hpp>

#include <dynfu/warp_field.hpp>

/* sys headers */
#include <cmath>
#include <ctgmath>
#include <iostream>
#include <memory>

#define KNN 8

/* fixture for testing class WarpProblem */
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
    void SetUp() override {
        options.linear_solver_type           = ceres::SPARSE_NORMAL_CHOLESKY;
        options.minimizer_progress_to_stdout = true;
        options.max_num_iterations           = 64;

        int noCores         = sysconf(_SC_NPROCESSORS_ONLN);
        options.num_threads = noCores;
    }

    /* Code here will be called immediately after each test (right
     * before the destructor). */
    void TearDown() override {}

    /* Objects declared here can be used by all tests in the test case for Solver. */
    float max_error = 1e-3;

    // std::vector<std::shared_ptr<Node>> nodes;

    std::shared_ptr<DualQuaternion<float>> dg_se3 =
        std::make_shared<DualQuaternion<float>>(0.f, 0.f, 0.f, 0.f, 0.f, 0.f);
    float dg_w = 2.f;

    std::shared_ptr<Frame> canonicalFrameWarpedToLive;
    std::shared_ptr<Frame> liveFrame;

    ceres::Solver::Options options;
};

/* */
TEST_F(SolverTest, SingleVertexTest) {
    Warpfield warpfield;
    std::vector<cv::Vec3f> sourceVertices;
    std::vector<cv::Vec3f> targetVertices;

    cv::Vec3f dg_v = {1.f, 1.f, 1.f};

    std::vector<std::shared_ptr<Node>> nodes;

    std::shared_ptr<Node> node = std::make_shared<Node>(dg_v, dg_se3, dg_w);
    nodes.push_back(node);

    dg_v = {1.f, 1.f, -1.f};
    node = std::make_shared<Node>(dg_v, dg_se3, dg_w);
    nodes.push_back(node);

    dg_v = {1.f, -1.f, 1.f};
    node = std::make_shared<Node>(dg_v, dg_se3, dg_w);
    nodes.push_back(node);

    dg_v = {-1.f, 1.f, 1.f};
    node = std::make_shared<Node>(dg_v, dg_se3, dg_w);
    nodes.push_back(node);

    dg_v = {-1.f, -1.f, 1.f};
    node = std::make_shared<Node>(dg_v, dg_se3, dg_w);
    nodes.push_back(node);

    dg_v = {-1.f, -1.f, -1.f};
    node = std::make_shared<Node>(dg_v, dg_se3, dg_w);
    nodes.push_back(node);

    dg_v = {1.f, -1.f, -1.f};
    node = std::make_shared<Node>(dg_v, dg_se3, dg_w);
    nodes.push_back(node);

    dg_v = {-1.f, 1.f, -1.f};
    node = std::make_shared<Node>(dg_v, dg_se3, dg_w);
    nodes.push_back(node);

    warpfield.init(nodes);

    cv::Vec3f sourceVertex(1.05, 0.05, 1);
    sourceVertices.emplace_back(sourceVertex);

    canonicalFrameWarpedToLive = std::make_shared<Frame>(0, sourceVertices, sourceVertices);

    cv::Vec3f targetVertex(1.0, 0.0, 1.0);
    targetVertices.emplace_back(targetVertex);

    liveFrame = std::make_shared<Frame>(1, targetVertices, targetVertices);

    WarpProblem warpProblem(options);
    warpProblem.optimiseWarpField(warpfield, canonicalFrameWarpedToLive, liveFrame);

    auto parameters = warpProblem.getParameters();

    int i = 0;
    for (auto neighbour : nodes) {
        cv::Vec3f translation(parameters[i][1], parameters[i][2], parameters[i][3]);
        neighbour->setRadialBasisWeight(parameters[i][0]);
        neighbour->setTranslation(translation);
        i++;
    }

    auto totalTransformation = warpfield.calcDQB(sourceVertex);
    auto result              = sourceVertex + totalTransformation->getTranslation();

    max_error = 0.01;
    ASSERT_NEAR(result[0], targetVertices[0][0], max_error);
    ASSERT_NEAR(result[1], targetVertices[0][1], max_error);
    ASSERT_NEAR(result[2], targetVertices[0][2], max_error);
}

/* */
TEST_F(SolverTest, MultipleVerticesTest) {
    Warpfield warpfield;
    std::vector<cv::Vec3f> sourceVertices;
    std::vector<cv::Vec3f> targetVertices;

    std::vector<std::shared_ptr<Node>> nodes;

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

    dg_v = {-1.f, 1.f, -1.f};
    node = std::make_shared<Node>(dg_v, dg_se3, dg_w);
    nodes.emplace_back(node);

    warpfield.init(nodes);

    sourceVertices.emplace_back(cv::Vec3f(-3, -3, -3));
    sourceVertices.emplace_back(cv::Vec3f(-2, -2, -2));
    sourceVertices.emplace_back(cv::Vec3f(0, 0, 0));
    sourceVertices.emplace_back(cv::Vec3f(2, 2, 2));
    sourceVertices.emplace_back(cv::Vec3f(3, 3, 3));

    canonicalFrameWarpedToLive = std::make_shared<Frame>(0, sourceVertices, sourceVertices);

    targetVertices.emplace_back(cv::Vec3f(-2.95f, -2.95f, -2.95f));
    targetVertices.emplace_back(cv::Vec3f(-1.95f, -1.95f, -1.95f));
    targetVertices.emplace_back(cv::Vec3f(0.05, 0.05, 0.05));
    targetVertices.emplace_back(cv::Vec3f(2.05, 2.05, 2.05));
    targetVertices.emplace_back(cv::Vec3f(3.05, 3.05, 3.05));

    liveFrame = std::make_shared<Frame>(1, targetVertices, targetVertices);

    WarpProblem warpProblem(options);
    warpProblem.optimiseWarpField(warpfield, canonicalFrameWarpedToLive, liveFrame);
    auto parameters = warpProblem.getParameters();

    int i = 0;
    int j = 0;
    for (auto vertex : sourceVertices) {
        cv::Vec3f totalTranslation;

        for (auto neighbour : nodes) {
            cv::Vec3f translation(parameters[i][1], parameters[i][2], parameters[i][3]);
            neighbour->setRadialBasisWeight(parameters[i][0]);
            neighbour->setTranslation(translation);

            totalTranslation += translation * neighbour->getTransformationWeight(vertex);
            i++;
        }

        ASSERT_NEAR((vertex + totalTranslation)[0], targetVertices[j][0], max_error);
        ASSERT_NEAR((vertex + totalTranslation)[1], targetVertices[j][1], max_error);
        ASSERT_NEAR((vertex + totalTranslation)[2], targetVertices[j][2], max_error);

        i = 0;
        j++;
    }
}
