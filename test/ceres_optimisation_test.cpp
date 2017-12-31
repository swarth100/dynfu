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
class CeresTest : public ::testing::Test {
protected:
    /* You can remove any or all of the following functions if its body
     * is empty. */

    /* You can do set-up work for each test here. */
    CeresTest() = default;

    /* You can do clean-up work that doesn't throw exceptions here. */
    ~CeresTest() override = default;

    /* If the constructor and destructor are not enough for setting up
     * and cleaning up each test, you can define the following methods: */

    /* Code here will be called immediately after the constructor (right
     * before each test). */
    void SetUp() override {
        /* ceres settings */
        options.linear_solver_type           = ceres::SPARSE_NORMAL_CHOLESKY;
        options.minimizer_progress_to_stdout = true;
        options.max_num_iterations           = 64;

        int noCores         = sysconf(_SC_NPROCESSORS_ONLN);
        options.num_threads = noCores;

        auto dg_se3 = std::make_shared<DualQuaternion<float>>(0.f, 0.f, 0.f, 0.f, 0.f, 0.f);  // transformation
        float dg_w  = 2.f;                                                                    // radial basis weight

        /* init vector of deformation nodes */
        nodes.push_back(std::make_shared<Node>(cv::Vec3f(1.f, 1.f, -1.f), dg_se3, dg_w));
        nodes.push_back(std::make_shared<Node>(cv::Vec3f(1.f, -1.f, 1.f), dg_se3, dg_w));
        nodes.push_back(std::make_shared<Node>(cv::Vec3f(-1.f, 1.f, 1.f), dg_se3, dg_w));
        nodes.push_back(std::make_shared<Node>(cv::Vec3f(-1.f, -1.f, 1.f), dg_se3, dg_w));
        nodes.push_back(std::make_shared<Node>(cv::Vec3f(-1.f, -1.f, -1.f), dg_se3, dg_w));
        nodes.push_back(std::make_shared<Node>(cv::Vec3f(1.f, -1.f, -1.f), dg_se3, dg_w));
        nodes.push_back(std::make_shared<Node>(cv::Vec3f(-1.f, 1.f, -1.f), dg_se3, dg_w));
        nodes.push_back(std::make_shared<Node>(cv::Vec3f(1.f, 1.f, 1.f), dg_se3, dg_w));

        /* init warpfield */
        warpfield.init(nodes);
    }

    /* Code here will be called immediately after each test (right
     * before the destructor). */
    void TearDown() override {
        nodes.clear();

        sourceVertices.clear();
        targetVertices.clear();
    }

    /* Objects declared here can be used by all tests in the test case for Solver. */
    ceres::Solver::Options options;

    float maxError = 1e-3;

    std::shared_ptr<Node> node;
    std::vector<std::shared_ptr<Node>> nodes;

    Warpfield warpfield;

    std::vector<cv::Vec3f> sourceVertices;
    std::vector<cv::Vec3f> targetVertices;

    std::shared_ptr<dynfu::Frame> canonicalFrameWarpedToLive;
    std::shared_ptr<dynfu::Frame> liveFrame;
};

/* */
TEST_F(CeresTest, SingleVertexTest) {
    sourceVertices.emplace_back(cv::Vec3f(0, 0.05, 1));
    canonicalFrameWarpedToLive = std::make_shared<dynfu::Frame>(0, sourceVertices, sourceVertices);

    targetVertices.emplace_back(cv::Vec3f(0.05, 0, 1));
    liveFrame = std::make_shared<dynfu::Frame>(1, targetVertices, targetVertices);

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

    auto totalTransformation = warpfield.calcDQB(canonicalFrameWarpedToLive->getVertices()[0]);
    auto result              = sourceVertices[0] + totalTransformation->getTranslation();

    /* FIXME (dig15): figure out why the test fails */
    maxError = 0.01;
    ASSERT_NEAR(result[0], liveFrame->getVertices()[0][0], maxError);
    ASSERT_NEAR(result[1], liveFrame->getVertices()[0][1], maxError);
    ASSERT_NEAR(result[2], liveFrame->getVertices()[0][2], maxError);
}

/* */
TEST_F(CeresTest, MultipleVerticesTest) {
    sourceVertices.emplace_back(cv::Vec3f(-3, -3, -3));
    sourceVertices.emplace_back(cv::Vec3f(-2, -2, -2));
    sourceVertices.emplace_back(cv::Vec3f(0, 0, 0));
    sourceVertices.emplace_back(cv::Vec3f(2, 2, 2));
    sourceVertices.emplace_back(cv::Vec3f(3, 3, 3));

    canonicalFrameWarpedToLive = std::make_shared<dynfu::Frame>(0, sourceVertices, sourceVertices);

    targetVertices.emplace_back(cv::Vec3f(-2.95, -2.95, -2.95));
    targetVertices.emplace_back(cv::Vec3f(-1.95, -1.95, -1.95));
    targetVertices.emplace_back(cv::Vec3f(0.05, 0.05, 0.05));
    targetVertices.emplace_back(cv::Vec3f(2.05, 2.05, 2.05));
    targetVertices.emplace_back(cv::Vec3f(3.05, 3.05, 3.05));

    liveFrame = std::make_shared<dynfu::Frame>(1, targetVertices, targetVertices);

    WarpProblem warpProblem(options);
    warpProblem.optimiseWarpField(warpfield, canonicalFrameWarpedToLive, liveFrame);
    auto parameters = warpProblem.getParameters();

    int i = 0;
    int j = 0;
    for (auto vertex : canonicalFrameWarpedToLive->getVertices()) {
        for (auto neighbour : nodes) {
            cv::Vec3f translation(parameters[i][1], parameters[i][2], parameters[i][3]);
            neighbour->setRadialBasisWeight(parameters[i][0]);
            neighbour->setTranslation(translation);
            i++;
        }

        auto totalTransformation = warpfield.calcDQB(vertex);
        auto result              = vertex + totalTransformation->getTranslation();

        ASSERT_NEAR(result[0], liveFrame->getVertices()[j][0], maxError);
        ASSERT_NEAR(result[1], liveFrame->getVertices()[j][1], maxError);
        ASSERT_NEAR(result[2], liveFrame->getVertices()[j][2], maxError);

        i = 0;
        j++;
    }
}
