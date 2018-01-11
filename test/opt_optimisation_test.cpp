/* gtest includes */
#include <gtest/gtest.h>

/* dynfu includes */
#include <dynfu/utils/frame.hpp>
#include <dynfu/utils/node.hpp>
#include <dynfu/utils/opt_solver.hpp>

#include <dynfu/warp_field.hpp>

/* sys headers */
#include <cmath>
#include <ctgmath>
#include <iostream>
#include <memory>

#define KNN 8

/* fixture for testing warping with opt */
class OptTest : public ::testing::Test {
protected:
    /* You can remove any or all of the following functions if its body
     * is empty. */

    /* You can do set-up work for each test here. */
    OptTest() = default;

    /* You can do clean-up work that doesn't throw exceptions here. */
    ~OptTest() override = default;

    /* If the constructor and destructor are not enough for setting up
     * and cleaning up each test, you can define the following methods: */

    /* Code here will be called immediately after the constructor (right
     * before each test). */
    void SetUp() override {
        /* opt settings */
        params.numIter            = 32;
        params.nonLinearIter      = 16;
        params.linearIter         = 256;
        params.useOpt             = false;
        params.useOptLM           = true;
        params.earlyOut           = true;
        params.optDoublePrecision = true;

        auto dg_se3 = std::make_shared<DualQuaternion<float>>(0.f, 0.f, 0.f, 0.f, 0.f, 0.f);  // transformation
        float dg_w  = 2.f;                                                                    // radial basis weight

        /* init node group 1 */
        nodesGroup1.push_back(std::make_shared<Node>(cv::Vec3f(1.f, 1.f, -1.f), dg_se3, dg_w));
        nodesGroup1.push_back(std::make_shared<Node>(cv::Vec3f(1.f, -1.f, 1.f), dg_se3, dg_w));
        nodesGroup1.push_back(std::make_shared<Node>(cv::Vec3f(-1.f, 1.f, 1.f), dg_se3, dg_w));
        nodesGroup1.push_back(std::make_shared<Node>(cv::Vec3f(-1.f, -1.f, 1.f), dg_se3, dg_w));
        nodesGroup1.push_back(std::make_shared<Node>(cv::Vec3f(-1.f, -1.f, -1.f), dg_se3, dg_w));
        nodesGroup1.push_back(std::make_shared<Node>(cv::Vec3f(1.f, -1.f, -1.f), dg_se3, dg_w));
        nodesGroup1.push_back(std::make_shared<Node>(cv::Vec3f(-1.f, 1.f, -1.f), dg_se3, dg_w));
        nodesGroup1.push_back(std::make_shared<Node>(cv::Vec3f(1.f, 1.f, 1.f), dg_se3, dg_w));

        /* init node group 2 */
        nodesGroup2.push_back(std::make_shared<Node>(cv::Vec3f(10.f, 10.f, 10.f), dg_se3, dg_w));
        nodesGroup2.push_back(std::make_shared<Node>(cv::Vec3f(9.f, 10.f, 10.f), dg_se3, dg_w));
        nodesGroup2.push_back(std::make_shared<Node>(cv::Vec3f(10.f, 9.f, 10.f), dg_se3, dg_w));
        nodesGroup2.push_back(std::make_shared<Node>(cv::Vec3f(10.f, 10.f, 9.f), dg_se3, dg_w));
        nodesGroup2.push_back(std::make_shared<Node>(cv::Vec3f(9.f, 9.f, 10.f), dg_se3, dg_w));
        nodesGroup2.push_back(std::make_shared<Node>(cv::Vec3f(9.f, 10.f, 9.f), dg_se3, dg_w));
        nodesGroup2.push_back(std::make_shared<Node>(cv::Vec3f(9.f, 9.f, 9.f), dg_se3, dg_w));
        nodesGroup2.push_back(std::make_shared<Node>(cv::Vec3f(9.f, 9.f, 9.f), dg_se3, dg_w));

        /* init all nodes */
        allNodes = nodesGroup1;
        allNodes.insert(allNodes.end(), nodesGroup2.begin(), nodesGroup2.end());
    }

    /* Code here will be called immediately after each test (right
     * before the destructor). */
    void TearDown() override {
        nodesGroup1.clear();
        nodesGroup2.clear();
        allNodes.clear();

        sourceVertices.clear();
        sourceNormals.clear();

        targetVertices.clear();
        targetNormals.clear();
    }

    /* Objects declared here can be used by all tests in the test case for Solver. */
    CombinedSolverParameters params;

    float maxError = 1e-3;

    std::shared_ptr<Node> node;

    std::vector<std::shared_ptr<Node>> nodesGroup1;
    std::vector<std::shared_ptr<Node>> nodesGroup2;
    std::vector<std::shared_ptr<Node>> allNodes;

    Warpfield warpfield;

    std::vector<cv::Vec3f> sourceVertices;
    std::vector<cv::Vec3f> sourceNormals;

    std::vector<cv::Vec3f> targetVertices;
    std::vector<cv::Vec3f> targetNormals;

    std::shared_ptr<dynfu::Frame> canonicalFrameWarpedToLive;
    std::shared_ptr<dynfu::Frame> liveFrame;
};

/* */
TEST_F(OptTest, SingleVertexOneGroupOfDeformationNodesTest) {
    warpfield.init(nodesGroup1);

    sourceVertices.emplace_back(cv::Vec3f(0, 0.05, 1));
    sourceNormals.emplace_back(cv::Vec3f(1, 1, 1));

    canonicalFrameWarpedToLive = std::make_shared<dynfu::Frame>(0, sourceVertices, sourceVertices);

    targetVertices.emplace_back(cv::Vec3f(0.05, 0.0, 1));
    targetNormals.emplace_back(cv::Vec3f(1, 1, 1));

    liveFrame = std::make_shared<dynfu::Frame>(1, targetVertices, targetVertices);

    CombinedSolver combinedSolver(warpfield, params);
    combinedSolver.initializeProblemInstance(canonicalFrameWarpedToLive, liveFrame);
    combinedSolver.solveAll();

    /* FIXME (dig15): figure out why the test fails */
    int j = 0;
    for (auto vertex : canonicalFrameWarpedToLive->getVertices()) {
        auto totalTransformation = warpfield.calcDQB(vertex);
        auto result              = vertex + totalTransformation->getTranslation();

        ASSERT_NEAR(result[0], liveFrame->getVertices()[j][0], maxError);
        ASSERT_NEAR(result[1], liveFrame->getVertices()[j][1], maxError);
        ASSERT_NEAR(result[2], liveFrame->getVertices()[j][2], maxError);

        j++;
    }
}

/* */
TEST_F(OptTest, MultipleVerticesOneGroupOfDeformationNodesTest) {
    warpfield.init(nodesGroup1);

    sourceVertices.emplace_back(cv::Vec3f(-3, -3, -3));
    sourceVertices.emplace_back(cv::Vec3f(-2, -2, -2));
    sourceVertices.emplace_back(cv::Vec3f(0.01, 0.01, 0.01));
    sourceVertices.emplace_back(cv::Vec3f(2, 2, 2));
    sourceVertices.emplace_back(cv::Vec3f(3, 3, 3));

    canonicalFrameWarpedToLive = std::make_shared<dynfu::Frame>(0, sourceVertices, sourceVertices);

    targetVertices.emplace_back(cv::Vec3f(-2.95, -2.95, -2.95));
    targetVertices.emplace_back(cv::Vec3f(-1.95, -1.95, -1.95));
    targetVertices.emplace_back(cv::Vec3f(0.05, 0.05, 0.05));
    targetVertices.emplace_back(cv::Vec3f(2.05, 2.05, 2.05));
    targetVertices.emplace_back(cv::Vec3f(3.05, 3.05, 3.05));

    liveFrame = std::make_shared<dynfu::Frame>(1, targetVertices, targetVertices);

    CombinedSolver combinedSolver(warpfield, params);
    combinedSolver.initializeProblemInstance(canonicalFrameWarpedToLive, liveFrame);
    combinedSolver.solveAll();

    int j = 0;
    for (auto vertex : canonicalFrameWarpedToLive->getVertices()) {
        auto totalTransformation = warpfield.calcDQB(vertex);
        auto result              = vertex + totalTransformation->getTranslation();

        ASSERT_NEAR(result[0], liveFrame->getVertices()[j][0], maxError);
        ASSERT_NEAR(result[1], liveFrame->getVertices()[j][1], maxError);
        ASSERT_NEAR(result[2], liveFrame->getVertices()[j][2], maxError);

        j++;
    }
}

/* */
TEST_F(OptTest, OneGroupOfVerticesTwoGroupsOfDeformationNodes) {
    warpfield.init(allNodes);

    sourceVertices.emplace_back(cv::Vec3f(-3, -3, -3));
    sourceVertices.emplace_back(cv::Vec3f(-2, -2, -2));
    sourceVertices.emplace_back(cv::Vec3f(0.01, 0.01, 0.01));
    sourceVertices.emplace_back(cv::Vec3f(2, 2, 2));
    sourceVertices.emplace_back(cv::Vec3f(3, 3, 3));

    canonicalFrameWarpedToLive = std::make_shared<dynfu::Frame>(0, sourceVertices, sourceVertices);

    targetVertices.emplace_back(cv::Vec3f(-2.95, -2.95, -2.95));
    targetVertices.emplace_back(cv::Vec3f(-1.95, -1.95, -1.95));
    targetVertices.emplace_back(cv::Vec3f(0.05, 0.05, 0.05));
    targetVertices.emplace_back(cv::Vec3f(2.05, 2.05, 2.05));
    targetVertices.emplace_back(cv::Vec3f(3.05, 3.05, 3.05));

    liveFrame = std::make_shared<dynfu::Frame>(1, targetVertices, targetVertices);

    CombinedSolver combinedSolver(warpfield, params);
    combinedSolver.initializeProblemInstance(canonicalFrameWarpedToLive, liveFrame);
    combinedSolver.solveAll();

    int j = 0;
    for (auto vertex : canonicalFrameWarpedToLive->getVertices()) {
        auto totalTransformation = warpfield.calcDQB(vertex);
        auto result              = vertex + totalTransformation->getTranslation();

        ASSERT_NEAR(result[0], liveFrame->getVertices()[j][0], maxError);
        ASSERT_NEAR(result[1], liveFrame->getVertices()[j][1], maxError);
        ASSERT_NEAR(result[2], liveFrame->getVertices()[j][2], maxError);

        j++;
    }
}

/* */
TEST_F(OptTest, TwoGroupsOfVerticesTwoGroupsOfDeformationNodes) {
    warpfield.init(allNodes);

    /* group 1 of source vertices */
    sourceVertices.emplace_back(cv::Vec3f(-3, -3, -3));
    sourceVertices.emplace_back(cv::Vec3f(-2, -2, -2));
    sourceVertices.emplace_back(cv::Vec3f(0.01, 0.01, 0.01));
    sourceVertices.emplace_back(cv::Vec3f(2, 2, 2));
    sourceVertices.emplace_back(cv::Vec3f(3, 3, 3));

    /* group 2 of source vertices */
    sourceVertices.emplace_back(cv::Vec3f(12, 12, 12));
    sourceVertices.emplace_back(cv::Vec3f(11, 11, 11));
    sourceVertices.emplace_back(cv::Vec3f(10, 10, 10));
    sourceVertices.emplace_back(cv::Vec3f(10.5, 10.5, 10.5));
    sourceVertices.emplace_back(cv::Vec3f(11.5, 11.5, 11.5));

    canonicalFrameWarpedToLive = std::make_shared<dynfu::Frame>(0, sourceVertices, sourceVertices);

    /* group 1 of target vertices */
    targetVertices.emplace_back(cv::Vec3f(-2.95, -2.95, -2.95));
    targetVertices.emplace_back(cv::Vec3f(-1.95, -1.95, -1.95));
    targetVertices.emplace_back(cv::Vec3f(0.05, 0.05, 0.05));
    targetVertices.emplace_back(cv::Vec3f(2.05, 2.05, 2.05));
    targetVertices.emplace_back(cv::Vec3f(3.05, 3.05, 3.05));

    /* group 2 of target vertices */
    targetVertices.emplace_back(cv::Vec3f(11.5, 11.5, 11.5));
    targetVertices.emplace_back(cv::Vec3f(10.5, 10.5, 10.5));
    targetVertices.emplace_back(cv::Vec3f(9.5, 9.5, 9.5));
    targetVertices.emplace_back(cv::Vec3f(10, 10, 10));
    targetVertices.emplace_back(cv::Vec3f(11, 11, 11));

    liveFrame = std::make_shared<dynfu::Frame>(1, targetVertices, targetVertices);

    CombinedSolver combinedSolver(warpfield, params);
    combinedSolver.initializeProblemInstance(canonicalFrameWarpedToLive, liveFrame);
    combinedSolver.solveAll();

    /* TODO (dig15): figure out why the test fails */
    maxError = 1.0;

    int j = 0;
    for (auto vertex : canonicalFrameWarpedToLive->getVertices()) {
        auto totalTransformation = warpfield.calcDQB(vertex);
        auto result              = vertex + totalTransformation->getTranslation();

        ASSERT_NEAR(result[0], liveFrame->getVertices()[j][0], maxError);
        ASSERT_NEAR(result[1], liveFrame->getVertices()[j][1], maxError);
        ASSERT_NEAR(result[2], liveFrame->getVertices()[j][2], maxError);

        j++;
    }
}

/* FIXME (dig15): figure out why the test fails */
TEST_F(OptTest, MultipleVerticesOneGroupOfDeformationNodesWarpAndReverseTest) {
    warpfield.init(nodesGroup1);

    sourceVertices.emplace_back(cv::Vec3f(-3, -3, -3));
    sourceVertices.emplace_back(cv::Vec3f(-2, -2, -2));
    sourceVertices.emplace_back(cv::Vec3f(0.01, 0.01, 0.01));
    sourceVertices.emplace_back(cv::Vec3f(2, 2, 2));
    sourceVertices.emplace_back(cv::Vec3f(3, 3, 3));

    canonicalFrameWarpedToLive = std::make_shared<dynfu::Frame>(0, sourceVertices, sourceVertices);

    targetVertices.emplace_back(cv::Vec3f(-2.95, -2.95, -2.95));
    targetVertices.emplace_back(cv::Vec3f(-1.95, -1.95, -1.95));
    targetVertices.emplace_back(cv::Vec3f(0.05, 0.05, 0.05));
    targetVertices.emplace_back(cv::Vec3f(2.05, 2.05, 2.05));
    targetVertices.emplace_back(cv::Vec3f(3.05, 3.05, 3.05));

    liveFrame = std::make_shared<dynfu::Frame>(1, targetVertices, targetVertices);

    CombinedSolver combinedSolver(warpfield, params);
    combinedSolver.initializeProblemInstance(canonicalFrameWarpedToLive, liveFrame);
    combinedSolver.solveAll();

    /* TODO (dig15): figure out why the test fails */
    maxError = 1.0;

    int j = 0;
    for (auto vertex : canonicalFrameWarpedToLive->getVertices()) {
        auto totalTransformation = warpfield.calcDQB(vertex);
        auto result              = vertex + totalTransformation->getTranslation();

        ASSERT_NEAR(result[0], liveFrame->getVertices()[j][0], maxError);
        ASSERT_NEAR(result[1], liveFrame->getVertices()[j][1], maxError);
        ASSERT_NEAR(result[2], liveFrame->getVertices()[j][2], maxError);

        j++;
    }

    /* reverse */
    auto temp                  = liveFrame;
    liveFrame                  = canonicalFrameWarpedToLive;
    canonicalFrameWarpedToLive = temp;

    CombinedSolver combinedSolverReverse(warpfield, params);
    combinedSolverReverse.initializeProblemInstance(canonicalFrameWarpedToLive, liveFrame);
    combinedSolverReverse.solveAll();

    j = 0;
    for (auto vertex : canonicalFrameWarpedToLive->getVertices()) {
        auto totalTransformation = warpfield.calcDQB(vertex);
        auto result              = vertex + totalTransformation->getTranslation();

        ASSERT_NEAR(result[0], liveFrame->getVertices()[j][0], maxError);
        ASSERT_NEAR(result[1], liveFrame->getVertices()[j][1], maxError);
        ASSERT_NEAR(result[2], liveFrame->getVertices()[j][2], maxError);

        j++;
    }
}

/* */
TEST_F(OptTest, MultipleVerticesOneGroupOfDeformationNodesNonRigidTest) {
    warpfield.init(nodesGroup1);

    sourceVertices.emplace_back(cv::Vec3f(-3, -3, -3));
    sourceVertices.emplace_back(cv::Vec3f(-2, -2, -2));
    sourceVertices.emplace_back(cv::Vec3f(0.01, 0.01, 0.01));
    sourceVertices.emplace_back(cv::Vec3f(2, 2, 2));
    sourceVertices.emplace_back(cv::Vec3f(3, 3, 3));

    canonicalFrameWarpedToLive = std::make_shared<dynfu::Frame>(0, sourceVertices, sourceVertices);

    targetVertices.emplace_back(cv::Vec3f(-2.95, -2.95, -2.95));
    targetVertices.emplace_back(cv::Vec3f(-2.0, -2.0, -2.0));
    targetVertices.emplace_back(cv::Vec3f(0.01, 0, 0));
    targetVertices.emplace_back(cv::Vec3f(2.5, 2.5, 2.5));
    targetVertices.emplace_back(cv::Vec3f(2.75, 2.75, 2.75));

    liveFrame = std::make_shared<dynfu::Frame>(1, targetVertices, targetVertices);

    CombinedSolver combinedSolver(warpfield, params);
    combinedSolver.initializeProblemInstance(canonicalFrameWarpedToLive, liveFrame);
    combinedSolver.solveAll();

    int j = 0;
    for (auto vertex : canonicalFrameWarpedToLive->getVertices()) {
        auto totalTransformation = warpfield.calcDQB(vertex);
        auto result              = vertex + totalTransformation->getTranslation();

        ASSERT_NEAR(result[0], liveFrame->getVertices()[j][0], maxError);
        ASSERT_NEAR(result[1], liveFrame->getVertices()[j][1], maxError);
        ASSERT_NEAR(result[2], liveFrame->getVertices()[j][2], maxError);

        j++;
    }
}

TEST_F(OptTest, RandomNonRigidTest) {
    std::vector<std::shared_ptr<Node>> nodesNonRigid;

    auto dg_se3 = std::make_shared<DualQuaternion<float>>(0.f, 0.f, 0.f, 0.f, 0.f, 0.f);  // transformation
    float dg_w  = 2.f;                                                                    // radial basis weight

    nodesNonRigid.push_back(std::make_shared<Node>(cv::Vec3f(1, 1, 1), dg_se3, dg_w));
    nodesNonRigid.push_back(std::make_shared<Node>(cv::Vec3f(1, 2, -1), dg_se3, dg_w));
    nodesNonRigid.push_back(std::make_shared<Node>(cv::Vec3f(1, -2, 1), dg_se3, dg_w));
    nodesNonRigid.push_back(std::make_shared<Node>(cv::Vec3f(1, -1, -1), dg_se3, dg_w));
    nodesNonRigid.push_back(std::make_shared<Node>(cv::Vec3f(-1, 1, 5), dg_se3, dg_w));
    nodesNonRigid.push_back(std::make_shared<Node>(cv::Vec3f(-1, 1, -1), dg_se3, dg_w));
    nodesNonRigid.push_back(std::make_shared<Node>(cv::Vec3f(-1, 1, 5), dg_se3, dg_w));
    nodesNonRigid.push_back(std::make_shared<Node>(cv::Vec3f(-1, -1, 1), dg_se3, dg_w));
    nodesNonRigid.push_back(std::make_shared<Node>(cv::Vec3f(-1, 1, 5), dg_se3, dg_w));
    nodesNonRigid.push_back(std::make_shared<Node>(cv::Vec3f(-1, -1, 1), dg_se3, dg_w));
    nodesNonRigid.push_back(std::make_shared<Node>(cv::Vec3f(-1, -1, -1), dg_se3, dg_w));
    nodesNonRigid.push_back(std::make_shared<Node>(cv::Vec3f(2, -3, -1), dg_se3, dg_w));

    warpfield.init(nodesNonRigid);

    sourceVertices.emplace_back(cv::Vec3f(-3, -3, -3));
    sourceVertices.emplace_back(cv::Vec3f(-2, -2, -2));
    sourceVertices.emplace_back(cv::Vec3f(0.01, 0.01, 0.01));
    sourceVertices.emplace_back(cv::Vec3f(2, 2, 2));
    sourceVertices.emplace_back(cv::Vec3f(3, 3, 3));
    sourceVertices.emplace_back(cv::Vec3f(3, 3, 3));

    canonicalFrameWarpedToLive = std::make_shared<dynfu::Frame>(0, sourceVertices, sourceVertices);

    targetVertices.emplace_back(cv::Vec3f(-2.95, -3.f, -2.95));
    targetVertices.emplace_back(cv::Vec3f(-1.95, -1.95, -2.f));
    targetVertices.emplace_back(cv::Vec3f(0.1, 0.1, 0.1));
    targetVertices.emplace_back(cv::Vec3f(2, 2., 2));
    targetVertices.emplace_back(cv::Vec3f(3.05, 3.05, 3.05));
    targetVertices.emplace_back(cv::Vec3f(3.05, 3.05, 3.05));

    liveFrame = std::make_shared<dynfu::Frame>(0, targetVertices, targetVertices);

    CombinedSolver combinedSolver(warpfield, params);
    combinedSolver.initializeProblemInstance(canonicalFrameWarpedToLive, liveFrame);
    combinedSolver.solveAll();

    int j = 0;
    for (auto vertex : canonicalFrameWarpedToLive->getVertices()) {
        auto totalTransformation = warpfield.calcDQB(vertex);
        auto result              = vertex + totalTransformation->getTranslation();

        ASSERT_NEAR(result[0], liveFrame->getVertices()[j][0], maxError);
        ASSERT_NEAR(result[1], liveFrame->getVertices()[j][1], maxError);
        ASSERT_NEAR(result[2], liveFrame->getVertices()[j][2], maxError);

        j++;
    }
}
