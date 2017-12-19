/* gtest includes */
#include <gtest/gtest.h>

/* dynfu includes */
#include <dynfu/utils/ceres_solver.hpp>
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

        params.numIter       = 32;
        params.nonLinearIter = 16;
        params.linearIter    = 256;
        params.useOpt        = false;
        params.useOptLM      = true;
        params.earlyOut      = true;

        dg_v      = {1.f, 1.f, -1.f};
        auto node = std::make_shared<Node>(dg_v, dg_se3, dg_w);
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

        dg_v = {1.f, 1.f, 1.f};
        node = std::make_shared<Node>(dg_v, dg_se3, dg_w);
        nodes.push_back(node);
    }

    /* Code here will be called immediately after each test (right
     * before the destructor). */
    void TearDown() override {
        nodes.clear();

        sourceVertices.clear();
        sourceNormals.clear();

        targetVertices.clear();
        targetNormals.clear();
    }

    /* Objects declared here can be used by all tests in the test case for Solver. */
    ceres::Solver::Options options;
    CombinedSolverParameters params;

    float max_error = 1e-3;

    std::shared_ptr<Node> node;
    std::vector<std::shared_ptr<Node>> nodes;

    Warpfield warpfield;

    cv::Vec3f dg_v;
    std::shared_ptr<DualQuaternion<float>> dg_se3 =
        std::make_shared<DualQuaternion<float>>(0.f, 0.f, 0.f, 0.f, 0.f, 0.f);
    float dg_w = 2.f;

    std::vector<cv::Vec3f> sourceVertices;
    std::vector<cv::Vec3f> sourceNormals;

    std::vector<cv::Vec3f> targetVertices;
    std::vector<cv::Vec3f> targetNormals;

    std::shared_ptr<dynfu::Frame> canonicalFrameWarpedToLive;
    std::shared_ptr<dynfu::Frame> liveFrame;
};

/* */
TEST_F(SolverTest, SingleVertexTest) {
    warpfield.init(nodes);

    sourceVertices.emplace_back(cv::Vec3f(1.05, 0.05, 1));
    canonicalFrameWarpedToLive = std::make_shared<dynfu::Frame>(0, sourceVertices, sourceVertices);

    targetVertices.emplace_back(cv::Vec3f(1.0, 0.0, 1.0));
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
    max_error = 0.01;
    ASSERT_NEAR(result[0], liveFrame->getVertices()[0][0], max_error);
    ASSERT_NEAR(result[1], liveFrame->getVertices()[0][1], max_error);
    ASSERT_NEAR(result[2], liveFrame->getVertices()[0][2], max_error);
}

/* */
TEST_F(SolverTest, MultipleVerticesTest) {
    warpfield.init(nodes);

    sourceVertices.emplace_back(cv::Vec3f(-3, -3, -3));
    sourceVertices.emplace_back(cv::Vec3f(-2, -2, -2));
    sourceVertices.emplace_back(cv::Vec3f(0, 0, 0));
    sourceVertices.emplace_back(cv::Vec3f(2, 2, 2));
    sourceVertices.emplace_back(cv::Vec3f(3, 3, 3));

    canonicalFrameWarpedToLive = std::make_shared<dynfu::Frame>(0, sourceVertices, sourceVertices);

    targetVertices.emplace_back(cv::Vec3f(-2.95f, -2.95f, -2.95f));
    targetVertices.emplace_back(cv::Vec3f(-1.95f, -1.95f, -1.95f));
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

        auto totalTransformation = warpfield->calcDQB(vertex);
        auto result              = vertex + totalTransformation->getTranslation();

        ASSERT_NEAR(result[0], liveFrame->getVertices()[j][0], max_error);
        ASSERT_NEAR(result[1], liveFrame->getVertices()[j][1], max_error);
        ASSERT_NEAR(result[2], liveFrame->getVertices()[j][2], max_error);

        i = 0;
        j++;
    }
}

/* */
TEST_F(SolverTest, SingleVertexOneGroupOfDeformationNodesTestOpt) {
    warpfield.init(nodes);

    sourceVertices.emplace_back(cv::Vec3f(1.05, 0.05, 1));
    sourceNormals.emplace_back(cv::Vec3f(1, 1, 1));
    canonicalFrameWarpedToLive = std::make_shared<dynfu::Frame>(
        0, sourceVertices, sourceVertices);  // FIXME (dig15): understand how to use the normals

    targetVertices.emplace_back(cv::Vec3f(1.0, 0.0, 1.0));
    targetNormals.emplace_back(cv::Vec3f(1, 1, 1));
    liveFrame = std::make_shared<dynfu::Frame>(1, targetVertices, targetVertices);

    CombinedSolver combinedSolver(warpfield, params);
    combinedSolver.initializeProblemInstance(canonicalFrameWarpedToLive, liveFrame);
    combinedSolver.solveAll();

    /* FIXME (dig15): figure out why the test fails */
    max_error = 0.01;

    int j = 0;
    for (auto vertex : canonicalFrameWarpedToLive->getVertices()) {
        auto totalTransformation = warpfield.calcDQB(vertex);
        auto result              = vertex + totalTransformation->getTranslation();

        ASSERT_NEAR(result[0], liveFrame->getVertices()[j][0], max_error);
        ASSERT_NEAR(result[1], liveFrame->getVertices()[j][1], max_error);
        ASSERT_NEAR(result[2], liveFrame->getVertices()[j][2], max_error);

        j++;
    }
}

/* */
TEST_F(SolverTest, MultipleVerticesOneGroupOfDeformationNodesTestOpt) {
    warpfield.init(nodes);

    sourceVertices.emplace_back(cv::Vec3f(-3, -3, -3));
    sourceVertices.emplace_back(cv::Vec3f(-2, -2, -2));
    sourceVertices.emplace_back(cv::Vec3f(0, 0, 0));
    sourceVertices.emplace_back(cv::Vec3f(2, 2, 2));
    sourceVertices.emplace_back(cv::Vec3f(3, 3, 3));

    canonicalFrameWarpedToLive = std::make_shared<dynfu::Frame>(0, sourceVertices, sourceVertices);

    targetVertices.emplace_back(cv::Vec3f(-2.95f, -2.95f, -2.95f));
    targetVertices.emplace_back(cv::Vec3f(-1.95f, -1.95f, -1.95f));
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

        ASSERT_NEAR(result[0], liveFrame->getVertices()[j][0], max_error);
        ASSERT_NEAR(result[1], liveFrame->getVertices()[j][1], max_error);
        ASSERT_NEAR(result[2], liveFrame->getVertices()[j][2], max_error);

        j++;
    }
}

/* */
TEST_F(SolverTest, OneGroupOfVerticesTwoGroupsOfDeformationNodesOpt) {
    dg_v = {10.f, 10.f, 10.f};
    node = std::make_shared<Node>(dg_v, dg_se3, dg_w);
    nodes.push_back(node);

    dg_v = {9.f, 10.f, 10.f};
    node = std::make_shared<Node>(dg_v, dg_se3, dg_w);
    nodes.push_back(node);

    dg_v = {10.f, 9.f, 10.f};
    node = std::make_shared<Node>(dg_v, dg_se3, dg_w);
    nodes.push_back(node);

    dg_v = {10.f, 10.f, 9.f};
    node = std::make_shared<Node>(dg_v, dg_se3, dg_w);
    nodes.push_back(node);

    dg_v = {9.f, 9.f, 10.f};
    node = std::make_shared<Node>(dg_v, dg_se3, dg_w);
    nodes.push_back(node);

    dg_v = {9.f, 10.f, 9.f};
    node = std::make_shared<Node>(dg_v, dg_se3, dg_w);
    nodes.push_back(node);

    dg_v = {9.f, 9.f, 9.f};
    node = std::make_shared<Node>(dg_v, dg_se3, dg_w);
    nodes.push_back(node);

    dg_v = {9.f, 9.f, 9.f};
    node = std::make_shared<Node>(dg_v, dg_se3, dg_w);
    nodes.push_back(node);

    warpfield = std::make_shared<Warpfield>();
    warpfield->init(nodes);

    sourceVertices.emplace_back(cv::Vec3f(-3, -3, -3));
    sourceVertices.emplace_back(cv::Vec3f(-2, -2, -2));
    sourceVertices.emplace_back(cv::Vec3f(0, 0, 0));
    sourceVertices.emplace_back(cv::Vec3f(2, 2, 2));
    sourceVertices.emplace_back(cv::Vec3f(3, 3, 3));

    canonicalFrameWarpedToLive = std::make_shared<dynfu::Frame>(0, sourceVertices, sourceVertices);

    targetVertices.emplace_back(cv::Vec3f(-2.95f, -2.95f, -2.95f));
    targetVertices.emplace_back(cv::Vec3f(-1.95f, -1.95f, -1.95f));
    targetVertices.emplace_back(cv::Vec3f(0.05, 0.05, 0.05));
    targetVertices.emplace_back(cv::Vec3f(2.05, 2.05, 2.05));
    targetVertices.emplace_back(cv::Vec3f(3.05, 3.05, 3.05));

    liveFrame = std::make_shared<dynfu::Frame>(1, targetVertices, targetVertices);

    CombinedSolver combinedSolver(warpfield, params);
    combinedSolver.initializeProblemInstance(canonicalFrameWarpedToLive, liveFrame);
    combinedSolver.solveAll();

    int j = 0;
    for (auto vertex : canonicalFrameWarpedToLive->getVertices()) {
<<<<<<< HEAD
        cv::Vec3f totalTranslation;

        auto neighbourNodes = warpfield->findNeighbors(KNN, vertex);

        for (auto neighbour : neighbourNodes) {
            cv::Vec3f translation = neighbour->getTransformation()->getTranslation();
            totalTranslation += translation;
        }
=======
        auto totalTransformation = warpfield.calcDQB(vertex);
        auto result              = vertex + totalTransformation->getTranslation();
>>>>>>> 8142940... test/solver_test.cpp: Add Use of DQB in tests

        ASSERT_NEAR(result[0], liveFrame->getVertices()[j][0], max_error);
        ASSERT_NEAR(result[1], liveFrame->getVertices()[j][1], max_error);
        ASSERT_NEAR(result[2], liveFrame->getVertices()[j][2], max_error);

        j++;
    }
<<<<<<< HEAD

    // // FIXME (dig15): can't use yet because Opt doesn't solve for weights
    // int j = 0;
    // for (auto vertex : canonicalFrameWarpedToLive->getVertices()) {
    //     cv::Vec3f totalTranslation;
    //
    //     auto totalTransformation = warpfield->calcDQB(vertex);
    //     auto result              = vertex + totalTransformation->getTranslation();
    //
    //     ASSERT_NEAR(result[0], liveFrame->getVertices()[j][0], max_error);
    //     ASSERT_NEAR(result[1], liveFrame->getVertices()[j][1], max_error);
    //     ASSERT_NEAR(result[2], liveFrame->getVertices()[j][2], max_error);
    //
    //     j++;
    // }
=======
>>>>>>> 8142940... test/solver_test.cpp: Add Use of DQB in tests
}

/* */
TEST_F(SolverTest, TwoGroupsOfVerticesTwoGroupsOfDeformationNodesOpt) {
    dg_v = {10.f, 10.f, 10.f};
    node = std::make_shared<Node>(dg_v, dg_se3, dg_w);
    nodes.push_back(node);

    dg_v = {9.f, 10.f, 10.f};
    node = std::make_shared<Node>(dg_v, dg_se3, dg_w);
    nodes.push_back(node);

    dg_v = {10.f, 9.f, 10.f};
    node = std::make_shared<Node>(dg_v, dg_se3, dg_w);
    nodes.push_back(node);

    dg_v = {10.f, 10.f, 9.f};
    node = std::make_shared<Node>(dg_v, dg_se3, dg_w);
    nodes.push_back(node);

    dg_v = {9.f, 9.f, 10.f};
    node = std::make_shared<Node>(dg_v, dg_se3, dg_w);
    nodes.push_back(node);

    dg_v = {9.f, 10.f, 9.f};
    node = std::make_shared<Node>(dg_v, dg_se3, dg_w);
    nodes.push_back(node);

    dg_v = {9.f, 9.f, 9.f};
    node = std::make_shared<Node>(dg_v, dg_se3, dg_w);
    nodes.push_back(node);

    dg_v = {9.f, 9.f, 9.f};
    node = std::make_shared<Node>(dg_v, dg_se3, dg_w);
    nodes.push_back(node);

    warpfield = std::make_shared<Warpfield>();
    warpfield->init(nodes);

    /* group 1 of source vertices */
    sourceVertices.emplace_back(cv::Vec3f(-3, -3, -3));
    sourceVertices.emplace_back(cv::Vec3f(-2, -2, -2));
    sourceVertices.emplace_back(cv::Vec3f(0, 0, 0));
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
    targetVertices.emplace_back(cv::Vec3f(-2.95f, -2.95f, -2.95f));
    targetVertices.emplace_back(cv::Vec3f(-1.95f, -1.95f, -1.95f));
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

    int j = 0;
    for (auto vertex : canonicalFrameWarpedToLive->getVertices()) {
        auto totalTransformation = warpfield.calcDQB(vertex);
        auto result              = vertex + totalTransformation->getTranslation();

        ASSERT_NEAR(result[0], liveFrame->getVertices()[j][0], max_error);
        ASSERT_NEAR(result[1], liveFrame->getVertices()[j][1], max_error);
        ASSERT_NEAR(result[2], liveFrame->getVertices()[j][2], max_error);

        j++;
    }
}

/* FIXME (dig15): figure out why the test fails */
TEST_F(SolverTest, MultipleVerticesOneGroupOfDeformationNodesWarpAndReverseTestOpt) {
    warpfield.init(nodes);

    sourceVertices.emplace_back(cv::Vec3f(-3, -3, -3));
    sourceVertices.emplace_back(cv::Vec3f(-2, -2, -2));
    sourceVertices.emplace_back(cv::Vec3f(0, 0, 0));
    sourceVertices.emplace_back(cv::Vec3f(2, 2, 2));
    sourceVertices.emplace_back(cv::Vec3f(3, 3, 3));

    canonicalFrameWarpedToLive = std::make_shared<dynfu::Frame>(0, sourceVertices, sourceVertices);

    targetVertices.emplace_back(cv::Vec3f(-2.95f, -2.95f, -2.95f));
    targetVertices.emplace_back(cv::Vec3f(-1.95f, -1.95f, -1.95f));
    targetVertices.emplace_back(cv::Vec3f(0.05, 0.05, 0.05));
    targetVertices.emplace_back(cv::Vec3f(2.05, 2.05, 2.05));
    targetVertices.emplace_back(cv::Vec3f(3.05, 3.05, 3.05));

    liveFrame = std::make_shared<dynfu::Frame>(1, targetVertices, targetVertices);

    CombinedSolver combinedSolver(warpfield, params);
    combinedSolver.initializeProblemInstance(canonicalFrameWarpedToLive, liveFrame);
    combinedSolver.solveAll();

    int j = 0;
    for (auto vertex : canonicalFrameWarpedToLive->getVertices()) {
<<<<<<< HEAD
        cv::Vec3f totalTranslation;

        auto neighbourNodes = warpfield->findNeighbors(KNN, vertex);

        for (auto neighbour : neighbourNodes) {
            cv::Vec3f translation = neighbour->getTransformation()->getTranslation();
            totalTranslation += translation;
        }
=======
        auto totalTransformation = warpfield.calcDQB(vertex);
        auto result              = vertex + totalTransformation->getTranslation();
>>>>>>> 8142940... test/solver_test.cpp: Add Use of DQB in tests

        ASSERT_NEAR(result[0], liveFrame->getVertices()[j][0], max_error);
        ASSERT_NEAR(result[1], liveFrame->getVertices()[j][1], max_error);
        ASSERT_NEAR(result[2], liveFrame->getVertices()[j][2], max_error);

        j++;
    }

<<<<<<< HEAD
    // // FIXME (dig15): can't use yet because Opt doesn't solve for weights
    // int j = 0;
    // for (auto vertex : canonicalFrameWarpedToLive->getVertices()) {
    //     cv::Vec3f totalTranslation;
    //
    //     auto totalTransformation = warpfield->calcDQB(vertex);
    //     auto result              = vertex + totalTransformation->getTranslation();
    //
    //     ASSERT_NEAR(result[0], liveFrame->getVertices()[j][0], max_error);
    //     ASSERT_NEAR(result[1], liveFrame->getVertices()[j][1], max_error);
    //     ASSERT_NEAR(result[2], liveFrame->getVertices()[j][2], max_error);
    //
    //     j++;
    // }

=======
>>>>>>> 8142940... test/solver_test.cpp: Add Use of DQB in tests
    /* reverse */
    auto temp                  = liveFrame;
    liveFrame                  = canonicalFrameWarpedToLive;
    canonicalFrameWarpedToLive = temp;

    CombinedSolver combinedSolverReverse(warpfield, params);
    combinedSolverReverse.initializeProblemInstance(canonicalFrameWarpedToLive, liveFrame);
    combinedSolverReverse.solveAll();

    j = 0;
    for (auto vertex : canonicalFrameWarpedToLive->getVertices()) {
<<<<<<< HEAD
        cv::Vec3f totalTranslation;

        auto neighbourNodes = warpfield->findNeighbors(KNN, vertex);

        for (auto neighbour : neighbourNodes) {
            cv::Vec3f translation = neighbour->getTransformation()->getTranslation();
            totalTranslation += translation;
        }
=======
        auto totalTransformation = warpfield.calcDQB(vertex);
        auto result              = vertex + totalTransformation->getTranslation();
>>>>>>> 8142940... test/solver_test.cpp: Add Use of DQB in tests

        ASSERT_NEAR(result[0], liveFrame->getVertices()[j][0], max_error);
        ASSERT_NEAR(result[1], liveFrame->getVertices()[j][1], max_error);
        ASSERT_NEAR(result[2], liveFrame->getVertices()[j][2], max_error);

        j++;
    }
<<<<<<< HEAD

    // // FIXME (dig15): can't use yet because Opt doesn't solve for weights
    // j = 0;
    // for (auto vertex : canonicalFrameWarpedToLive->getVertices()) {
    //     cv::Vec3f totalTranslation;
    //
    //     auto totalTransformation = warpfield->calcDQB(vertex);
    //     auto result              = vertex + totalTransformation->getTranslation();
    //
    //     ASSERT_NEAR(result[0], liveFrame->getVertices()[j][0], max_error);
    //     ASSERT_NEAR(result[1], liveFrame->getVertices()[j][1], max_error);
    //     ASSERT_NEAR(result[2], liveFrame->getVertices()[j][2], max_error);
    //
    //     j++;
    // }
=======
>>>>>>> 8142940... test/solver_test.cpp: Add Use of DQB in tests
}

/* */
TEST_F(SolverTest, MultipleVerticesOneGroupOfDeformationNodesNonRigidTest) {
    warpfield.init(nodes);

    sourceVertices.emplace_back(cv::Vec3f(-3, -3, -3));
    sourceVertices.emplace_back(cv::Vec3f(-2, -2, -2));
    sourceVertices.emplace_back(cv::Vec3f(0, 0, 0));
    sourceVertices.emplace_back(cv::Vec3f(2, 2, 2));
    sourceVertices.emplace_back(cv::Vec3f(3, 3, 3));

    canonicalFrameWarpedToLive = std::make_shared<dynfu::Frame>(0, sourceVertices, sourceVertices);

    targetVertices.emplace_back(cv::Vec3f(-2.95f, -2.95f, -2.95f));
    targetVertices.emplace_back(cv::Vec3f(-2.05f, -2.05f, -2.05f));
    targetVertices.emplace_back(cv::Vec3f(0, 0, 0));
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

        ASSERT_NEAR(result[0], liveFrame->getVertices()[j][0], max_error);
        ASSERT_NEAR(result[1], liveFrame->getVertices()[j][1], max_error);
        ASSERT_NEAR(result[2], liveFrame->getVertices()[j][2], max_error);

        j++;
    }
}

TEST_F(SolverTest, MihaiNonRigidTest) {
    Warpfield warpfieldNonRigid;
    std::vector<std::shared_ptr<Node>> nodesNonRigid;

    cv::Vec3f dg_v2   = {1, 1, 1};
    auto nodeNonRigid = std::make_shared<Node>(dg_v2, dg_se3, dg_w);
    nodesNonRigid.push_back(nodeNonRigid);

    dg_v2        = {1, 2, -1};
    nodeNonRigid = std::make_shared<Node>(dg_v2, dg_se3, dg_w);
    nodesNonRigid.push_back(nodeNonRigid);

    dg_v2        = {1, -2, 1};
    nodeNonRigid = std::make_shared<Node>(dg_v2, dg_se3, dg_w);
    nodesNonRigid.push_back(nodeNonRigid);

    dg_v2        = {1, -1, -1};
    nodeNonRigid = std::make_shared<Node>(dg_v2, dg_se3, dg_w);
    nodesNonRigid.push_back(nodeNonRigid);

    dg_v2        = {-1, 1, 5};
    nodeNonRigid = std::make_shared<Node>(dg_v2, dg_se3, dg_w);
    nodesNonRigid.push_back(nodeNonRigid);

    dg_v2        = {-1, 1, -1};
    nodeNonRigid = std::make_shared<Node>(dg_v2, dg_se3, dg_w);
    nodesNonRigid.push_back(nodeNonRigid);

    dg_v2        = {-1, 1, 5};
    nodeNonRigid = std::make_shared<Node>(dg_v2, dg_se3, dg_w);
    nodesNonRigid.push_back(nodeNonRigid);

    dg_v2        = {-1, -1, 1};
    nodeNonRigid = std::make_shared<Node>(dg_v2, dg_se3, dg_w);
    nodesNonRigid.push_back(nodeNonRigid);

    dg_v2        = {-1, 1, 5};
    nodeNonRigid = std::make_shared<Node>(dg_v2, dg_se3, dg_w);
    nodesNonRigid.push_back(nodeNonRigid);

    dg_v2        = {-1, -1, 1};
    nodeNonRigid = std::make_shared<Node>(dg_v2, dg_se3, dg_w);
    nodesNonRigid.push_back(nodeNonRigid);

    dg_v2        = {-1, -1, -1};
    nodeNonRigid = std::make_shared<Node>(dg_v2, dg_se3, dg_w);
    nodesNonRigid.push_back(nodeNonRigid);

    dg_v2        = {2, -3, -1};
    nodeNonRigid = std::make_shared<Node>(dg_v2, dg_se3, dg_w);
    nodesNonRigid.push_back(nodeNonRigid);

    warpfieldNonRigid.init(nodesNonRigid);

    std::vector<cv::Vec3f> source_vertices;
    source_vertices.emplace_back(cv::Vec3f(-3, -3, -3));
    source_vertices.emplace_back(cv::Vec3f(-2, -2, -2));
    source_vertices.emplace_back(cv::Vec3f(0, 0, 0));
    source_vertices.emplace_back(cv::Vec3f(2, 2, 2));
    source_vertices.emplace_back(cv::Vec3f(3, 3, 3));
    source_vertices.emplace_back(cv::Vec3f(3, 3, 3));

    canonicalFrameWarpedToLive = std::make_shared<dynfu::Frame>(0, source_vertices, source_vertices);

    std::vector<cv::Vec3f> target_vertices;
    target_vertices.emplace_back(cv::Vec3f(-2.95f, -3.f, -2.95f));
    target_vertices.emplace_back(cv::Vec3f(-1.95f, -1.95f, -2.f));
    target_vertices.emplace_back(cv::Vec3f(0.1, 0.1, 0.1));
    target_vertices.emplace_back(cv::Vec3f(2, 2.5f, 2));
    target_vertices.emplace_back(cv::Vec3f(3.05, 3.05, 3.05));
    target_vertices.emplace_back(cv::Vec3f(3.05, 3.05, 3.05));

    liveFrame = std::make_shared<dynfu::Frame>(0, target_vertices, target_vertices);

    CombinedSolver combinedSolver(warpfieldNonRigid, params);
    combinedSolver.initializeProblemInstance(canonicalFrameWarpedToLive, liveFrame);
    combinedSolver.solveAll();

    int j = 0;
    for (auto vertex : canonicalFrameWarpedToLive->getVertices()) {
        auto totalTransformation = warpfieldNonRigid.calcDQB(vertex);
        auto result              = vertex + totalTransformation->getTranslation();

        ASSERT_NEAR(result[0], liveFrame->getVertices()[j][0], max_error);
        ASSERT_NEAR(result[1], liveFrame->getVertices()[j][1], max_error);
        ASSERT_NEAR(result[2], liveFrame->getVertices()[j][2], max_error);

        j++;
    }
}
