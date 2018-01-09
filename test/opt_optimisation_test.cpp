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

        float RAD90 = M_PI / 2;
        float RAD30 = M_PI / 6;

        auto dg_se3 = std::make_shared<DualQuaternion<float>>(0, 0, 0, 0, 0, 0);  // transformation
        float dg_w  = 2;                                                          // radial basis weight

        /* init node group 1 */
        nodesGroup1.push_back(std::make_shared<Node>(pcl::PointXYZ(3, 1, -1), dg_se3, dg_w));
        nodesGroup1.push_back(std::make_shared<Node>(pcl::PointXYZ(1, 1, 1), dg_se3, dg_w));
        nodesGroup1.push_back(std::make_shared<Node>(pcl::PointXYZ(-1, 2, 3), dg_se3, dg_w));
        nodesGroup1.push_back(std::make_shared<Node>(pcl::PointXYZ(-1, -1, 1), dg_se3, dg_w));
        nodesGroup1.push_back(std::make_shared<Node>(pcl::PointXYZ(-2, -1, -1), dg_se3, dg_w));
        nodesGroup1.push_back(std::make_shared<Node>(pcl::PointXYZ(2, -1, -3), dg_se3, dg_w));
        nodesGroup1.push_back(std::make_shared<Node>(pcl::PointXYZ(-1, 1, -1), dg_se3, dg_w));
        nodesGroup1.push_back(std::make_shared<Node>(pcl::PointXYZ(2, 1, 1), dg_se3, dg_w));

        /* init node group 2 */
        nodesGroup2.push_back(std::make_shared<Node>(pcl::PointXYZ(10, 10, 10), dg_se3, dg_w));
        nodesGroup2.push_back(std::make_shared<Node>(pcl::PointXYZ(9, 11.1, 10), dg_se3, dg_w));
        nodesGroup2.push_back(std::make_shared<Node>(pcl::PointXYZ(10, 9, 10), dg_se3, dg_w));
        nodesGroup2.push_back(std::make_shared<Node>(pcl::PointXYZ(10, 12, 9), dg_se3, dg_w));
        nodesGroup2.push_back(std::make_shared<Node>(pcl::PointXYZ(9, 11, 10), dg_se3, dg_w));
        nodesGroup2.push_back(std::make_shared<Node>(pcl::PointXYZ(12, 10, 9), dg_se3, dg_w));
        nodesGroup2.push_back(std::make_shared<Node>(pcl::PointXYZ(9, 9, 12), dg_se3, dg_w));
        nodesGroup2.push_back(std::make_shared<Node>(pcl::PointXYZ(10.5, 9, 9), dg_se3, dg_w));
        nodesGroup2.push_back(std::make_shared<Node>(pcl::PointXYZ(10.5, 12, 12), dg_se3, dg_w));
        nodesGroup2.push_back(std::make_shared<Node>(pcl::PointXYZ(11, 11, 10.9), dg_se3, dg_w));

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
    }

    /* Objects declared here can be used by all tests in the test case for Solver. */
    CombinedSolverParameters params;

    float maxError      = 1e-3;
    const float epsilon = 1.192092896e-7f;

    std::shared_ptr<Node> node;

    std::vector<std::shared_ptr<Node>> nodesGroup1;
    std::vector<std::shared_ptr<Node>> nodesGroup2;
    std::vector<std::shared_ptr<Node>> allNodes;

    Warpfield warpfield;

    pcl::PointCloud<pcl::PointXYZ> sourceVertices;
    pcl::PointCloud<pcl::Normal> sourceNormals;

    pcl::PointCloud<pcl::PointXYZ> targetVertices;
    pcl::PointCloud<pcl::Normal> targetNormals;

    std::shared_ptr<dynfu::Frame> canonicalFrameWarpedToLive;
    std::shared_ptr<dynfu::Frame> liveFrame;
};

// /* FIXME (dig15); rotation of a 2-d vertex */
// TEST_F(OptTest, SimpleRotationTest) {
//     warpfield.init(nodesGroup1);
//
//     sourceVertices.push_back(pcl::PointXYZ(0, 0, 1));
//     sourceNormals.push_back(pcl::Normal(1, 1, 1));
//
//     canonicalFrameWarpedToLive = std::make_shared<dynfu::Frame>(0, sourceVertices, sourceNormals);
//
//     targetVertices.push_back(pcl::PointXYZ(1, 0, 0));
//     targetNormals.push_back(pcl::Normal(1, 1, 1));
//
//     liveFrame = std::make_shared<dynfu::Frame>(1, targetVertices, targetNormals);
//
//     CombinedSolver combinedSolver(warpfield, params);
//     combinedSolver.initializeProblemInstance(canonicalFrameWarpedToLive, liveFrame);
//     combinedSolver.solveAll();
//
//     int j = 0;
//     for (auto vertex : canonicalFrameWarpedToLive->getVertices()) {
//         auto totalTransformation = warpfield.calcDQB(vertex);
//         auto result              = totalTransformation->transformVertex(vertex);
//
//         std::cout << *totalTransformation << std::endl;
//         std::cout << result.x << " " << result.y << " " << result.z << std::endl;
//         ASSERT_NEAR(result.x, liveFrame->getVertices()[j].x, maxError);
//         ASSERT_NEAR(result.y, liveFrame->getVertices()[j].y, maxError);
//         ASSERT_NEAR(result.z, liveFrame->getVertices()[j].z, maxError);
//
//         j++;
//     }
// }
//
// /* FIXME (dig15); rotation of a 2-d triangle */
// TEST_F(OptTest, 2DTriangleRotationTest) {
//     warpfield.init(nodesGroup1);
//
//     sourceVertices.push_back(pcl::PointXYZ(0, 0, 1));
//     sourceVertices.push_back(pcl::PointXYZ(1, 0, 1));
//     sourceVertices.push_back(pcl::PointXYZ(1, 1, 1));
//
//     sourceNormals.push_back(pcl::Normal(1, 1, 1));
//     sourceNormals.push_back(pcl::Normal(1, 1, 1));
//     sourceNormals.push_back(pcl::Normal(1, 1, 1));
//
//     canonicalFrameWarpedToLive = std::make_shared<dynfu::Frame>(0, sourceVertices, sourceNormals);
//
//     targetVertices.push_back(pcl::PointXYZ(1, 1, 1));
//     targetVertices.push_back(pcl::PointXYZ(1, 0, 1));
//     targetVertices.push_back(pcl::PointXYZ(2, 0, 1));
//
//     targetNormals.push_back(pcl::Normal(1, 1, 1));
//     targetNormals.push_back(pcl::Normal(1, 1, 1));
//     targetNormals.push_back(pcl::Normal(1, 1, 1));
//
//     liveFrame = std::make_shared<dynfu::Frame>(1, targetVertices, targetNormals);
//
//     CombinedSolver combinedSolver(warpfield, params);
//     combinedSolver.initializeProblemInstance(canonicalFrameWarpedToLive, liveFrame);
//     combinedSolver.solveAll();
//
//     int j = 0;
//     for (auto vertex : canonicalFrameWarpedToLive->getVertices()) {
//         auto totalTransformation = warpfield.calcDQB(vertex);
//         auto result              = totalTransformation->transformVertex(vertex);
//
//         std::cout << *totalTransformation << std::endl;
//         std::cout << result.x << " " << result.y << " " << result.z << std::endl;
//         ASSERT_NEAR(result.x, liveFrame->getVertices()[j].x, maxError);
//         ASSERT_NEAR(result.y, liveFrame->getVertices()[j].y, maxError);
//         ASSERT_NEAR(result.z, liveFrame->getVertices()[j].z, maxError);
//
//         j++;
//     }
// }

/* */
TEST_F(OptTest, SingleVertexOneGroupOfDeformationNodesTest) {
    warpfield.init(nodesGroup1);

    sourceVertices.push_back(pcl::PointXYZ(0, 0.04, 0));
    sourceNormals.push_back(pcl::Normal(1, 1, 1));

    canonicalFrameWarpedToLive = std::make_shared<dynfu::Frame>(0, sourceVertices, sourceNormals);

    targetVertices.push_back(pcl::PointXYZ(0.01, 0.03, 0));
    targetNormals.push_back(pcl::Normal(1, 1, 1));

    liveFrame = std::make_shared<dynfu::Frame>(1, targetVertices, targetNormals);

    CombinedSolver combinedSolver(warpfield, params);
    combinedSolver.initializeProblemInstance(canonicalFrameWarpedToLive, liveFrame);
    combinedSolver.solveAll();

    int j = 0;
    for (auto vertex : canonicalFrameWarpedToLive->getVertices()) {
        auto totalTransformation = warpfield.calcDQB(vertex);
        auto result              = totalTransformation->transformVertex(vertex);

        ASSERT_NEAR(result.x, liveFrame->getVertices()[j].x, maxError);
        ASSERT_NEAR(result.y, liveFrame->getVertices()[j].y, maxError);
        ASSERT_NEAR(result.z, liveFrame->getVertices()[j].z, maxError);

        j++;
    }
}

/* */
TEST_F(OptTest, TwoVerticesOneNotMovingOneGroupOfDeformationNodesTest) {
    warpfield.init(allNodes);

    sourceVertices.push_back(pcl::PointXYZ(0, 0.05, 1));
    sourceVertices.push_back(pcl::PointXYZ(2, 2, 2));

    sourceNormals.push_back(pcl::Normal(1, 1, 1));
    sourceNormals.push_back(pcl::Normal(1, 1, 1));

    canonicalFrameWarpedToLive = std::make_shared<dynfu::Frame>(0, sourceVertices, sourceNormals);

    targetVertices.push_back(pcl::PointXYZ(0.01, 0.04, 1.01));
    targetVertices.push_back(pcl::PointXYZ(2, 2, 2));

    targetNormals.push_back(pcl::Normal(1, 1, 1));
    targetNormals.push_back(pcl::Normal(1, 1, 1));

    liveFrame = std::make_shared<dynfu::Frame>(1, targetVertices, targetNormals);

    CombinedSolver combinedSolver(warpfield, params);
    combinedSolver.initializeProblemInstance(canonicalFrameWarpedToLive, liveFrame);
    combinedSolver.solveAll();

    int j = 0;
    for (auto vertex : canonicalFrameWarpedToLive->getVertices()) {
        auto totalTransformation = warpfield.calcDQB(vertex);
        auto result              = totalTransformation->transformVertex(vertex);

        ASSERT_NEAR(result.x, liveFrame->getVertices()[j].x, maxError);
        ASSERT_NEAR(result.y, liveFrame->getVertices()[j].y, maxError);
        ASSERT_NEAR(result.z, liveFrame->getVertices()[j].z, maxError);

        j++;
    }
}

/* */
TEST_F(OptTest, MultipleVerticesOneGroupOfDeformationNodesTest) {
    warpfield.init(nodesGroup1);

    sourceVertices.push_back(pcl::PointXYZ(-3, -3, -3));
    sourceVertices.push_back(pcl::PointXYZ(-2, -2, -2));
    sourceVertices.push_back(pcl::PointXYZ(0.01, 0.01, 0.01));
    sourceVertices.push_back(pcl::PointXYZ(2, 2, 2));
    sourceVertices.push_back(pcl::PointXYZ(3, 3, 3));

    sourceNormals.push_back(pcl::Normal(1, 1, 1));
    sourceNormals.push_back(pcl::Normal(1, 1, 1));
    sourceNormals.push_back(pcl::Normal(1, 1, 1));
    sourceNormals.push_back(pcl::Normal(1, 1, 1));
    sourceNormals.push_back(pcl::Normal(1, 1, 1));

    canonicalFrameWarpedToLive = std::make_shared<dynfu::Frame>(0, sourceVertices, sourceNormals);

    targetVertices.push_back(pcl::PointXYZ(-2.99, -2.99, -2.99));
    targetVertices.push_back(pcl::PointXYZ(-1.99, -1.99, -1.99));
    targetVertices.push_back(pcl::PointXYZ(0.02, 0.02, 0.02));
    targetVertices.push_back(pcl::PointXYZ(2.01, 2.01, 2.01));
    targetVertices.push_back(pcl::PointXYZ(3.01, 3.01, 3.01));

    targetNormals.push_back(pcl::Normal(1, 1, 1));
    targetNormals.push_back(pcl::Normal(1, 1, 1));
    targetNormals.push_back(pcl::Normal(1, 1, 1));
    targetNormals.push_back(pcl::Normal(1, 1, 1));
    targetNormals.push_back(pcl::Normal(1, 1, 1));

    liveFrame = std::make_shared<dynfu::Frame>(1, targetVertices, targetNormals);

    CombinedSolver combinedSolver(warpfield, params);
    combinedSolver.initializeProblemInstance(canonicalFrameWarpedToLive, liveFrame);
    combinedSolver.solveAll();

    int j = 0;
    for (auto vertex : canonicalFrameWarpedToLive->getVertices()) {
        auto totalTransformation = warpfield.calcDQB(vertex);
        auto result              = totalTransformation->transformVertex(vertex);

        ASSERT_NEAR(result.x, liveFrame->getVertices()[j].x, maxError);
        ASSERT_NEAR(result.y, liveFrame->getVertices()[j].y, maxError);
        ASSERT_NEAR(result.z, liveFrame->getVertices()[j].z, maxError);

        j++;
    }
}

/* */
TEST_F(OptTest, OneGroupOfVerticesTwoGroupsOfDeformationNodes) {
    warpfield.init(allNodes);

    sourceVertices.push_back(pcl::PointXYZ(-3, -3, -3));
    sourceVertices.push_back(pcl::PointXYZ(-2, -2, -2));
    sourceVertices.push_back(pcl::PointXYZ(0.01, 0.01, 0.01));
    sourceVertices.push_back(pcl::PointXYZ(2, 2, 2));
    sourceVertices.push_back(pcl::PointXYZ(3, 3, 3));

    sourceNormals.push_back(pcl::Normal(1, 1, 1));
    sourceNormals.push_back(pcl::Normal(1, 1, 1));
    sourceNormals.push_back(pcl::Normal(1, 1, 1));
    sourceNormals.push_back(pcl::Normal(1, 1, 1));
    sourceNormals.push_back(pcl::Normal(1, 1, 1));

    canonicalFrameWarpedToLive = std::make_shared<dynfu::Frame>(0, sourceVertices, sourceNormals);

    targetVertices.push_back(pcl::PointXYZ(-2.99, -2.99, -2.99));
    targetVertices.push_back(pcl::PointXYZ(-1.99, -1.99, -1.99));
    targetVertices.push_back(pcl::PointXYZ(0.02, 0.02, 0.02));
    targetVertices.push_back(pcl::PointXYZ(2.01, 2.01, 2.01));
    targetVertices.push_back(pcl::PointXYZ(3.01, 3.01, 3.01));

    targetNormals.push_back(pcl::Normal(1, 1, 1));
    targetNormals.push_back(pcl::Normal(1, 1, 1));
    targetNormals.push_back(pcl::Normal(1, 1, 1));
    targetNormals.push_back(pcl::Normal(1, 1, 1));
    targetNormals.push_back(pcl::Normal(1, 1, 1));

    liveFrame = std::make_shared<dynfu::Frame>(1, targetVertices, targetNormals);

    CombinedSolver combinedSolver(warpfield, params);
    combinedSolver.initializeProblemInstance(canonicalFrameWarpedToLive, liveFrame);
    combinedSolver.solveAll();

    int j = 0;
    for (auto vertex : canonicalFrameWarpedToLive->getVertices()) {
        auto totalTransformation = warpfield.calcDQB(vertex);
        auto result              = totalTransformation->transformVertex(vertex);

        ASSERT_NEAR(result.x, liveFrame->getVertices()[j].x, maxError);
        ASSERT_NEAR(result.y, liveFrame->getVertices()[j].y, maxError);
        ASSERT_NEAR(result.z, liveFrame->getVertices()[j].z, maxError);

        j++;
    }
}

/* */
TEST_F(OptTest, TwoGroupsOfVerticesTwoGroupsOfDeformationNodes) {
    warpfield.init(allNodes);

    /* group 1 of source vertices */
    sourceVertices.push_back(pcl::PointXYZ(-3, -3, -3));
    sourceVertices.push_back(pcl::PointXYZ(-2, -2, -2));
    sourceVertices.push_back(pcl::PointXYZ(0.01, 0.01, 0.01));
    sourceVertices.push_back(pcl::PointXYZ(2, 2, 2));
    sourceVertices.push_back(pcl::PointXYZ(3, 3, 3));

    sourceNormals.push_back(pcl::Normal(1, 1, 1));
    sourceNormals.push_back(pcl::Normal(1, 1, 1));
    sourceNormals.push_back(pcl::Normal(1, 1, 1));
    sourceNormals.push_back(pcl::Normal(1, 1, 1));
    sourceNormals.push_back(pcl::Normal(1, 1, 1));

    /* group 2 of source vertices */
    sourceVertices.push_back(pcl::PointXYZ(12, 12, 12));
    sourceVertices.push_back(pcl::PointXYZ(11, 11, 11));
    sourceVertices.push_back(pcl::PointXYZ(10, 10, 10));
    sourceVertices.push_back(pcl::PointXYZ(10.5, 10.5, 10.5));
    sourceVertices.push_back(pcl::PointXYZ(11.5, 11.5, 11.5));

    sourceNormals.push_back(pcl::Normal(1, 1, 1));
    sourceNormals.push_back(pcl::Normal(1, 1, 1));
    sourceNormals.push_back(pcl::Normal(1, 1, 1));
    sourceNormals.push_back(pcl::Normal(1, 1, 1));
    sourceNormals.push_back(pcl::Normal(1, 1, 1));

    canonicalFrameWarpedToLive = std::make_shared<dynfu::Frame>(0, sourceVertices, sourceNormals);

    /* group 1 of target vertices */
    targetVertices.push_back(pcl::PointXYZ(-2.99, -2.99, -2.99));
    targetVertices.push_back(pcl::PointXYZ(-1.99, -1.99, -1.99));
    targetVertices.push_back(pcl::PointXYZ(0.02, 0.02, 0.02));
    targetVertices.push_back(pcl::PointXYZ(2.01, 2.01, 2.01));
    targetVertices.push_back(pcl::PointXYZ(3.01, 3.01, 3.01));

    targetNormals.push_back(pcl::Normal(1, 1, 1));
    targetNormals.push_back(pcl::Normal(1, 1, 1));
    targetNormals.push_back(pcl::Normal(1, 1, 1));
    targetNormals.push_back(pcl::Normal(1, 1, 1));
    targetNormals.push_back(pcl::Normal(1, 1, 1));

    /* group 2 of target vertices */
    targetVertices.push_back(pcl::PointXYZ(11.99, 11.99, 11.99));
    targetVertices.push_back(pcl::PointXYZ(10.99, 10.99, 10.99));
    targetVertices.push_back(pcl::PointXYZ(9.99, 9.99, 9.99));
    targetVertices.push_back(pcl::PointXYZ(10.51, 10.51, 10.51));
    targetVertices.push_back(pcl::PointXYZ(11.49, 11.49, 11.49));

    targetNormals.push_back(pcl::Normal(1, 1, 1));
    targetNormals.push_back(pcl::Normal(1, 1, 1));
    targetNormals.push_back(pcl::Normal(1, 1, 1));
    targetNormals.push_back(pcl::Normal(1, 1, 1));
    targetNormals.push_back(pcl::Normal(1, 1, 1));

    liveFrame = std::make_shared<dynfu::Frame>(1, targetVertices, targetNormals);

    CombinedSolver combinedSolver(warpfield, params);
    combinedSolver.initializeProblemInstance(canonicalFrameWarpedToLive, liveFrame);
    combinedSolver.solveAll();

    int j = 0;
    for (auto vertex : canonicalFrameWarpedToLive->getVertices()) {
        auto totalTransformation = warpfield.calcDQB(vertex);
        auto result              = totalTransformation->transformVertex(vertex);

        ASSERT_NEAR(result.x, liveFrame->getVertices()[j].x, maxError);
        ASSERT_NEAR(result.y, liveFrame->getVertices()[j].y, maxError);
        ASSERT_NEAR(result.z, liveFrame->getVertices()[j].z, maxError);

        j++;
    }
}

TEST_F(OptTest, MultipleVerticesOneGroupOfDeformationNodesWarpAndReverseTest) {
    warpfield.init(nodesGroup1);

    sourceVertices.push_back(pcl::PointXYZ(-3, -3, -3));
    sourceVertices.push_back(pcl::PointXYZ(-2, -2, -2));
    sourceVertices.push_back(pcl::PointXYZ(0.04, 0.04, 0.04));
    sourceVertices.push_back(pcl::PointXYZ(2, 2, 2));
    sourceVertices.push_back(pcl::PointXYZ(3, 3, 3));

    sourceNormals.push_back(pcl::Normal(1, 1, 1));
    sourceNormals.push_back(pcl::Normal(1, 1, 1));
    sourceNormals.push_back(pcl::Normal(1, 1, 1));
    sourceNormals.push_back(pcl::Normal(1, 1, 1));
    sourceNormals.push_back(pcl::Normal(1, 1, 1));

    canonicalFrameWarpedToLive = std::make_shared<dynfu::Frame>(0, sourceVertices, sourceNormals);

    targetVertices.push_back(pcl::PointXYZ(-2.99, -2.99, -2.99));
    targetVertices.push_back(pcl::PointXYZ(-1.99, -1.99, -1.99));
    targetVertices.push_back(pcl::PointXYZ(0.05, 0.05, 0.05));
    targetVertices.push_back(pcl::PointXYZ(2.01, 2.01, 2.01));
    targetVertices.push_back(pcl::PointXYZ(3.01, 3.01, 3.01));

    targetNormals.push_back(pcl::Normal(1, 1, 1));
    targetNormals.push_back(pcl::Normal(1, 1, 1));
    targetNormals.push_back(pcl::Normal(1, 1, 1));
    targetNormals.push_back(pcl::Normal(1, 1, 1));
    targetNormals.push_back(pcl::Normal(1, 1, 1));

    liveFrame = std::make_shared<dynfu::Frame>(1, targetVertices, targetNormals);

    CombinedSolver combinedSolver(warpfield, params);
    combinedSolver.initializeProblemInstance(canonicalFrameWarpedToLive, liveFrame);
    combinedSolver.solveAll();

    int j = 0;
    for (auto vertex : canonicalFrameWarpedToLive->getVertices()) {
        auto totalTransformation = warpfield.calcDQB(vertex);
        auto result              = totalTransformation->transformVertex(vertex);

        ASSERT_NEAR(result.x, liveFrame->getVertices()[j].x, maxError);
        ASSERT_NEAR(result.y, liveFrame->getVertices()[j].y, maxError);
        ASSERT_NEAR(result.z, liveFrame->getVertices()[j].z, maxError);

        j++;
    }

    auto temp                  = canonicalFrameWarpedToLive;
    canonicalFrameWarpedToLive = liveFrame;
    liveFrame                  = temp;

    CombinedSolver combinedSolverReverse(warpfield, params);
    combinedSolverReverse.initializeProblemInstance(canonicalFrameWarpedToLive, liveFrame);
    combinedSolverReverse.solveAll();

    j = 0;
    for (auto vertex : canonicalFrameWarpedToLive->getVertices()) {
        auto totalTransformation = warpfield.calcDQB(vertex);
        auto result              = totalTransformation->transformVertex(vertex);

        ASSERT_NEAR(result.x, liveFrame->getVertices()[j].x, maxError);
        ASSERT_NEAR(result.y, liveFrame->getVertices()[j].y, maxError);
        ASSERT_NEAR(result.z, liveFrame->getVertices()[j].z, maxError);

        j++;
    }
}
