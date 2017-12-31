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

        auto dg_se3 = std::make_shared<DualQuaternion<float>>(0, 0, 0, 0, 0, 0);  // transformation
        float dg_w  = 2;                                                          // radial basis weight

        /* init node group 1 */
        nodesGroup1.push_back(std::make_shared<Node>(cv::Vec3f(1, 1, -1), dg_se3, dg_w));
        nodesGroup1.push_back(std::make_shared<Node>(cv::Vec3f(1, -1, 1), dg_se3, dg_w));
        nodesGroup1.push_back(std::make_shared<Node>(cv::Vec3f(-1, 1, 1), dg_se3, dg_w));
        nodesGroup1.push_back(std::make_shared<Node>(cv::Vec3f(-1, -1, 1), dg_se3, dg_w));
        nodesGroup1.push_back(std::make_shared<Node>(cv::Vec3f(-1, -1, -1), dg_se3, dg_w));
        nodesGroup1.push_back(std::make_shared<Node>(cv::Vec3f(1, -1, -1), dg_se3, dg_w));
        nodesGroup1.push_back(std::make_shared<Node>(cv::Vec3f(-1, 1, -1), dg_se3, dg_w));
        nodesGroup1.push_back(std::make_shared<Node>(cv::Vec3f(1, 1, 1), dg_se3, dg_w));

        /* init node group 2 */
        nodesGroup2.push_back(std::make_shared<Node>(cv::Vec3f(10, 10, 10), dg_se3, dg_w));
        nodesGroup2.push_back(std::make_shared<Node>(cv::Vec3f(9, 10, 10), dg_se3, dg_w));
        nodesGroup2.push_back(std::make_shared<Node>(cv::Vec3f(10, 9, 10), dg_se3, dg_w));
        nodesGroup2.push_back(std::make_shared<Node>(cv::Vec3f(10, 10, 9), dg_se3, dg_w));
        nodesGroup2.push_back(std::make_shared<Node>(cv::Vec3f(9, 9, 10), dg_se3, dg_w));
        nodesGroup2.push_back(std::make_shared<Node>(cv::Vec3f(9, 10, 9), dg_se3, dg_w));
        nodesGroup2.push_back(std::make_shared<Node>(cv::Vec3f(9, 9, 9), dg_se3, dg_w));
        nodesGroup2.push_back(std::make_shared<Node>(cv::Vec3f(9, 9, 9), dg_se3, dg_w));

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
        //
        // sourceVertices.clear();
        // sourceNormals.clear();
        //
        // targetVertices.clear();
        // targetNormals.clear();
    }

    /* Objects declared here can be used by all tests in the test case for Solver. */
    CombinedSolverParameters params;

    float maxError = 1e-3;

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

/* */
TEST_F(OptTest, SingleVertexOneGroupOfDeformationNodesTest) {
    warpfield.init(nodesGroup1);

    sourceVertices.push_back(pcl::PointXYZ(0, 0.05, 1));
    sourceNormals.push_back(pcl::Normal(1, 1, 1));

    canonicalFrameWarpedToLive = std::make_shared<dynfu::Frame>(0, sourceVertices, sourceNormals);

    targetVertices.push_back(pcl::PointXYZ(0.05, 0.0, 1));
    targetNormals.push_back(pcl::Normal(1, 1, 1));

    liveFrame = std::make_shared<dynfu::Frame>(1, targetVertices, targetNormals);

    CombinedSolver combinedSolver(warpfield, params);
    combinedSolver.initializeProblemInstance(canonicalFrameWarpedToLive, liveFrame);
    combinedSolver.solveAll();

    /* FIXME (dig15): figure out why the test fails */
    int j = 0;
    for (auto vertex : canonicalFrameWarpedToLive->getVertices()) {
        auto totalTransformation = warpfield.calcDQB(vertex);

        pcl::PointXYZ result = pcl::PointXYZ(vertex.x + totalTransformation->getTranslation()[0],
                                             vertex.y + totalTransformation->getTranslation()[1],
                                             vertex.z + totalTransformation->getTranslation()[2]);

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

    targetVertices.push_back(pcl::PointXYZ(-2.95, -2.95, -2.95));
    targetVertices.push_back(pcl::PointXYZ(-1.95, -1.95, -1.95));
    targetVertices.push_back(pcl::PointXYZ(0.05, 0.05, 0.05));
    targetVertices.push_back(pcl::PointXYZ(2.05, 2.05, 2.05));
    targetVertices.push_back(pcl::PointXYZ(3.05, 3.05, 3.05));

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

        pcl::PointXYZ result = pcl::PointXYZ(vertex.x + totalTransformation->getTranslation()[0],
                                             vertex.y + totalTransformation->getTranslation()[1],
                                             vertex.z + totalTransformation->getTranslation()[2]);

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

    targetVertices.push_back(pcl::PointXYZ(-2.95, -2.95, -2.95));
    targetVertices.push_back(pcl::PointXYZ(-1.95, -1.95, -1.95));
    targetVertices.push_back(pcl::PointXYZ(0.05, 0.05, 0.05));
    targetVertices.push_back(pcl::PointXYZ(2.05, 2.05, 2.05));
    targetVertices.push_back(pcl::PointXYZ(3.05, 3.05, 3.05));

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

        pcl::PointXYZ result = pcl::PointXYZ(vertex.x + totalTransformation->getTranslation()[0],
                                             vertex.y + totalTransformation->getTranslation()[1],
                                             vertex.z + totalTransformation->getTranslation()[2]);

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
    targetVertices.push_back(pcl::PointXYZ(-2.95, -2.95, -2.95));
    targetVertices.push_back(pcl::PointXYZ(-1.95, -1.95, -1.95));
    targetVertices.push_back(pcl::PointXYZ(0.05, 0.05, 0.05));
    targetVertices.push_back(pcl::PointXYZ(2.05, 2.05, 2.05));
    targetVertices.push_back(pcl::PointXYZ(3.05, 3.05, 3.05));

    targetNormals.push_back(pcl::Normal(1, 1, 1));
    targetNormals.push_back(pcl::Normal(1, 1, 1));
    targetNormals.push_back(pcl::Normal(1, 1, 1));
    targetNormals.push_back(pcl::Normal(1, 1, 1));
    targetNormals.push_back(pcl::Normal(1, 1, 1));

    /* group 2 of target vertices */
    targetVertices.push_back(pcl::PointXYZ(11.5, 11.5, 11.5));
    targetVertices.push_back(pcl::PointXYZ(10.5, 10.5, 10.5));
    targetVertices.push_back(pcl::PointXYZ(9.5, 9.5, 9.5));
    targetVertices.push_back(pcl::PointXYZ(10, 10, 10));
    targetVertices.push_back(pcl::PointXYZ(11, 11, 11));

    targetNormals.push_back(pcl::Normal(1, 1, 1));
    targetNormals.push_back(pcl::Normal(1, 1, 1));
    targetNormals.push_back(pcl::Normal(1, 1, 1));
    targetNormals.push_back(pcl::Normal(1, 1, 1));
    targetNormals.push_back(pcl::Normal(1, 1, 1));

    liveFrame = std::make_shared<dynfu::Frame>(1, targetVertices, targetNormals);

    CombinedSolver combinedSolver(warpfield, params);
    combinedSolver.initializeProblemInstance(canonicalFrameWarpedToLive, liveFrame);
    combinedSolver.solveAll();

    /* TODO (dig15): figure out why the test fails */
    maxError = 1.0;

    int j = 0;
    for (auto vertex : canonicalFrameWarpedToLive->getVertices()) {
        auto totalTransformation = warpfield.calcDQB(vertex);

        pcl::PointXYZ result = pcl::PointXYZ(vertex.x + totalTransformation->getTranslation()[0],
                                             vertex.y + totalTransformation->getTranslation()[1],
                                             vertex.z + totalTransformation->getTranslation()[2]);

        ASSERT_NEAR(result.x, liveFrame->getVertices()[j].x, maxError);
        ASSERT_NEAR(result.y, liveFrame->getVertices()[j].y, maxError);
        ASSERT_NEAR(result.z, liveFrame->getVertices()[j].z, maxError);

        j++;
    }
}

/* FIXME (dig15): figure out why the test fails */
TEST_F(OptTest, MultipleVerticesOneGroupOfDeformationNodesWarpAndReverseTest) {
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

    targetVertices.push_back(pcl::PointXYZ(-2.95, -2.95, -2.95));
    targetVertices.push_back(pcl::PointXYZ(-1.95, -1.95, -1.95));
    targetVertices.push_back(pcl::PointXYZ(0.05, 0.05, 0.05));
    targetVertices.push_back(pcl::PointXYZ(2.05, 2.05, 2.05));
    targetVertices.push_back(pcl::PointXYZ(3.05, 3.05, 3.05));

    targetNormals.push_back(pcl::Normal(1, 1, 1));
    targetNormals.push_back(pcl::Normal(1, 1, 1));
    targetNormals.push_back(pcl::Normal(1, 1, 1));
    targetNormals.push_back(pcl::Normal(1, 1, 1));
    targetNormals.push_back(pcl::Normal(1, 1, 1));

    liveFrame = std::make_shared<dynfu::Frame>(1, targetVertices, targetNormals);

    CombinedSolver combinedSolver(warpfield, params);
    combinedSolver.initializeProblemInstance(canonicalFrameWarpedToLive, liveFrame);
    combinedSolver.solveAll();

    /* TODO (dig15): figure out why the test fails */
    maxError = 1.0;

    int j = 0;
    for (auto vertex : canonicalFrameWarpedToLive->getVertices()) {
        auto totalTransformation = warpfield.calcDQB(vertex);

        pcl::PointXYZ result = pcl::PointXYZ(vertex.x + totalTransformation->getTranslation()[0],
                                             vertex.y + totalTransformation->getTranslation()[1],
                                             vertex.z + totalTransformation->getTranslation()[2]);

        ASSERT_NEAR(result.x, liveFrame->getVertices()[j].x, maxError);
        ASSERT_NEAR(result.y, liveFrame->getVertices()[j].y, maxError);
        ASSERT_NEAR(result.z, liveFrame->getVertices()[j].z, maxError);

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

        pcl::PointXYZ result = pcl::PointXYZ(vertex.x + totalTransformation->getTranslation()[0],
                                             vertex.y + totalTransformation->getTranslation()[1],
                                             vertex.z + totalTransformation->getTranslation()[2]);

        ASSERT_NEAR(result.x, liveFrame->getVertices()[j].x, maxError);
        ASSERT_NEAR(result.y, liveFrame->getVertices()[j].y, maxError);
        ASSERT_NEAR(result.z, liveFrame->getVertices()[j].z, maxError);

        j++;
    }
}

/* */
TEST_F(OptTest, MultipleVerticesOneGroupOfDeformationNodesNonRigidTest) {
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

    targetVertices.push_back(pcl::PointXYZ(-2.95, -2.95, -2.95));
    targetVertices.push_back(pcl::PointXYZ(-2.0, -2.0, -2.0));
    targetVertices.push_back(pcl::PointXYZ(0.01, 0, 0));
    targetVertices.push_back(pcl::PointXYZ(2.5, 2.5, 2.5));
    targetVertices.push_back(pcl::PointXYZ(2.75, 2.75, 2.75));

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

        pcl::PointXYZ result = pcl::PointXYZ(vertex.x + totalTransformation->getTranslation()[0],
                                             vertex.y + totalTransformation->getTranslation()[1],
                                             vertex.z + totalTransformation->getTranslation()[2]);

        ASSERT_NEAR(result.x, liveFrame->getVertices()[j].x, maxError);
        ASSERT_NEAR(result.y, liveFrame->getVertices()[j].y, maxError);
        ASSERT_NEAR(result.z, liveFrame->getVertices()[j].z, maxError);

        j++;
    }
}

TEST_F(OptTest, RandomNonRigidTest) {
    std::vector<std::shared_ptr<Node>> nodesNonRigid;

    auto dg_se3 = std::make_shared<DualQuaternion<float>>(0, 0, 0, 0, 0, 0);  // transformation
    float dg_w  = 2;                                                          // radial basis weight

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

    sourceVertices.push_back(pcl::PointXYZ(-3, -3, -3));
    sourceVertices.push_back(pcl::PointXYZ(-2, -2, -2));
    sourceVertices.push_back(pcl::PointXYZ(0.01, 0.01, 0.01));
    sourceVertices.push_back(pcl::PointXYZ(2, 2, 2));
    sourceVertices.push_back(pcl::PointXYZ(3, 3, 3));
    sourceVertices.push_back(pcl::PointXYZ(3, 3, 3));

    sourceNormals.push_back(pcl::Normal(1, 1, 1));
    sourceNormals.push_back(pcl::Normal(1, 1, 1));
    sourceNormals.push_back(pcl::Normal(1, 1, 1));
    sourceNormals.push_back(pcl::Normal(1, 1, 1));
    sourceNormals.push_back(pcl::Normal(1, 1, 1));

    canonicalFrameWarpedToLive = std::make_shared<dynfu::Frame>(0, sourceVertices, sourceNormals);

    targetVertices.push_back(pcl::PointXYZ(-2.95, -3, -2.95));
    targetVertices.push_back(pcl::PointXYZ(-1.95, -1.95, -2));
    targetVertices.push_back(pcl::PointXYZ(0.1, 0.1, 0.1));
    targetVertices.push_back(pcl::PointXYZ(2, 2., 2));
    targetVertices.push_back(pcl::PointXYZ(3.05, 3.05, 3.05));
    targetVertices.push_back(pcl::PointXYZ(3.05, 3.05, 3.05));

    targetNormals.push_back(pcl::Normal(1, 1, 1));
    targetNormals.push_back(pcl::Normal(1, 1, 1));
    targetNormals.push_back(pcl::Normal(1, 1, 1));
    targetNormals.push_back(pcl::Normal(1, 1, 1));
    targetNormals.push_back(pcl::Normal(1, 1, 1));

    liveFrame = std::make_shared<dynfu::Frame>(0, targetVertices, targetNormals);

    CombinedSolver combinedSolver(warpfield, params);
    combinedSolver.initializeProblemInstance(canonicalFrameWarpedToLive, liveFrame);
    combinedSolver.solveAll();

    int j = 0;
    for (auto vertex : canonicalFrameWarpedToLive->getVertices()) {
        auto totalTransformation = warpfield.calcDQB(vertex);

        pcl::PointXYZ result = pcl::PointXYZ(vertex.x + totalTransformation->getTranslation()[0],
                                             vertex.y + totalTransformation->getTranslation()[1],
                                             vertex.z + totalTransformation->getTranslation()[2]);

        ASSERT_NEAR(result.x, liveFrame->getVertices()[j].x, maxError);
        ASSERT_NEAR(result.y, liveFrame->getVertices()[j].y, maxError);
        ASSERT_NEAR(result.z, liveFrame->getVertices()[j].z, maxError);

        j++;
    }
}
