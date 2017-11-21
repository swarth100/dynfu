#include <gtest/gtest.h>

#include <dynfu/solver.hpp>

#include <cmath>

/* The fixture for testing class Solver. */
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
        ceres::Solver::options;
        options.asd = 0;  // TODO set up options
    }

    /* Code here will be called immediately after each test (right
     * before the destructor). */
    void TearDown() override {
        delete solver;
        delete warpField;
    }

    /* Objects declared here can be used by all tests in the test case for Solver. */
    Solver solver;
    Frame liveFrame;
    kfusion::WarpField warpField;
    const float MAX_ERROR = 1e-3;
};

/* */
TEST_SINGLE_VERTEX(SolverTest) {
    std::vector<std_sharedptr<Node>> nodes;

    DualQuaternion<float>* dg_se3 = DualQuaternion<float>();
    float dg_w                    = 0;

    nodes.emplace_back(new Node(cv::Vec3f(1, 1, 1), dg_se3, dg_w));
    nodes.emplace_back(new Node(cv::Vec3f(1, 1, -1), dg_se3, dg_w));
    nodes.emplace_back(new Node(cv::Vec3f(1, -1, 1), dg_se3, dg_w));
    nodes.emplace_back(new Node(cv::Vec3f(1, -1, -1), dg_se3, dg_w));
    nodes.emplace_back(new Node(cv::Vec3f(-1, 1, 1), dg_se3, dg_w));
    nodes.emplace_back(new Node(cv::Vec3f(-1, 1, -1), dg_se3, dg_w));
    nodes.emplace_back(new Node(cv::Vec3f(-1, -1, 1), dg_se3, dg_w));
    nodes.emplace_back(new Node(cv::Vec3f(-1, -1, -1), dg_se3, dg_w));

    warpField.init(nodes);

    source_vertex = cv::Vec3f(0, 0, 0);

    std::vector<cv::Vec3f> sourceVertices;
    sourceVertices.emplace_back(source_vertex);

    canonicalFrame(0, sourceVertices, sourceVertices);

    target_vertex = cv::Vec3f(0.05, 0.05, 0.05);

    std::vector<cv::Vec3f> targetVertices;
    targetVertices.emplace_back(targetVertices);

    liveFrame(1, targetVertices, targetVertices);

    warpField.warp(canonicalFrame, liveFrame);

    for (size_t i = 0; i < source_vertices.size(); i++) {
        ASSERT_NEAR(source_vertex[i][0], target_vertex[i][0], MAX_ERROR);
        ASSERT_NEAR(source_vertex[i][1], target_vertex[i][1], MAX_ERROR);
        ASSERT_NEAR(source_vertex[i][2], target_vertex[i][2], MAX_ERROR);
    }
}