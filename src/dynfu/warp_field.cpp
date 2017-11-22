#include <dynfu/warp_field.hpp>

/* sys headers */
#include <cmath>
#include <ctgmath>

/* TODO: Add comment */
Warpfield::Warpfield() {
    PointCloud cloud;
    std::vector<cv::Vec3f> warp_init;
    warp_init.emplace_back(cv::Vec3f(1, 1, 1));
    warp_init.emplace_back(cv::Vec3f(1, 1, -1));
    warp_init.emplace_back(cv::Vec3f(1, -1, 1));
    warp_init.emplace_back(cv::Vec3f(1, -1, -1));
    warp_init.emplace_back(cv::Vec3f(-1, 1, 1));
    warp_init.emplace_back(cv::Vec3f(-1, 1, -1));
    warp_init.emplace_back(cv::Vec3f(-1, -1, 1));
    warp_init.emplace_back(cv::Vec3f(-1, -1, -1));
    warp_init.emplace_back(cv::Vec3f(0, 0, 0));
    cloud.pts = warp_init;

    kdTree = std::make_shared<kd_tree_t>(3, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    kdTree->buildIndex();

    /* Attempt to initiate a warp field with 9 vectors.
     * The vectors correspond to the 8 vertices of a cube + the origin
     */

    std::vector<cv::Vec3f> canonical_vertices;
    canonical_vertices.emplace_back(cv::Vec3f(0, 0, 0));
    canonical_vertices.emplace_back(cv::Vec3f(-1, -1, -1));
    canonical_vertices.emplace_back(cv::Vec3f(1, 1, 1));
    canonical_vertices.emplace_back(cv::Vec3f(2, 2, 2));
    canonical_vertices.emplace_back(cv::Vec3f(3, 3, 3));


    for(auto v : canonical_vertices) {
        auto res = findNeighbors(1, v);
        for (auto r : res) {
            std::cout << r << std::endl;
        }
    }
}

/* TODO: Add comment */
Warpfield::~Warpfield() = default;

/* TODO: Add comment */
void Warpfield::init(std::vector<std::shared_ptr<Node>> /*nodes*/) {
    // initialise all deformation nodes
}

void Warpfield::warp(std::shared_ptr<Frame> /*liveFrame*/) {
    // calculate DQB for all points
    // warps all points

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
}

/*
 * Returns a vector of all nodes in the warp field.
 */
std::vector<std::shared_ptr<Node>> Warpfield::getNodes() {}

/* Find the index of k closest neighbour for the given point */
std::vector<size_t> Warpfield::findNeighbors(int numNeighbour, cv::Vec3f point) {
    /* Not used, ignores the distance to the nodes for now */
    std::vector<float> outDistSqr(numNeighbour);
    std::vector<size_t> retIndex(numNeighbour);
    /* Unpack the Vec3f into vector */
    std::vector<float> query = {point[0], point[1], point[2]};
    int n                    = kdTree->knnSearch(&query[0], numNeighbour, &retIndex[0], &outDistSqr[0]);
    retIndex.resize(n);
    return retIndex;
}

// void Warpfield::addNode(Node newNode) { nodes.emplace_back(newNode); }
