#include <dynfu/warp_field.hpp>

/* sys headers */
#include <cmath>
#include <ctgmath>

/* TODO: Add comment */
Warpfield::Warpfield() = default;

/* TODO: Add comment */
Warpfield::~Warpfield() {
    delete cloud;
}

/* TODO: Add comment */
void Warpfield::init(std::vector<std::shared_ptr<Node>>& nodes) {
    // initialise all deformation nodes
    this->nodes = nodes;

    /* Hold deformation nodes position */
    std::vector<cv::Vec3f> deformationNodesPosition;

    for (auto node : this->nodes) {
        deformationNodesPosition.push_back(node->getPosition());
    }
    cloud = new PointCloud;
    cloud->pts = deformationNodesPosition;
    kdTree    = std::make_shared<kd_tree_t>(3, *cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    kdTree->buildIndex();
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

/* Find the nodes index of k closest neighbour for the given point */
std::vector<std::shared_ptr<Node>> Warpfield::findNeighbors(int numNeighbor, std::shared_ptr<Node> node) {
    auto point = node->getPosition();
    std::cout << "point" << point[0] << " " << point[1] << " " << point[2] << std::endl;
    /* Not used, ignores the distance to the nodes for now */
    std::vector<float> outDistSqr(numNeighbor);
    std::vector<size_t> retIndex(numNeighbor);
    /* Unpack the Vec3f into vector */
    std::vector<float> query = {point[0], point[1], point[2]};
    int n                    = kdTree->knnSearch(&query[0], numNeighbor, &retIndex[0], &outDistSqr[0]);
    retIndex.resize(n);
    std::vector<std::shared_ptr<Node>> neighborNodes;
    for (auto index : retIndex) {
        neighborNodes.push_back(nodes[index]);
    }
    return neighborNodes;
}

// void Warpfield::addNode(Node newNode) { nodes.emplace_back(newNode); }
