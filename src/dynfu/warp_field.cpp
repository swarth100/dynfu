#include <dynfu/warp_field.hpp>

/* Sys Headers */
#include <cmath>
#include <ctgmath>
#include <string>

/* PCL Headers */
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

/* TODO: Add comment */
Warpfield::Warpfield() = default;

/* TODO: Add comment */
Warpfield::~Warpfield() { delete cloud; }

/* TODO: Add comment */
void Warpfield::init(std::vector<std::shared_ptr<Node>>& nodes) {
    // initialise all deformation nodes
    this->nodes = nodes;

    /* Hold deformation nodes position */
    std::vector<cv::Vec3f> deformationNodesPosition;

    for (auto node : this->nodes) {
        deformationNodesPosition.push_back(node->getPosition());
    }

    /* Save the deformation Nodes to PCL format */
    this->saveToPcl(deformationNodesPosition);

    cloud      = new PointCloud;
    cloud->pts = deformationNodesPosition;
    kdTree     = std::make_shared<kd_tree_t>(3, *cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
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

/* -------------------------------------------------------------------------- */
/* PRIVATE METHODS */

/* */
int Warpfield::getFrameNum() { return this->frameNum++; }

/* */
void Warpfield::saveToPcl(std::vector<cv::Vec3f> vectors) {
    /* Initiate the PCL Cloud */
    auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();

    (*cloud).width  = vectors.size();
    (*cloud).height = 1;

    (*cloud).points.resize((*cloud).width * (*cloud).height);

    /* Iterate through vectors */
    for (size_t i = 0; i < vectors.size(); i++) {
        const cv::Vec3f& pt = vectors[i];

        pcl::PointXYZ point = pcl::PointXYZ();
        point.x             = pt[0];
        point.y             = pt[1];
        point.z             = pt[2];

        (*cloud).points[i] = point;
    }

    /* Save to PCL */
    std::string filenameStr = ("files/PCLFrame" + std::to_string(this->getFrameNum()) + ".pcd");
    pcl::io::savePCDFileASCII(filenameStr, (*cloud));

    /* Print out to stderr when after successful save */
    std::cout << "Saved " << (*cloud).points.size() << " data points to " << filenameStr << std::endl;
}

// void Warpfield::addNode(Node newNode) { nodes.emplace_back(newNode); }
