#ifndef DYNFU_WARP_FIELD_HPP
#define DYNFU_WARP_FIELD_HPP

/* ceres includes */
#include <ceres/ceres.h>

/* dynfu includes */
#include <dynfu/utils/frame.hpp>
#include <dynfu/utils/node.hpp>

/* kinfu includes */
#include <kfusion/types.hpp>

/* nanoflann dependencies */
#include <nanoflann/nanoflann.hpp>
#include <nanoflann/pointcloud.hpp>

/* sys headers */
#include <cmath>
#include <ctgmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

/* set the no. of nearest neighbours to consider */
#define KNN 8

typedef nanoflann::L2_Simple_Adaptor<float, nanoflann::PointCloud> nanoflannAdaptor;
typedef nanoflann::KDTreeSingleIndexAdaptor<nanoflannAdaptor, nanoflann::PointCloud, 3> kd_tree_t;

class Warpfield {
    /* Type for the index tree */
public:
    Warpfield();
    Warpfield(const Warpfield& w);

    ~Warpfield();

    /* initialise the warp field */
    void init(std::vector<std::shared_ptr<Node>> nodes);

    /* add new deformation node to the warp field */
    void addNode(std::shared_ptr<Node> newNode);

    /* return a vector of all nodes in the warp field */
    std::vector<std::shared_ptr<Node>> getNodes();

    /* return a dual quaternion which represents the dual quaternion blending for a point */
    std::shared_ptr<DualQuaternion<float>> calcDQB(cv::Vec3f point);

    /* warp the live frame to canonical frame */
    std::shared_ptr<dynfu::Frame> warpToCanonical(std::shared_ptr<dynfu::Frame> liveFrame);
    /* warp the canonical frame to live frame */
    std::shared_ptr<dynfu::Frame> warpToLive(std::shared_ptr<dynfu::Frame> canonicalFrame);

    /* find a given no. of closest neighbours */
    std::vector<std::shared_ptr<Node>> findNeighbors(int numNeighbor, cv::Vec3f vertex);

    /* find a given no. of closest neighbours */
    std::vector<size_t> findNeighborsIndex(int numNeighbor, cv::Vec3f vertex);

private:
    /* frame counter */
    int frameNum = 0;

    /* list of currently held deformation nodes */
    std::vector<std::shared_ptr<Node>> nodes;

    /* cloud data */
    std::shared_ptr<nanoflann::PointCloud> cloud;

    /* KD-tree for deformation nodes */
    std::shared_ptr<kd_tree_t> kdTree;

    /* Getter for pcl cloud counter */
    int getFrameNum();
};

/* DYNFU_WARP_FIELD_HPP */
#endif
