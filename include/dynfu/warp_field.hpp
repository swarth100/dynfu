#ifndef DYNFU_WARP_FIELD_HPP
#define DYNFU_WARP_FIELD_HPP

/* kinfu includes */
#include <kfusion/types.hpp>

/* dynfu includes */
#include <dynfu/utils/frame.hpp>
#include <dynfu/utils/node.hpp>

/* ceres includes */
#include <ceres/ceres.h>

/* sys headers */
#include <iostream>
#include <memory>
#include <vector>

/* Nanoflann dependencies */
#include <nanoflann/nanoflann.hpp>
#include <nanoflann/pointcloud.hpp>

/* Set max amount of closest neighbours to consider */
#define KNN 8

typedef nanoflann::L2_Simple_Adaptor<float, PointCloud> nanoflannAdaptor;
typedef nanoflann::KDTreeSingleIndexAdaptor<nanoflannAdaptor, PointCloud, 3> kd_tree_t;

class Warpfield {
    /* Type for the index tree */
public:
    Warpfield();
    ~Warpfield();
    Warpfield(const Warpfield& w);

    /* Initialises the warpfield's canonical Frame*/
    void init(std::vector<std::shared_ptr<Node>> nodes);

    /* Finds a set amount of closest neighbours */
    std::vector<std::shared_ptr<Node>> findNeighbors(int numNeighbor, cv::Vec3f vertex);

    /* Warps the given field according to the solver's deformation node data */
    void warp(std::shared_ptr<Frame> liveFrame);

    /* Returns a vector of all nodes in the warp field. */
    std::vector<std::shared_ptr<Node>> getNodes();

    void addNode(Node newNode);

private:
    /* Save the given vec3s to PCL format */
    void saveToPcl(std::vector<cv::Vec3f> vectors);

    /* Getter for pcl cloud counter */
    int getFrameNum();

    /* PCL Frame counter */
    int frameNum = 0;

    /* List of currently held deformation nodes */
    std::vector<std::shared_ptr<Node>> nodes;

    /* KD-tree for deformation nodes */
    std::shared_ptr<kd_tree_t> kdTree;

    /* Holds cloud data */
    std::shared_ptr<PointCloud> cloud;
};

/* DYNFU_WARP_FIELD_HPP */
#endif
