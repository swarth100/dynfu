#ifndef DYNFU_WARP_FIELD_HPP
#define DYNFU_WARP_FIELD_HPP

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
#define KNN_NEIGHBOURS 8

/* Type for the index tree */
typedef nanoflann::L2_Simple_Adaptor<float, PointCloud> nanoflannAdaptor;
typedef nanoflann::KDTreeSingleIndexAdaptor<nanoflannAdaptor, PointCloud, 3> kd_tree_t;

class Warpfield {
public:
    Warpfield();
    ~Warpfield();

    /* Initialises the warpfield's canonical Frame*/
    void init(std::vector<std::shared_ptr<Node>> nodes);

    /* Finds a set amount of closest neighbours */
    std::vector<int> findNeighbors(int numNeighbour, cv::Vec3f point);

    /* Warps the given field according to the solver's deformation node data */
    void warp(std::shared_ptr<Frame> liveFrame);

    /* Returns a vector of all nodes in the warp field. */
    std::vector<std::shared_ptr<Node>> getNodes();

    void addNode(Node newNode);

private:
    /* List of currently held deformation nodes */
    std::vector<std::shared_ptr<Node>> nodes;

    /* Original canonical frame */
    std::shared_ptr<Frame> canonicalFrame;
};

/* DYNFU_WARP_FIELD_HPP */
#endif
