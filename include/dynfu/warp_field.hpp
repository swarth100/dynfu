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

typedef nanoflann::L2_Simple_Adaptor<float, nanoflann::PointCloud> nanoflannAdaptor;
typedef nanoflann::KDTreeSingleIndexAdaptor<nanoflannAdaptor, nanoflann::PointCloud, 3> kd_tree_t;

class Warpfield {
    /* Type for the index tree */
public:
    Warpfield();
    ~Warpfield();
    Warpfield(const Warpfield& w);

    /* initialise the warp field */
    void init(std::vector<std::shared_ptr<Node>> nodes);

    /* add new deformation node to the warp field */
    void addNode(Node newNode);

    /* return a vector of all nodes in the warp field */
    std::vector<std::shared_ptr<Node>> getNodes();

    /* return a dual quaternion which represents the dual quaternion blending for a point */
    std::shared_ptr<DualQuaternion<float>> calcDQB(cv::Vec3f point);

    /* warp a canonical frame according to the data stored in the warpfield */
    void warp(std::shared_ptr<dynfu::Frame> liveFrame);

    /* Find a set amount of closest neighbours */
    std::vector<std::shared_ptr<Node>> findNeighbors(int numNeighbor, cv::Vec3f vertex);

    /* Find index of set amount of closest neighbours */
    std::vector<size_t> findNeighborsIndex(int numNeighbor, cv::Vec3f vertex);

private:
    /* PCL frame counter */
    int frameNum = 0;

    /* list of currently held deformation nodes */
    std::vector<std::shared_ptr<Node>> nodes;

    /* cloud data */
    std::shared_ptr<nanoflann::PointCloud> cloud;

    /* KD-tree for deformation nodes */
    std::shared_ptr<kd_tree_t> kdTree;

    /* Getter for pcl cloud counter */
    int getFrameNum();

    /* Save the given vec3s to PCL format */
    void saveToPcl(std::vector<cv::Vec3f> vectors);
};

/* DYNFU_WARP_FIELD_HPP */
#endif
