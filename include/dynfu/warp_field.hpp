#ifndef DYNFU_WARP_FIELD_HPP
#define DYNFU_WARP_FIELD_HPP

/* dynfu includes */
#include <dynfu/utils/frame.hpp>
#include <dynfu/utils/node.hpp>

/* kinfu includes */
#include <kfusion/types.hpp>

/* nanoflann dependencies */
#include <nanoflann/nanoflann.hpp>
#include <nanoflann/pointcloud.hpp>

/* pcl includes */
#include <pcl/filters/voxel_grid.h>

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
public:
    Warpfield();
    ~Warpfield();

    /* initialise the warp field */
    void init(float epsilon, std::vector<std::shared_ptr<Node>> nodes);

    /* return a vector of all nodes in the warp field */
    std::vector<std::shared_ptr<Node>> getNodes();
    /* add new deformation node to the warp field */
    void addNode(std::shared_ptr<Node> newNode);

    /* get the set of vertices in the frame not supported by the warpfield */
    pcl::PointCloud<pcl::PointXYZ>::Ptr getUnsupportedVertices(std::shared_ptr<dynfu::Frame> frame);
    /* updates warpfield with new deformation nodes if unsupported vertices are detected */
    void update(std::shared_ptr<dynfu::Frame> frame);

    /* get kd-tree */
    std::shared_ptr<kd_tree_t> getKdTree();
    /* find a given no. of closest neighbours of a vertex */
    std::vector<std::shared_ptr<Node>> findNeighbors(int numNeighbor, pcl::PointXYZ vertex);
    /* find a given no. of closest neighbours of a vertex */
    std::vector<size_t> findNeighborsIndex(int numNeighbor, pcl::PointXYZ vertex);

    /* return a dual quaternion which represents the dual quaternion blending for a given point */
    std::shared_ptr<DualQuaternion<float>> calcDQB(pcl::PointXYZ point);

    /* warp canonical frame to live frame */
    std::shared_ptr<dynfu::Frame> warpToLive(std::shared_ptr<dynfu::Frame> canonicalFrame);

private:
    /* frame counter */
    int frameNum = 0;
    /* list of currently held deformation nodes */
    std::vector<std::shared_ptr<Node>> nodes;
    /* decimation density */
    float epsilon;
    /* cloud data */
    std::shared_ptr<nanoflann::PointCloud> cloud;
    /* KD-tree for deformation nodes */
    std::shared_ptr<kd_tree_t> kdTree;
    /* getter for the pcl cloud counter */
    int getFrameNum();
};

/* DYNFU_WARP_FIELD_HPP */
#endif
