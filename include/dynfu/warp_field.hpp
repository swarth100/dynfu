#ifndef DYNFU_WARP_FIELD_HPP
#define DYNFU_WARP_FIELD_HPP

#include <iostream>
#include <nanoflann/nanoflann.hpp>
#include <nanoflann/pointcloud.hpp>
class Warpfield {
private:
    /* KD-tree for the deformation nodes */
public:
    /* Type for the index tree */
    typedef nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, PointCloud>, PointCloud, 3>
        kd_tree_t;

    Warpfield();
    //~Warpfield();
};

/* DYNFU_WARP_FIELD_HPP */
#endif
