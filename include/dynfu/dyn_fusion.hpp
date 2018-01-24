#pragma once

/* dynfu includes */
#include <dynfu/utils/dual_quaternion.hpp>
#include <dynfu/utils/frame.hpp>
#include <dynfu/utils/opt_solver.hpp>
#include <dynfu/utils/pointcloud_viz.hpp>
#include <dynfu/warp_field.hpp>

/* kinfu includes */
#include <kfusion/kinfu.hpp>

/* pcl includes */
#include <pcl/PolygonMesh.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <pcl/surface/marching_cubes_rbf.h>

/* sys headers */
#include <math.h>
#include <thread>

/* dynfu parameters */
struct DynFuParams {
    /* set default dynfu params */
    static DynFuParams defaultParams();

    /* kinfu params to use in dynfu */
    kfusion::KinFuParams kinfuParams;

    float tukeyOffset;

    float lambda;   /* regularisation parameter */
    float psi_data; /* parameter to calculate tukey biweights */
    float psi_reg;  /* parameter to calculate huber weights */

    int L;    /* no. of levels in the regularisation hierarchy */
    int beta; /* ??? */

    float epsilon; /* decimation density */
};

/* */
class DynFusion : public kfusion::KinFu {
public:
    /* default constructor */
    DynFusion(const DynFuParams &params);
    /* default destructor */
    ~DynFusion();

    /* get dynfu params */
    DynFuParams &params();

    /* run algorithm on all frames */
    bool operator()(const kfusion::cuda::Depth &depth, const kfusion::cuda::Image &image = kfusion::cuda::Image());

    /* initialise dynfu */
    void init(pcl::PointCloud<pcl::PointXYZ> &canonicalVertices, pcl::PointCloud<pcl::Normal> &canonicalNormals);
    /* initialise canonical frame with vertices and normals */
    void initCanonicalFrame(pcl::PointCloud<pcl::PointXYZ> &vertices, pcl::PointCloud<pcl::Normal> &normals);

    /* update the current live frame */
    void addLiveFrame(int frameID, pcl::PointCloud<pcl::PointXYZ> &vertices, pcl::PointCloud<pcl::Normal> &normals);

    /* warp canonical frame to live frame using Opt */
    void warpCanonicalToLiveOpt(cv::Affine3f affine);

    /* get the canonical frame warped to live */
    std::shared_ptr<dynfu::Frame> getCanonicalWarpedToLive();

private:
    /* algo parameters */
    DynFuParams dynfuParams;

    /* canonical frame */
    std::shared_ptr<dynfu::Frame> canonicalFrame;
    /* canonical frame warped to live */
    std::shared_ptr<dynfu::Frame> canonicalFrameWarpedToLive;
    /* live frame */
    std::shared_ptr<dynfu::Frame> liveFrame;

    /* warpfield */
    std::shared_ptr<Warpfield> warpfield;

    /* find the corresponding vertices and normals of canonical frame in the live frame */
    std::shared_ptr<dynfu::Frame> findCorrespondingFrame(pcl::PointCloud<pcl::PointXYZ> canonicalVertices,
                                                         pcl::PointCloud<pcl::Normal> canonicalNormals,
                                                         pcl::PointCloud<pcl::PointXYZ> liveVertices);
};
