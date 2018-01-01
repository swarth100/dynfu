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

/* */
class DynFusion : public kfusion::KinFu {
public:
    DynFusion(const kfusion::KinFuParams &params);

    ~DynFusion();

    bool operator()(const kfusion::cuda::Depth &depth, const kfusion::cuda::Image &image = kfusion::cuda::Image());

    void init(kfusion::cuda::Cloud &vertices, kfusion::cuda::Normals &normals);

    void initCanonicalFrame(pcl::PointCloud<pcl::PointXYZ> &vertices, pcl::PointCloud<pcl::Normal> &normals);

    // void updateCanonicalFrame();

    /* update the warp field if the no. of deformation nodes is insufficient to capture the geometry of canonical
     * model
     */
    void updateWarpfield();

    /* update the affine transformation from icp */
    void updateAffine(cv::Affine3f newAffine);

    /* get the affine transformation */
    cv::Affine3f getLiveToCanonicalAffine();

    /* warp canonical frame to live frame using Opt */
    void warpCanonicalToLiveOpt();

    /* update the current live frame */
    void addLiveFrame(int frameID, kfusion::cuda::Cloud &vertices, kfusion::cuda::Normals &normals);

    /* construct a polygon mesh from the vertices and normals via marching cubes */
    void reconstructSurface();

    /* get the canonical frame warped to live */
    std::shared_ptr<dynfu::Frame> getCanonicalWarpedToLive();
    /* get the surface of the volume warped from canonical to live */
    pcl::PolygonMesh getCanonicalWarpedToLiveSurface();

    /* control the thread deletion */
    static bool nextFrameReady;

    /* convert OpenCV matrix to PointCloud<PointXYZ> */
    pcl::PointCloud<pcl::PointXYZ> matToPointCloudVertices(cv::Mat matrix);
    /* convert OpenCV matrix to PointCloud<Normal> */
    pcl::PointCloud<pcl::Normal> matToPointCloudNormals(cv::Mat matrix);
    /* convert vec3f to OpenCV matrix */
    cv::Mat vectorToMat(std::vector<cv::Vec3f> vec);
    /* convert OpenCV matrix to cloud */
    kfusion::cuda::Cloud matToCloud(cv::Mat matrix);

    void renderCanonicalWarpedToLive(kfusion::cuda::Image /* &image */, int /* flag */);

private:
    std::shared_ptr<dynfu::Frame> canonicalFrame;

    std::shared_ptr<dynfu::Frame> canonicalFrameAffine;

    std::shared_ptr<dynfu::Frame> canonicalWarpedToLive;
    pcl::PolygonMesh canonicalWarpedToLiveMesh;

    std::shared_ptr<dynfu::Frame> liveFrame;

    cv::Affine3f affineLiveToCanonical;
    std::shared_ptr<Warpfield> warpfield;

    /* find the corresponding vertices/normals of canonical frame in the live vertices */
    std::shared_ptr<dynfu::Frame> findCorrespondingFrame(pcl::PointCloud<pcl::PointXYZ> canonicalVertices,
                                                         pcl::PointCloud<pcl::Normal> canonicalNormals,
                                                         pcl::PointCloud<pcl::PointXYZ> liveVertices);

    /* check if kfusion::point contains nan's */
    static bool isNaN(kfusion::Point pt);
    /* check if kfusion::point is 0 */
    static bool isZero(kfusion::Point pt);

    /* convert cloud to OpenCV matrix */
    cv::Mat cloudToMat(kfusion::cuda::Cloud cloud);
    /* convert normals to OpenCV matrix */
    cv::Mat normalsToMat(kfusion::cuda::Normals normals);
};
