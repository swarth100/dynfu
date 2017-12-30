#ifndef DYNFU_DYNFUSION_HPP
#define DYNFU_DYNFUSION_HPP

/* ceres includes */
#include <ceres/ceres.h>

/* dynfu includes */
#include <dynfu/utils/ceres_solver.hpp>
#include <dynfu/utils/dual_quaternion.hpp>
#include <dynfu/utils/frame.hpp>
#include <dynfu/utils/opt_solver.hpp>
#include <dynfu/utils/pointcloud_viz.hpp>
#include <dynfu/warp_field.hpp>

/* kinfu includes */
#include <kfusion/types.hpp>

/* pcl includes */
#include <pcl/PolygonMesh.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <pcl/surface/marching_cubes_rbf.h>

/* sys headers */
#include <math.h>
#include <thread>

using namespace kfusion;
using namespace kfusion::cuda;

/* */
class DynFusion {
public:
    DynFusion();
    ~DynFusion();

    void init(kfusion::cuda::Cloud &vertices, kfusion::cuda::Normals &normals);

    void initCanonicalFrame(std::vector<cv::Vec3f> &vertices, std::vector<cv::Vec3f> &normals);

    // void updateCanonicalFrame();

    /* update the warp field if the no. of deformation nodes is insufficient to capture the geometry of canonical
     * model
     */
    void updateWarpfield();
    /* update the affine transformation from icp */
    void updateAffine(cv::Affine3f newAffine);

    /* get the affine transformation */
    cv::Affine3f getLiveToCanonicalAffine();

    /* warp canonical frame to live frame using Ceres */
    void warpCanonicalToLive();
    /* warp canonical frame to live frame using Opt */
    void warpCanonicalToLiveOpt();

    /* update the current live frame */
    void addLiveFrame(int frameID, kfusion::cuda::Cloud &vertices, kfusion::cuda::Normals &normals);

    /* construct a polygon mesh from the vertices and normals via marching cubes */
    pcl::PolygonMesh reconstructSurface();

    /* get the canonical frame warped to live */
    std::shared_ptr<dynfu::Frame> getCanonicalWarpedToLive();

    /* control the thread deletion */
    static bool nextFrameReady;

    /* convert vec3f to OpenCV matrix */
    cv::Mat vectorToMat(std::vector<cv::Vec3f> vec);
    /* convert OpenCV matrix to cloud */
    kfusion::cuda::Cloud matToCloud(cv::Mat matrix);

private:
    std::shared_ptr<dynfu::Frame> canonicalFrame;
    std::shared_ptr<dynfu::Frame> canonicalFrameAffine;
    std::shared_ptr<dynfu::Frame> canonicalWarpedToLive;
    std::shared_ptr<dynfu::Frame> liveFrame;

    cv::Affine3f affineLiveToCanonical;
    std::shared_ptr<Warpfield> warpfield;

    /* find the corresponding vertices/normals of canonical frame in the live vertices */
    std::shared_ptr<dynfu::Frame> findCorrespondingFrame(std::vector<cv::Vec3f> canonicalVertices,
                                                         std::vector<cv::Vec3f> canonicalNormals,
                                                         std::vector<cv::Vec3f> liveVertices);

    /* check if kfusion::point contains nan's */
    static bool isNaN(kfusion::Point pt);
    /* check if kfusion::point is 0 */
    static bool isZero(kfusion::Point pt);

    /* convert cloud to OpenCV matrix */
    cv::Mat cloudToMat(kfusion::cuda::Cloud cloud);
    /* convert normals to OpenCV matrix */
    cv::Mat normalsToMat(kfusion::cuda::Normals normals);
    /* convert OpenCV matrix to vector of Vec3f */
    std::vector<cv::Vec3f> matToVector(cv::Mat);
};

/* DYNFU_DYNFUSION_HPP */
#endif
