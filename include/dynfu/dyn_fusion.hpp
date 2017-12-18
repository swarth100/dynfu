#ifndef DYNFU_DYNFUSION_HPP
#define DYNFU_DYNFUSION_HPP

/* dynfu includes */
#include <dynfu/utils/ceres_solver.hpp>
#include <dynfu/utils/dual_quaternion.hpp>
#include <dynfu/utils/frame.hpp>
#include <dynfu/warp_field.hpp>

/* ceres includes */
#include <ceres/ceres.h>

/* typedefs */
#include <kfusion/types.hpp>

/* sys headers */
#include <math.h>

/* */
class DynFusion {
public:
    DynFusion();
    ~DynFusion();

    void init(kfusion::cuda::Cloud &vertices);

    void initCanonicalFrame(std::vector<cv::Vec3f> vertices, std::vector<cv::Vec3f> normals);
    // void updateCanonicalFrame();
    void warpCanonicalToLive();

    /* update the current live frame */
    void addLiveFrame(int frameID, kfusion::cuda::Cloud &vertices, kfusion::cuda::Normals &normals);

private:
    std::shared_ptr<Frame> canonicalFrame;
    std::shared_ptr<Frame> liveFrame;
    std::shared_ptr<Frame> canonicalWarpedToLive;
    std::shared_ptr<Warpfield> warpfield;

    /* Convert the cloud to opencv matrix */
    cv::Mat cloudToMat(kfusion::cuda::Cloud cloud);
    /* Convert the normals to opencv matrix */
    cv::Mat normalsToMat(kfusion::cuda::Normals normals);
    std::vector<cv::Vec3f> matToVector(cv::Mat);
};

/* DYNFU_DYNFUSION_HPP */
#endif
