#ifndef DYNFU_DYNFUSION_HPP
#define DYNFU_DYNFUSION_HPP

#include <dynfu/utils/dual_quaternion.hpp>
#include <dynfu/utils/frame.hpp>
#include <dynfu/warp_field.hpp>

/* Typedefs */
#include <kfusion/types.hpp>

#include <math.h>

/* */
class DynFusion {
public:
    DynFusion();
    DynFusion(std::vector<cv::Vec3f> vertices, std::vector<cv::Vec3f> normals);
    ~DynFusion();

    void init(kfusion::cuda::Cloud &vertices);

    void initCanonicalFrame();
    // void updateCanonicalFrame();
    void warpCanonicalToLive();

    /* Returns a dual quaternion which represents the dual quaternion blending for a point */
    std::shared_ptr<DualQuaternion<float>> calcDQB(cv::Vec3f point);
    /* Update the current live frame */
    void addLiveFrame(int frameID, kfusion::cuda::Cloud &vertices, kfusion::cuda::Normals &normals);
    /* Get weight of node on point for DQB */
    static float getWeight(std::shared_ptr<Node> node, cv::Vec3f point);

    template <typename T, typename S>
    static S getWeightT(T position, S weight, T point) {
        float distance_norm = cv::norm(position - point);
        return exp((-1 * pow(distance_norm, 2)) / (2 * pow(weight, 2)));
    }

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
