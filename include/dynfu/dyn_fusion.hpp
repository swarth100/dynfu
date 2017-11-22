#ifndef DYNFU_DYNFUSION_HPP
#define DYNFU_DYNFUSION_HPP

#include <dynfu/utils/frame.hpp>
#include <dynfu/utils/solver.hpp>
#include <dynfu/warp_field.hpp>

/* Typedefs */
#include <kfusion/types.hpp>

/* */
class DynFusion {
public:
    DynFusion(std::vector<cv::Vec3f> vertices, std::vector<cv::Vec3f> normals);
    ~DynFusion();

    void init(kfusion::cuda::Cloud &vertices);

    void initCanonicalFrame();
    // void updateCanonicalFrame();
    void warpCanonicalToLive();

private:
    std::shared_ptr<Frame> canonicalFrame;
    std::shared_ptr<Frame> liveFrame;
    std::shared_ptr<Frame> canonicalWarpedToLive;
    std::shared_ptr<Warpfield> warpfield;
    std::shared_ptr<Solver<float>> solver;
};

/* DYNFU_DYNFUSION_HPP */
#endif
