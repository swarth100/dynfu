#pragma once

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

/* dynfu includes */
#include <dynfu/utils/frame.hpp>
#include <dynfu/warp_field.hpp>

/* kinfu includes */
#include <kfusion/types.hpp>

/* opt includes */
#include <CombinedSolverBase.h>
#include <CombinedSolverParameters.h>
#include <OptGraph.h>
#include <OptImage.h>

class CombinedSolver : public CombinedSolverBase {
public:
    /* default constructor */
    CombinedSolver(Warpfield warpfield, CombinedSolverParameters params, float tukeyOffset, float psi_data,
                   float lambda, float psi_reg);

    /* init the problem */
    void initializeProblemInstance(const std::shared_ptr<dynfu::Frame> canonicalFrame,
                                   const std::shared_ptr<dynfu::Frame> liveFrame);

    /* initialise data graph */
    void initializeDataGraph();
    /* initialise regularisation graph */
    void initializeRegGraph();

    /* set solver params and problem params */
    void combinedSolveInit() override;
    /* do nothing */
    void combinedSolveFinalize() override;

    /* do nothing */
    void preSingleSolve() override;
    /* copy results to cpu */
    void postSingleSolve() override;

    /* set tukey biweights and huber weights */
    virtual void preNonlinearSolve(int iteration) override;
    /* do nothing */
    virtual void postNonlinearSolve(int iteration) override;

    /* set coordinates of vertices */
    void resetGPUMemory();
    /* update tukey biweights; for use pre non-linear solve */
    void updateTukeyBiweights();
    /* update huber weights; for use pre non-linear solve */
    void updateHuberWeights();

    /* copy results from gpu to cpu */
    void copyResultToCPUFromFloat3();

private:
    /* warpfield to use in the solver */
    Warpfield m_warpfield;
    /* solver parameters */
    CombinedSolverParameters m_solverParameters;

    float tukeyOffset;
    /* parameter to calculate tukey biweights */
    float psi_data;

    /* regularisation parameter */
    float lambda;
    /* parameter to calculate huber weights */
    float psi_reg;

    /* current index in the solver */
    std::vector<unsigned int> m_dims;

    /* canonical vertices stored in a pcl pointcloud */
    pcl::PointCloud<pcl::PointXYZ> m_canonicalVerticesPCL;
    /* live vertices stored in a pcl pointcloud */
    pcl::PointCloud<pcl::PointXYZ> m_liveVerticesPCL;
    /* canonical normals stored in a pcl pointcloud */
    pcl::PointCloud<pcl::Normal> m_canonicalNormalsPCL;
    /* live normals stored in a pcl pointcloud */
    pcl::PointCloud<pcl::Normal> m_liveNormalsPCL;

    /* OPT IMAGES */
    std::shared_ptr<OptImage> m_canonicalVertices;
    std::shared_ptr<OptImage> m_canonicalNormals;
    std::shared_ptr<OptImage> m_liveVertices;
    std::shared_ptr<OptImage> m_liveNormals;

    std::shared_ptr<OptGraph> m_dataGraph;
    std::shared_ptr<OptGraph> m_regGraph;

    std::shared_ptr<OptImage> m_dg_v;
    std::shared_ptr<OptImage> m_translations;
    std::shared_ptr<OptImage> m_rotations;
    std::shared_ptr<OptImage> m_dg_w;

    std::shared_ptr<OptImage> m_tukeyBiweights;
    std::shared_ptr<OptImage> m_huberWeights;

    /* per-term regularisation weight */
    float w_reg;

    /* calculate tukey biweight */
    float calcTukeyBiweight(float tukeyOffset, float c, pcl::PointXYZ ptError);
    /* calculate huber weight */
    float calcHuberWeight(float k, float e);
};
