#pragma once

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

/* dynfu includes */
#include <dynfu/utils/cuda_utils.h>
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
    CombinedSolver(Warpfield warpfield, CombinedSolverParameters params);

    void initializeProblemInstance(const std::shared_ptr<dynfu::Frame> canonicalFrame,
                                   const std::shared_ptr<dynfu::Frame> liveFrame);

    /* initialise data graph */
    void initializeDataGraph();
    /* initialise regularisation graph */
    void initializeRegGraph();

    void combinedSolveInit() override;
    void combinedSolveFinalize() override;

    void preSingleSolve() override;
    void postSingleSolve() override;

    virtual void preNonlinearSolve(int iteration) override;
    virtual void postNonlinearSolve(int iteration) override;

    /* set coordinates of vertices */
    void resetGPUMemory();
    /* update tukey biweights; for use pre non-linear solve */
    void updateTukeyBiweights();

    void copyResultToCPUFromFloat3();

private:
    Warpfield m_warpfield;
    CombinedSolverParameters m_solverParameters;

    std::vector<unsigned int> m_dims;  // curent index in the solver

    pcl::PointCloud<pcl::PointXYZ> m_canonicalVerticesPCL;
    pcl::PointCloud<pcl::PointXYZ> m_liveVerticesPCL;
    pcl::PointCloud<pcl::Normal> m_canonicalNormalsPCL;
    pcl::PointCloud<pcl::Normal> m_liveNormalsPCL;

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

    /* calculate tukey biweight given the tukey offset, parameter c, and the point error */
    float calcTukeyBiweight(float tukeyOffset, float c, pcl::PointXYZ ptError);
};
