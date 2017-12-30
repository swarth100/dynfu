#pragma once

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

/* kinfu includes */
#include <kfusion/types.hpp>

/* dynfu includes */
#include <dynfu/utils/frame.hpp>
#include <dynfu/warp_field.hpp>

/* opt includes */
#include <CombinedSolverBase.h>
#include <CombinedSolverParameters.h>
#include <OptGraph.h>
#include <OptImage.h>

class CombinedSolver : public CombinedSolverBase {
public:
    CombinedSolver(std::shared_ptr<Warpfield> warpfield, CombinedSolverParameters params);

    void initializeProblemInstance(const std::shared_ptr<dynfu::Frame> canonicalFrame,
                                   const std::shared_ptr<dynfu::Frame> liveFrame);

    void initializeConnectivity(std::vector<cv::Vec3f> canonicalVertices);

    void combinedSolveInit() override;

    void preSingleSolve() override;

    void postSingleSolve() override;

    virtual void preNonlinearSolve(int iteration) override;

    virtual void postNonlinearSolve(int iteration) override;

    void combinedSolveFinalize() override;

    void resetGPUMemory();

    void copyResultToCPUFromFloat3();

private:
    std::shared_ptr<Warpfield> m_warpfield;
    CombinedSolverParameters m_solverParameters;

    std::vector<unsigned int> m_dims;  // curent index in the solver

    std::vector<cv::Vec3f> m_canonicalVerticesOpenCV;
    std::vector<cv::Vec3f> m_liveVerticesOpenCV;
    std::vector<cv::Vec3f> m_canonicalNormalsOpenCV;
    std::vector<cv::Vec3f> m_liveNormalsOpenCV;

    std::shared_ptr<OptImage> m_canonicalVertices;
    std::shared_ptr<OptImage> m_canonicalNormals;
    std::shared_ptr<OptImage> m_liveVertices;
    std::shared_ptr<OptImage> m_liveNormals;

    std::shared_ptr<OptGraph> m_dataGraph;

    std::shared_ptr<OptImage> m_nodeCoordinates;

    std::shared_ptr<OptImage> m_rotation;
    std::shared_ptr<OptImage> m_translation;
    std::shared_ptr<OptImage> m_radialBasisWeights;

    std::vector<cv::Vec3f> m_results;
};
