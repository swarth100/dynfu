#pragma once

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
    CombinedSolver(Warpfield warpfield, CombinedSolverParameters params);

    void initializeProblemInstance(std::shared_ptr<Frame> canonicalFrame, std::shared_ptr<Frame> liveFrame);

    void initializeConnectivity(const std::vector<cv::Vec3f> canonicalVertices);

    void combinedSolveInit() override;

    void preSingleSolve() override;

    void postSingleSolve() override;

    void preNonlinearSolve(int) override;

    void postNonlinearSolve(int) override;

    void combinedSolveFinalize() override;

    void resetGPUMemory();

    std::vector<cv::Vec3f> getResult();

    void copyResultToCPUFromFloat3();

private:
    Warpfield m_warpfield;

    std::vector<cv::Vec3f> m_canonicalVerticesOpenCV;
    std::vector<cv::Vec3f> m_liveVerticesOpenCV;
    std::vector<cv::Vec3f> m_canonicalNormalsOpenCV;
    std::vector<cv::Vec3f> m_liveNormalsOpenCV;

    std::shared_ptr<OptImage> m_canonicalVertices;
    std::shared_ptr<OptImage> m_canonicalNormals;
    std::shared_ptr<OptImage> m_liveVertices;
    std::shared_ptr<OptImage> m_liveNormals;

    std::vector<cv::Vec3f> m_results;

    std::shared_ptr<OptGraph> m_dataGraph;
    std::shared_ptr<OptGraph> m_regGraph;

    std::shared_ptr<OptImage> m_rotation;
    std::shared_ptr<OptImage> m_translation;

    std::shared_ptr<OptImage> m_transformationWeights;

    std::vector<unsigned int> m_dims;  // curent index in the solver

    CombinedSolverParameters m_solverParameters;

    float m_functionTolerance;
    float m_paramTolerance;
    float m_trust_region_radius;
};
