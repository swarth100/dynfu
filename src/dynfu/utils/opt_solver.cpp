#include <dynfu/utils/opt_solver.hpp>

CombinedSolver::CombinedSolver(Warpfield warpfield, CombinedSolverParameters params) {
    m_warpfield        = warpfield;
    m_solverParameters = params;
}

void CombinedSolver::initializeProblemInstance(std::shared_ptr<Frame> canonicalFrame,
                                               std::shared_ptr<Frame> liveFrame) {
    m_canonicalVerticesOpenCV = canonicalFrame->getVertices();
    m_canonicalNormalsOpenCV  = canonicalFrame->getNormals();
    m_liveVerticesOpenCV      = liveFrame->getVertices();
    m_liveNormalsOpenCV       = liveFrame->getNormals();

    unsigned int D = m_warpfield.getNodes().size();
    unsigned int N = m_canonicalVerticesOpenCV.size();

    m_dims = {D, N};

    m_canonicalVertices = createEmptyOptImage({N}, OptImage::Type::FLOAT, 3, OptImage::GPU, true);
    m_canonicalNormals  = createEmptyOptImage({N}, OptImage::Type::FLOAT, 3, OptImage::GPU, true);
    m_liveVertices      = createEmptyOptImage({N}, OptImage::Type::FLOAT, 3, OptImage::GPU, true);
    m_liveNormals       = createEmptyOptImage({N}, OptImage::Type::FLOAT, 3, OptImage::GPU, true);

    m_rotation    = createEmptyOptImage({D}, OptImage::Type::FLOAT, 3, OptImage::GPU, true);
    m_translation = createEmptyOptImage({D}, OptImage::Type::FLOAT, 3, OptImage::GPU, true);

    m_transformationWeights = createEmptyOptImage({N}, OptImage::Type::FLOAT, 8, OptImage::GPU, true);

    resetGPUMemory();

    initializeConnectivity(m_canonicalVerticesOpenCV);
    addOptSolvers(m_dims, std::string("include/dynfu/utils/terra/energy.t"));
}

void CombinedSolver::initializeConnectivity(const std::vector<cv::Vec3f> canonicalVertices) {
    unsigned int N = (unsigned int) canonicalVertices.size();

    std::vector<std::vector<int>> graphVector(8 /* KNN */ + 1, std::vector<int>(N));
    std::vector<float[8]> weights(N);

    for (int count = 0; count < canonicalVertices.size(); count++) {
        graphVector[0].push_back(count);

        for (int i = 1; i < graphVector.size() - 3; i++) {
            graphVector[i].push_back(1);
        }

        // m_transformationWeights->update(weights);
        m_dataGraph = std::make_shared<OptGraph>(graphVector);
    }
}

void CombinedSolver::combinedSolveInit() {
    m_functionTolerance = 1e-5f;
    m_paramTolerance    = 1e-5f;

    m_problemParams.set("canonicalVertices", m_canonicalVertices);
    m_problemParams.set("canonicalNormals", m_canonicalNormals);

    m_problemParams.set("liveVertices", m_liveVertices);
    m_problemParams.set("liveNormals", m_liveNormals);

    m_problemParams.set("dataGraph", m_dataGraph);
    //        m_problemParams.set("regGraph", m_regGraph);

    m_problemParams.set("rotation", m_rotation);
    m_problemParams.set("translation", m_translation);

    m_problemParams.set("transformationWeights", m_transformationWeights);

    m_solverParams.set("lIterations", &m_combinedSolverParameters.linearIter);
    m_solverParams.set("nIterations", &m_combinedSolverParameters.nonLinearIter);

    m_solverParams.set("function_tolerance", &m_functionTolerance);
}

void CombinedSolver::preSingleSolve() {}

void CombinedSolver::postSingleSolve() { copyResultToCPUFromFloat3(); }

void CombinedSolver::preNonlinearSolve(int) {}

void CombinedSolver::postNonlinearSolve(int) {}

void CombinedSolver::combinedSolveFinalize() {
    reportFinalCosts("Robust Mesh Deformation", m_combinedSolverParameters, getCost("Opt(GN)"), getCost("Opt(LM)"),
                     nan(""));
}

void CombinedSolver::resetGPUMemory() {
    uint N = (uint) m_canonicalVerticesOpenCV.size();

    std::vector<float3> h_canonicalVertices(N);
    std::vector<float3> h_canonicalNormals(N);

    std::vector<float3> h_liveVertices(N);
    std::vector<float3> h_liveNormals(N);

    for (int i = 0; i < N; i++) {
        if (std::isnan(m_canonicalVerticesOpenCV[i][0]) || std::isnan(m_canonicalVerticesOpenCV[i][1]) ||
            std::isnan(m_canonicalVerticesOpenCV[i][2]))
            continue;

        if (std::isnan(m_canonicalNormalsOpenCV[i][0]) || std::isnan(m_canonicalNormalsOpenCV[i][1]) ||
            std::isnan(m_canonicalNormalsOpenCV[i][2]))
            continue;

        if (std::isnan(m_liveVerticesOpenCV[i][0]) || std::isnan(m_liveVerticesOpenCV[i][1]) ||
            std::isnan(m_liveVerticesOpenCV[i][2]))
            continue;

        if (std::isnan(m_liveNormalsOpenCV[i][0]) || std::isnan(m_liveNormalsOpenCV[i][1]) ||
            std::isnan(m_liveNormalsOpenCV[i][2]))
            continue;

        h_canonicalVertices[i] = make_float3(m_canonicalVerticesOpenCV[i][0], m_canonicalVerticesOpenCV[i][1],
                                             m_canonicalVerticesOpenCV[i][2]);
        h_canonicalNormals[i] =
            make_float3(m_canonicalNormalsOpenCV[i][0], m_canonicalNormalsOpenCV[i][1], m_canonicalNormalsOpenCV[i][2]);

        h_liveVertices[i] =
            make_float3(m_liveVerticesOpenCV[i][0], m_liveVerticesOpenCV[i][1], m_liveVerticesOpenCV[i][2]);
        h_liveNormals[i] = make_float3(m_liveNormalsOpenCV[i][0], m_liveNormalsOpenCV[i][1], m_liveNormalsOpenCV[i][2]);
    }

    m_canonicalVertices->update(h_canonicalVertices);
    m_canonicalNormals->update(h_canonicalNormals);

    m_liveVertices->update(h_liveVertices);
    m_liveNormals->update(h_liveNormals);

    uint D = (uint) m_warpfield.getNodes().size();

    std::vector<float3> h_translation(D);
    std::vector<float3> h_rotation(D);

    for (int i = 0; i < m_warpfield.getNodes().size(); i++) {
        auto nodeTransformation = m_warpfield.getNodes()[i]->getTransformation();

        auto nodeTranslation = nodeTransformation->getTranslation();
        h_translation[i]     = make_float3(nodeTranslation[0], nodeTranslation[1], nodeTranslation[2]);

        auto nodeRotation = nodeTransformation->getRotation();
        h_rotation[i]     = make_float3(0.f, 0.f, 0.f);  // TODO (dig15): set this properly
    }

    m_rotation->update(h_rotation);
    m_translation->update(h_translation);
}

std::vector<cv::Vec3f> getResult() {}  // FIXME (dig15): return the result properly

void CombinedSolver::copyResultToCPUFromFloat3() {
    unsigned int N = (unsigned int) m_warpfield.getNodes().size();

    std::vector<float3> h_translation(N);

    m_translation->copyTo(h_translation);

    for (unsigned int i = 0; i < N; i++)
        m_warpfield.getNodes()[i]->setTranslation(
            cv::Vec3f(h_translation[i].x, h_translation[i].y, h_translation[i].z));
}
