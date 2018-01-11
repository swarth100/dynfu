#include <dynfu/utils/opt_solver.hpp>

CombinedSolver::CombinedSolver(Warpfield warpfield, CombinedSolverParameters params) {
    m_warpfield                = warpfield;
    m_combinedSolverParameters = params;
}

void CombinedSolver::initializeProblemInstance(const std::shared_ptr<dynfu::Frame> canonicalFrame,
                                               const std::shared_ptr<dynfu::Frame> liveFrame) {
    m_canonicalVerticesOpenCV = canonicalFrame->getVertices();
    m_canonicalNormalsOpenCV  = canonicalFrame->getNormals();
    m_liveVerticesOpenCV      = liveFrame->getVertices();
    m_liveNormalsOpenCV       = liveFrame->getNormals();

    unsigned int D = m_warpfield.getNodes().size();

    /* number of non-zero vertices */
    unsigned int N = 0;

    for (int i = 0; i < m_canonicalVerticesOpenCV.size(); i++) {
        auto vertex     = m_canonicalVerticesOpenCV[i];
        auto normal     = m_canonicalNormalsOpenCV[i];
        auto vertexLive = m_canonicalVerticesOpenCV[i];
        auto normalLive = m_canonicalNormalsOpenCV[i];

        if (cv::norm(vertex) != 0 && cv::norm(normal) != 0 && cv::norm(vertexLive) != 0 && cv::norm(normalLive) != 0) {
            N++;
        }
    }

    std::cout << "no. of non-zero canonical vertices: " << N << std::endl;

    m_dims = {D, N};

    m_canonicalVertices = createEmptyOptImage({N}, OptImage::Type::FLOAT, 3, OptImage::GPU, true);
    m_canonicalNormals  = createEmptyOptImage({N}, OptImage::Type::FLOAT, 3, OptImage::GPU, true);
    m_liveVertices      = createEmptyOptImage({N}, OptImage::Type::FLOAT, 3, OptImage::GPU, true);
    m_liveNormals       = createEmptyOptImage({N}, OptImage::Type::FLOAT, 3, OptImage::GPU, true);

    m_nodeCoordinates = createEmptyOptImage({D}, OptImage::Type::FLOAT, 3, OptImage::GPU, true);

    m_rotation           = createEmptyOptImage({D}, OptImage::Type::FLOAT, 3, OptImage::GPU, true);
    m_translation        = createEmptyOptImage({D}, OptImage::Type::FLOAT, 3, OptImage::GPU, true);
    m_radialBasisWeights = createEmptyOptImage({D}, OptImage::Type::FLOAT, 1, OptImage::GPU, true);

    resetGPUMemory();

    initializeConnectivity();

    addOptSolvers(m_dims, std::string(TOSTRING(TERRA_SOLVER_FILE)));
}

void CombinedSolver::initializeConnectivity() {
    unsigned int N                   = m_dims[1];
    unsigned int noCanonicalVertices = m_canonicalVerticesOpenCV.size();

    std::vector<std::vector<int>> indices(9, std::vector<int>(N));

    int k = 0;
    for (int count = 0; count < noCanonicalVertices; count++) {
        if (cv::norm(m_canonicalVerticesOpenCV[count]) != 0 && cv::norm(m_canonicalNormalsOpenCV[count]) != 0 &&
            cv::norm(m_liveVerticesOpenCV[count]) != 0 && cv::norm(m_liveNormalsOpenCV[count]) != 0) {
            indices[0].push_back(k);

            auto vertexNeighbours    = m_warpfield.findNeighbors(KNN, m_canonicalVerticesOpenCV[count]);
            auto vertexNeighboursIdx = m_warpfield.findNeighborsIndex(KNN, m_canonicalVerticesOpenCV[count]);

            for (int i = 1; i < indices.size(); i++) {
                indices[i].push_back(vertexNeighboursIdx[i - 1]);
            }

            k++;
        }
    }

    m_dataGraph = std::make_shared<OptGraph>(indices);

    std::cout << "initialized connectivity" << std::endl;
}

void CombinedSolver::combinedSolveInit() {
    m_solverParams.set("nIterations", &m_combinedSolverParameters.nonLinearIter);
    m_solverParams.set("lIterations", &m_combinedSolverParameters.linearIter);

    m_problemParams.set("nodeCoordinates", m_nodeCoordinates);

    m_problemParams.set("rotation", m_rotation);
    m_problemParams.set("translation", m_translation);
    m_problemParams.set("radialBasisWeights", m_radialBasisWeights);

    m_problemParams.set("canonicalVertices", m_canonicalVertices);
    m_problemParams.set("canonicalNormals", m_canonicalNormals);

    m_problemParams.set("liveVertices", m_liveVertices);
    m_problemParams.set("liveNormals", m_liveNormals);

    m_problemParams.set("dataGraph", m_dataGraph);
}

void CombinedSolver::preSingleSolve() {}

void CombinedSolver::postSingleSolve() { copyResultToCPUFromFloat3(); }

void CombinedSolver::preNonlinearSolve(int /* iteration */) {}

void CombinedSolver::postNonlinearSolve(int /* iteration */) {}

void CombinedSolver::combinedSolveFinalize() {
    reportFinalCosts("warp field optimisation", m_combinedSolverParameters, getCost("Opt(GN)"), getCost("Opt(LM)"),
                     nan(""));
}

void CombinedSolver::resetGPUMemory() {
    unsigned int N                   = m_dims[1];
    unsigned int noCanonicalVertices = m_canonicalNormalsOpenCV.size();

    std::vector<float3> h_canonicalVertices(N);
    std::vector<float3> h_canonicalNormals(N);

    std::vector<float3> h_liveVertices(N);
    std::vector<float3> h_liveNormals(N);

    int k = 0;
    for (int i = 0; i < noCanonicalVertices; i++) {
        if (cv::norm(m_canonicalVerticesOpenCV[i]) != 0 && cv::norm(m_canonicalNormalsOpenCV[i]) != 0 &&
            cv::norm(m_liveVerticesOpenCV[i]) != 0 && cv::norm(m_liveNormalsOpenCV[i]) != 0) {
            h_canonicalVertices[k] = make_float3(m_canonicalVerticesOpenCV[i][0], m_canonicalVerticesOpenCV[i][1],
                                                 m_canonicalVerticesOpenCV[i][2]);
            h_canonicalNormals[k]  = make_float3(m_canonicalNormalsOpenCV[i][0], m_canonicalNormalsOpenCV[i][1],
                                                m_canonicalNormalsOpenCV[i][2]);

            h_liveVertices[k] =
                make_float3(m_liveVerticesOpenCV[i][0], m_liveVerticesOpenCV[i][1], m_liveVerticesOpenCV[i][2]);
            h_liveNormals[k] =
                make_float3(m_liveNormalsOpenCV[i][0], m_liveNormalsOpenCV[i][1], m_liveNormalsOpenCV[i][2]);
            k++;
        }
    }

    m_canonicalVertices->update(h_canonicalVertices);
    m_canonicalNormals->update(h_canonicalNormals);

    m_liveVertices->update(h_liveVertices);
    m_liveNormals->update(h_liveNormals);

    auto D = m_dims[0];

    std::vector<float3> h_coordinates(D);

    std::vector<float3> h_translation(D);
    std::vector<float3> h_rotation(D);
    std::vector<float> h_radialBasisWeight(D);

    for (int i = 0; i < D; i++) {
        auto nodeCoordinates = m_warpfield.getNodes()[i]->getPosition();

        auto nodeTransformation = m_warpfield.getNodes()[i]->getTransformation();

        auto nodeTranslation       = nodeTransformation->getTranslation();
        auto nodeRotation          = nodeTransformation->getRotation();
        auto nodeRadialBasisWeight = m_warpfield.getNodes()[i]->getRadialBasisWeight();

        h_coordinates[i] = make_float3(nodeCoordinates[0], nodeCoordinates[1], nodeCoordinates[2]);

        h_translation[i]       = make_float3(nodeTranslation[0], nodeTranslation[1], nodeTranslation[2]);
        h_rotation[i]          = make_float3(0.f, 0.f, 0.f);  // FIXME (dig15): set the rotations
        h_radialBasisWeight[i] = nodeRadialBasisWeight;
    }

    m_nodeCoordinates->update(h_coordinates);

    m_translation->update(h_translation);
    m_rotation->update(h_rotation);
    m_radialBasisWeights->update(h_radialBasisWeight);
}

void CombinedSolver::copyResultToCPUFromFloat3() {
    auto D = m_dims[0];

    std::vector<float3> h_translation(D);
    std::vector<float> h_radialBasisWeights(D);

    m_translation->copyTo(h_translation);
    m_radialBasisWeights->copyTo(h_radialBasisWeights);

    for (unsigned int i = 0; i < D; i++) {
        m_warpfield.getNodes()[i]->updateTranslation(
            cv::Vec3f(h_translation[i].x, h_translation[i].y, h_translation[i].z));
        m_warpfield.getNodes()[i]->setRadialBasisWeight(h_radialBasisWeights[i]);
    }
}
