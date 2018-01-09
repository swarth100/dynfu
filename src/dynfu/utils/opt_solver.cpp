#include <dynfu/utils/opt_solver.hpp>

CombinedSolver::CombinedSolver(Warpfield warpfield, CombinedSolverParameters params) {
    m_warpfield                = warpfield;
    m_combinedSolverParameters = params;
}

void CombinedSolver::initializeProblemInstance(const std::shared_ptr<dynfu::Frame> canonicalFrame,
                                               const std::shared_ptr<dynfu::Frame> liveFrame) {
    m_canonicalVerticesPCL = canonicalFrame->getVertices();
    m_canonicalNormalsPCL  = canonicalFrame->getNormals();
    m_liveVerticesPCL      = liveFrame->getVertices();
    m_liveNormalsPCL       = liveFrame->getNormals();

    unsigned int D = m_warpfield.getNodes().size();
    unsigned int N = m_canonicalVerticesPCL.size();

    /* TODO (dig15): figure out why this no. changes between frames */
    std::cout << "no. of non-zero canonical vertices: " << N << std::endl;

    m_dims = {D, N};

    m_canonicalVertices = createEmptyOptImage({N}, OptImage::Type::FLOAT, 3, OptImage::GPU, false);
    m_canonicalNormals  = createEmptyOptImage({N}, OptImage::Type::FLOAT, 3, OptImage::GPU, false);
    m_liveVertices      = createEmptyOptImage({N}, OptImage::Type::FLOAT, 3, OptImage::GPU, false);
    m_liveNormals       = createEmptyOptImage({N}, OptImage::Type::FLOAT, 3, OptImage::GPU, false);

    m_dg_v         = createEmptyOptImage({D}, OptImage::Type::FLOAT, 3, OptImage::GPU, false);
    m_translations = createEmptyOptImage({D}, OptImage::Type::FLOAT, 3, OptImage::GPU, true);
    m_rotations    = createEmptyOptImage({D}, OptImage::Type::FLOAT, 4, OptImage::GPU, true);
    m_dg_w         = createEmptyOptImage({D}, OptImage::Type::FLOAT, 1, OptImage::GPU, false);

    resetGPUMemory();

    initializeDataGraph();
    initializeRegGraph();

    addOptSolvers(m_dims, std::string(TOSTRING(TERRA_SOLVER_FILE)));
}

void CombinedSolver::initializeDataGraph() {
    unsigned int N = m_dims[1];
    std::vector<std::vector<int>> indices(9, std::vector<int>(N));

    for (int count = 0; count < N; count++) {
        indices[0][count] = count;

        auto vertexNeighboursIdx = m_warpfield.findNeighborsIndex(8, m_canonicalVerticesPCL[count]);
        for (int i = 1; i < 9; i++) {
            indices[i][count] = vertexNeighboursIdx[i - 1];
        }
    }

    m_dataGraph = std::make_shared<OptGraph>(indices);

    std::cout << "initialized data graph" << std::endl;
}

void CombinedSolver::initializeRegGraph() {
    // int knn = 4;
    //
    // auto kdTree = m_warpfield.getKdTree();
    //
    // /* not used, ignores the distance to the nodes for now */
    // std::vector<float> outDistSqr(4);
    // std::vector<size_t> retIndex(4);
    //
    // unsigned int D = m_dims[0];
    // std::vector<std::vector<int>> indices(5, std::vector<int>(D));
    //
    // int i = 0;
    // for (auto node : m_warpfield.getNodes()) {
    //     pcl::PointXYZ dg_v       = node->getPosition();
    //     std::vector<float> query = {dg_v.x, dg_v.y, dg_v.z};
    //     int n                    = kdTree->knnSearch(&query[0], knn, &retIndex[0], &outDistSqr[0]);
    //     retIndex.resize(n);
    //
    //     indices[0][i] = i;
    //
    //     int j = 1;
    //     for (auto idx : retIndex) {
    //         indices[i][j] = retIndex[j - 1];
    //         j++;
    //     }
    //
    //     i++;
    // }
    //
    // m_regGraph = std::make_shared<OptGraph>(indices);

    std::cout << "initialised regularisation graph" << std::endl;
}

void CombinedSolver::combinedSolveInit() {
    m_solverParams.set("nIterations", &m_combinedSolverParameters.nonLinearIter);
    m_solverParams.set("lIterations", &m_combinedSolverParameters.linearIter);

    m_problemParams.set("dg_v", m_dg_v);
    m_problemParams.set("translations", m_translations);
    m_problemParams.set("rotations", m_rotations);
    m_problemParams.set("dg_w", m_dg_w);

    m_problemParams.set("canonicalVertices", m_canonicalVertices);
    m_problemParams.set("canonicalNormals", m_canonicalNormals);

    m_problemParams.set("liveVertices", m_liveVertices);
    m_problemParams.set("liveNormals", m_liveNormals);

    m_problemParams.set("dataGraph", m_dataGraph);
    // m_problemParams.set("regGraph", m_regGraph);
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
    unsigned int N = m_dims[1];

    std::vector<float3> h_canonicalVertices(N);
    std::vector<float3> h_canonicalNormals(N);

    std::vector<float3> h_liveVertices(N);
    std::vector<float3> h_liveNormals(N);

    for (int i = 0; i < N; i++) {
        h_canonicalVertices[i] =
            make_float3(m_canonicalVerticesPCL[i].x, m_canonicalVerticesPCL[i].y, m_canonicalVerticesPCL[i].z);
        h_canonicalNormals[i] = make_float3(m_canonicalNormalsPCL[i].normal_x, m_canonicalNormalsPCL[i].normal_y,
                                            m_canonicalNormalsPCL[i].normal_z);

        h_liveVertices[i] = make_float3(m_liveVerticesPCL[i].x, m_liveVerticesPCL[i].y, m_liveVerticesPCL[i].z);
        h_liveNormals[i] =
            make_float3(m_liveNormalsPCL[i].normal_x, m_liveNormalsPCL[i].normal_y, m_liveNormalsPCL[i].normal_z);
    }

    m_canonicalVertices->update(h_canonicalVertices);
    m_canonicalNormals->update(h_canonicalNormals);

    m_liveVertices->update(h_liveVertices);
    m_liveNormals->update(h_liveNormals);

    auto D = m_dims[0];

    std::vector<float3> h_dg_v(D);
    std::vector<float3> h_translations(D);
    std::vector<float4> h_rotations(D);
    std::vector<float> h_dg_w(D);

    for (int i = 0; i < D; i++) {
        auto node = m_warpfield.getNodes()[i];

        auto nodeCoordinates = node->getPosition();
        h_dg_v[i]            = make_float3(nodeCoordinates.x, nodeCoordinates.y, nodeCoordinates.z);

        auto dg_se3          = node->getTransformation();
        auto nodeTranslation = dg_se3->getTranslation();
        auto nodeRotation    = dg_se3->getRotation();

        h_translations[i] = make_float3(nodeTranslation[0], nodeTranslation[1], nodeTranslation[2]);
        h_rotations[i]    = make_float4(nodeRotation.R_component_1(), nodeRotation.R_component_2(),
                                     nodeRotation.R_component_3(), nodeRotation.R_component_4());

        auto dg_w = node->getRadialBasisWeight();
        h_dg_w[i] = dg_w;
    }

    m_dg_v->update(h_dg_v);
    m_translations->update(h_translations);
    m_rotations->update(h_rotations);
    m_dg_w->update(h_dg_w);
}

void CombinedSolver::copyResultToCPUFromFloat3() {
    auto D = m_dims[0];

    std::vector<float3> h_translations(D);
    std::vector<float4> h_rotations(D);

    m_translations->copyTo(h_translations);
    m_rotations->copyTo(h_rotations);

    for (unsigned int i = 0; i < D; i++) {
        auto realWithUpdate = boost::math::quaternion<float>(1.f, 0.f, 0.f, 0.f);
        auto dualWithUpdate =
            boost::math::quaternion<float>(0, h_translations[i].x, h_translations[i].y, h_translations[i].z) * 0.5f;

        m_warpfield.getNodes()[i]->updateTransformation(realWithUpdate, dualWithUpdate);
    }
}
