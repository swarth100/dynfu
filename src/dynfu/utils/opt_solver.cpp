#include <dynfu/utils/opt_solver.hpp>

CombinedSolver::CombinedSolver(Warpfield warpfield, CombinedSolverParameters params, float tukeyOffset, float psi_data,
                               float lambda, float psi_reg) {
    m_warpfield                = warpfield;
    m_combinedSolverParameters = params;

    this->tukeyOffset = tukeyOffset;
    this->psi_data    = psi_data;

    this->lambda  = lambda;
    this->psi_reg = psi_reg;
}

void CombinedSolver::initializeProblemInstance(const std::shared_ptr<dynfu::Frame> canonicalFrame,
                                               const std::shared_ptr<dynfu::Frame> liveFrame) {
    m_canonicalVerticesPCL = canonicalFrame->getVertices();
    m_canonicalNormalsPCL  = canonicalFrame->getNormals();
    m_liveVerticesPCL      = liveFrame->getVertices();
    m_liveNormalsPCL       = liveFrame->getNormals();

    unsigned int D = m_warpfield.getNodes().size();
    unsigned int N = m_canonicalVerticesPCL.size();

    std::cout << "no. of non-zero canonical vertices: " << N << std::endl;

    m_dims = {D, N};

    /* set per-term regularisation weight */
    w_reg = sqrt(lambda / (D * KNN));

    m_canonicalVertices = createEmptyOptImage({N}, OptImage::Type::FLOAT, 3, OptImage::GPU, false);
    m_canonicalNormals  = createEmptyOptImage({N}, OptImage::Type::FLOAT, 3, OptImage::GPU, false);
    m_liveVertices      = createEmptyOptImage({N}, OptImage::Type::FLOAT, 3, OptImage::GPU, false);
    m_liveNormals       = createEmptyOptImage({N}, OptImage::Type::FLOAT, 3, OptImage::GPU, false);

    m_dg_v         = createEmptyOptImage({D}, OptImage::Type::FLOAT, 3, OptImage::GPU, false);
    m_translations = createEmptyOptImage({D}, OptImage::Type::FLOAT, 3, OptImage::GPU, true);
    m_rotations    = createEmptyOptImage({D}, OptImage::Type::FLOAT, 3, OptImage::GPU, true);
    m_dg_w         = createEmptyOptImage({D}, OptImage::Type::FLOAT, 1, OptImage::GPU, false);

    m_tukeyBiweights = createEmptyOptImage({N}, OptImage::Type::FLOAT, 1, OptImage::GPU, false);
    m_huberWeights   = createEmptyOptImage({D}, OptImage::Type::FLOAT, 1, OptImage::GPU, false);

    resetGPUMemory();

    updateTukeyBiweights();
    updateHuberWeights();

    initializeDataGraph();
    initializeRegGraph();

    addOptSolvers(m_dims, std::string(TOSTRING(TERRA_SOLVER_FILE)));
}

void CombinedSolver::initializeDataGraph() {
    unsigned int N = m_dims[1];
    std::vector<std::vector<int>> indices(KNN + 1, std::vector<int>(N));

    for (int count = 0; count < N; count++) {
        indices[0][count] = count;

        auto vertexNeighboursIdx = m_warpfield.findNeighborsIndex(KNN, m_canonicalVerticesPCL[count]);
        for (int i = 1; i < KNN + 1; i++) {
            indices[i][count] = vertexNeighboursIdx[i - 1];
        }
    }

    m_dataGraph = std::make_shared<OptGraph>(indices);

    std::cout << "initialized data graph" << std::endl;
}

void CombinedSolver::initializeRegGraph() {
    auto kdTree = m_warpfield.getKdTree();

    /* not used, ignores the distance to the nodes for now */
    std::vector<float> outDistSqr(8);
    std::vector<size_t> retIndex(8);

    unsigned int D = m_dims[0];
    std::vector<std::vector<int>> indices(KNN + 1, std::vector<int>(D));

    int i = 0;
    for (auto node : m_warpfield.getNodes()) {
        pcl::PointXYZ dg_v       = node->getPosition();
        std::vector<float> query = {dg_v.x, dg_v.y, dg_v.z};
        int n                    = kdTree->knnSearch(&query[0], KNN, &retIndex[0], &outDistSqr[0]);
        retIndex.resize(n);

        indices[0][i] = i;

        int j = 1;
        for (auto idx : retIndex) {
            indices[j][i] = retIndex[j - 1];
            j++;
        }

        i++;
    }

    m_regGraph = std::make_shared<OptGraph>(indices);

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
    m_problemParams.set("tukeyBiweights", m_tukeyBiweights);

    m_problemParams.set("regGraph", m_regGraph);
    m_problemParams.set("huberWeights", m_huberWeights);

    m_problemParams.set("w_reg", &w_reg);
}

void CombinedSolver::preSingleSolve() {}

void CombinedSolver::postSingleSolve() { copyResultToCPUFromFloat3(); }

void CombinedSolver::preNonlinearSolve(int /* iteration */) {
    copyResultToCPUFromFloat3();

    updateTukeyBiweights();
    updateHuberWeights();
}

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
    std::vector<float3> h_rotations(D);
    std::vector<float> h_dg_w(D);

    for (int i = 0; i < D; i++) {
        auto node = m_warpfield.getNodes()[i];

        pcl::PointXYZ nodeCoordinates = node->getPosition();
        h_dg_v[i]                     = make_float3(nodeCoordinates.x, nodeCoordinates.y, nodeCoordinates.z);

        /* TODO (dig15): pass in data from icp */
        h_translations[i] = make_float3(0.f, 0.f, 0.f);
        h_rotations[i]    = make_float3(0.f, 0.f, 0.f);

        h_dg_w[i] = node->getRadialBasisWeight();
    }

    m_dg_v->update(h_dg_v);
    m_translations->update(h_translations);
    m_rotations->update(h_rotations);
    m_dg_w->update(h_dg_w);
}

float CombinedSolver::calcTukeyBiweight(float tukeyOffset, float c, pcl::PointXYZ ptError) {
    auto ptErrorDistScaled = sqrt(ptError.x * ptError.x + ptError.y * ptError.y + ptError.z * ptError.z) / tukeyOffset;

    if (ptErrorDistScaled < c) {
        return pow(1.f - pow(ptErrorDistScaled, 2) / pow(c, 2), 2);
    }

    return 0.f;
}

void CombinedSolver::updateTukeyBiweights() {
    auto N = m_dims[1];
    std::vector<float> h_tukeyBiweights(N);

    for (unsigned int i = 0; i < N; i++) {
        auto canonicalVertex = m_canonicalVerticesPCL[i];
        auto liveVertex      = m_liveVerticesPCL[i];

        auto dqBlend                    = m_warpfield.calcDQB(canonicalVertex);
        auto canonicalVertexTransformed = dqBlend->transformVertex(canonicalVertex);
        pcl::PointXYZ ptError(liveVertex.x - canonicalVertexTransformed.x, liveVertex.y - canonicalVertexTransformed.y,
                              liveVertex.z - canonicalVertexTransformed.z);

        h_tukeyBiweights[i] = calcTukeyBiweight(tukeyOffset, psi_data, ptError);
    }

    m_tukeyBiweights->update(h_tukeyBiweights);
}

float CombinedSolver::calcHuberWeight(float k, float e) {
    if (abs(e) <= k) {
        return 1.f;
    }

    return k / abs(e);
}

void CombinedSolver::updateHuberWeights() {
    auto D                = m_dims[0];
    auto deformationNodes = m_warpfield.getNodes();

    std::vector<float> h_huberWeights(D);

    for (unsigned int i = 0; i < D; i++) {
        auto nodeNeighboursIdx = m_warpfield.findNeighborsIndex(KNN, deformationNodes[i]->getPosition());

        for (auto neighbourIdx : nodeNeighboursIdx) {
            auto neighbourCoordinates = deformationNodes[neighbourIdx]->getPosition();

            auto neighbourTransformed1 =
                deformationNodes[i]->getTransformation()->transformVertex(neighbourCoordinates);
            auto neighbourTransformed2 =
                deformationNodes[neighbourIdx]->getTransformation()->transformVertex(neighbourCoordinates);

            pcl::PointXYZ ptError(neighbourTransformed1.x - neighbourTransformed2.x,
                                  neighbourTransformed1.y - neighbourTransformed2.y,
                                  neighbourTransformed1.z - neighbourTransformed2.z);
            float e = sqrt(ptError.x * ptError.x + ptError.y * ptError.y + ptError.z * ptError.z);

            h_huberWeights[i] = calcHuberWeight(psi_reg, e);
        }
    }

    m_huberWeights->update(h_huberWeights);
}

void CombinedSolver::copyResultToCPUFromFloat3() {
    auto D = m_dims[0];

    std::vector<float3> h_translations(D);
    std::vector<float3> h_rotations(D);

    m_translations->copyTo(h_translations);
    m_rotations->copyTo(h_rotations);

    for (unsigned int i = 0; i < D; i++) {
        std::shared_ptr<DualQuaternion<float>> dq = std::make_shared<DualQuaternion<float>>(
            0, 0, 0, h_translations[i].x, h_translations[i].y, h_translations[i].z);

        m_warpfield.getNodes()[i]->updateTransformation(dq);
    }
}
