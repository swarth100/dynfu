#include <dynfu/dyn_fusion.hpp>

DynFusion::DynFusion() = default;

/* initialise dynamicfusion with the initals vertices and normals */
void DynFusion::init(kfusion::cuda::Cloud &vertices, kfusion::cuda::Cloud &normals) {
    cv::Mat cloudHost = cloudToMat(vertices);
    std::vector<cv::Vec3f> canonicalVertices(cloudHost.rows * cloudHost.cols);

    for (int y = 0; y < cloudHost.cols; ++y) {
        for (int x = 0; x < cloudHost.rows; ++x) {
            auto point = cloudHost.at<kfusion::Point>(x, y);
            if (!(std::isnan(point.x) || std::isnan(point.y) || std::isnan(point.z))) {
                canonicalVertices[x + cloudHost.rows * y] = cv::Vec3f(point.x, point.y, point.z);
            }
        }
    }

    cloudHost = cloudToMat(normals);
    std::vector<cv::Vec3f> canonicalNormals(cloudHost.rows * cloudHost.cols);

    for (int y = 0; y < cloudHost.cols; ++y) {
        for (int x = 0; x < cloudHost.rows; ++x) {
            auto point = cloudHost.at<kfusion::Point>(x, y);
            if (!(std::isnan(point.x) || std::isnan(point.y) || std::isnan(point.z))) {
                canonicalNormals[x + cloudHost.rows * y] = cv::Vec3f(point.x, point.y, point.z);
            }
        }
    }

    initCanonicalFrame(canonicalVertices, canonicalNormals);

    /* Sample the deformation nodes */
    int steps = 50;
    std::vector<std::shared_ptr<Node>> deformationNodes;
    for (int i = 0; i < canonicalVertices.size() - steps; i += steps) {
        auto dq = std::make_shared<DualQuaternion<float>>(0.f, 0.f, 0.f, 0.f, 0.f, 0.f);
        deformationNodes.push_back(std::make_shared<Node>(canonicalVertices[i], dq, 1.f));
    }
    /* Initialise the warp field with the inital frames vertices */
    warpfield = std::make_shared<Warpfield>();
    warpfield->init(deformationNodes);
}

/* TODO: Add comment */
DynFusion::~DynFusion() = default;

/* TODO: Add comment */
void DynFusion::initCanonicalFrame(std::vector<cv::Vec3f> vertices, std::vector<cv::Vec3f> /* normals */) {
    this->canonicalFrame = std::make_shared<dynfu::Frame>(0, vertices, vertices);
}

/* TODO: Add comment */
void DynFusion::warpCanonicalToLive() {
    ceres::Solver::Options options;
    options.linear_solver_type           = ceres::SPARSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations           = 64;

    int noCores         = sysconf(_SC_NPROCESSORS_ONLN);
    options.num_threads = noCores;

    WarpProblem warpProblem(options);
    warpProblem.optimiseWarpField(*warpfield, this->canonicalFrame, this->liveFrame);

    auto parameters = warpProblem.getParameters();

    int i = 0;
    int j = 0;
    for (auto vertex : canonicalFrame->getVertices()) {
        cv::Vec3f totalTranslation;

        for (auto neighbour : this->warpfield->getNodes()) {
            cv::Vec3f translation(parameters[i][1], parameters[i][2], parameters[i][3]);
            neighbour->setRadialBasisWeight(parameters[i][0]);
            neighbour->setTranslation(translation);

            totalTranslation += translation * neighbour->getTransformationWeight(vertex);
            i++;
        }
        i = 0;
        j++;
    }
}

void DynFusion::warpCanonicalToLiveOpt() {
    CombinedSolverParameters params;
    params.numIter       = 20;
    params.nonLinearIter = 15;
    params.linearIter    = 250;
    params.useOpt        = false;
    params.useOptLM      = true;
    params.earlyOut      = true;

    CombinedSolver combinedSolver(*warpfield, params);
    combinedSolver.initializeProblemInstance(this->canonicalFrame, this->liveFrame);
    combinedSolver.solveAll();

    int i = 0;
    int j = 0;
    for (auto vertex : canonicalFrame->getVertices()) {
        cv::Vec3f totalTranslation;

        auto neighbourNodes = warpfield->findNeighbors(8, vertex);

        for (auto neighbour : neighbourNodes) {
            cv::Vec3f translation = neighbour->getTransformation()->getTranslation();
            // totalTranslation += translation;
            i++;
        }

        i = 0;
        j++;
    }
}

void DynFusion::addLiveFrame(int frameID, kfusion::cuda::Cloud &vertices, kfusion::cuda::Normals &normals) {
    auto liveFrameVertices = matToVector(cloudToMat(vertices));
    auto liveFrameNormals  = matToVector(normalsToMat(normals));

    liveFrame = std::make_shared<dynfu::Frame>(frameID, liveFrameVertices, liveFrameNormals);
}

cv::Mat DynFusion::cloudToMat(kfusion::cuda::Cloud cloud) {
    cv::Mat cloudHost;
    cloudHost.create(cloud.rows(), cloud.cols(), CV_32FC4);
    cloud.download(cloudHost.ptr<kfusion::Point>(), cloudHost.step);
    return cloudHost;
}

cv::Mat DynFusion::normalsToMat(kfusion::cuda::Normals normals) {
    cv::Mat normalsHost;
    normalsHost.create(normals.rows(), normals.cols(), CV_32FC4);
    normals.download(normalsHost.ptr<kfusion::Point>(), normalsHost.step);
    return normalsHost;
}

std::vector<cv::Vec3f> DynFusion::matToVector(cv::Mat matrix) {
    std::vector<cv::Vec3f> vector(matrix.cols * matrix.rows);
    for (int y = 0; y < matrix.cols; ++y) {
        for (int x = 0; x < matrix.rows; ++x) {
            auto point = matrix.at<kfusion::Point>(x, y);
            if (!(std::isnan(point.x) || std::isnan(point.y) || std::isnan(point.z))) {
                vector[x + matrix.rows * y] = cv::Vec3f(point.x, point.y, point.z);
            }
        }
    }
    return vector;
}
