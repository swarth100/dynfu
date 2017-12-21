#include <dynfu/dyn_fusion.hpp>

/* TODO: Add comment */
DynFusion::DynFusion() = default;

/* TODO: Add comment */
DynFusion::~DynFusion() = default;

/* initialise dynamicfusion with the initial vertices and normals */
void DynFusion::init(kfusion::cuda::Cloud &vertices, kfusion::cuda::Normals &normals) {
    cv::Mat cloudHost = cloudToMat(vertices);
    std::vector<cv::Vec3f> canonicalVertices(cloudHost.rows * cloudHost.cols);

    std::cout << "no. of canonical vertices: " << cloudHost.cols * cloudHost.rows << std::endl;

    for (int y = 0; y < cloudHost.rows; ++y) {
        for (int x = 0; x < cloudHost.cols; ++x) {
            auto point = cloudHost.at<kfusion::Point>(y, x);

            if (!isNaN(point)) {
                canonicalVertices[x + cloudHost.cols * y] = cv::Vec3f(point.x, point.y, point.z);
            }
        }
    }

    cv::Mat normalHost = cloudToMat(normals);
    std::vector<cv::Vec3f> canonicalNormals(normalHost.rows * normalHost.cols);

    std::cout << "no. of canonical normals: " << normalHost.cols * normalHost.rows << std::endl;

    for (int y = 0; y < normalHost.cols; ++y) {
        for (int x = 0; x < normalHost.rows; ++x) {
            auto point = normalHost.at<kfusion::Point>(x, y);

            if (!isNaN(point)) {
                canonicalNormals[x + cloudHost.cols * y] = cv::Vec3f(point.x, point.y, point.z);
            }
        }
    }

    initCanonicalFrame(canonicalVertices, canonicalNormals);

    /* sample the deformation nodes */
    int noDeformationNodes = 8192;

    std::vector<std::shared_ptr<Node>> deformationNodes;

    int i = 1;
    while (i <= noDeformationNodes) {
        auto k = rand() % (canonicalVertices.size() - 1);

        auto dq = std::make_shared<DualQuaternion<float>>(0.f, 0.f, 0.f, 0.f, 0.f, 0.f);
        deformationNodes.push_back(std::make_shared<Node>(canonicalVertices[k], dq, 2.f));
        i++;
    }

    /* Initialise the warp field with the inital frames vertices */
    warpfield = std::make_shared<Warpfield>();
    warpfield->init(deformationNodes);

    std::cout << "initialised warpfield" << std::endl;
}

/* TODO: Add comment */
void DynFusion::initCanonicalFrame(std::vector<cv::Vec3f> vertices, std::vector<cv::Vec3f> normals) {
    this->canonicalFrame = std::make_shared<dynfu::Frame>(0, vertices, normals);
}

void DynFusion::updateWarpfield() {
    std::vector<cv::Vec3f> unsupportedVertices;
    float min;

    for (auto vertex : canonicalFrame->getVertices()) {
        for (auto neighbour : warpfield->findNeighbors(KNN, vertex)) {
            min = cv::norm(vertex - neighbour->getPosition()) / neighbour->getRadialBasisWeight();

            if (min > 1) {
                unsupportedVertices.emplace_back(vertex);
                break;
            }
        }
    }

    std::cout << "no. of unsupported vertices: " << unsupportedVertices.size() << std::endl;

    /* TODO (dig15): sample the new deformation nodes in a more intelligent way */
    for (int i = 0; i < unsupportedVertices.size(); i += 5) {
        auto dg_se3 = warpfield->calcDQB(unsupportedVertices[i]);

        warpfield->addNode(std::make_shared<Node>(unsupportedVertices[i], dg_se3, 2.f));
    }

    std::cout << "finished updating the warpfield" << std::endl;
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
    warpProblem.optimiseWarpField(warpfield, this->canonicalFrame, this->liveFrame);

    auto parameters = warpProblem.getParameters();

    int i = 0;
    int j = 0;
    for (auto vertex : canonicalFrame->getVertices()) {
        cv::Vec3f totalTranslation;

        auto neighbourNodes = warpfield->findNeighbors(KNN, vertex);

        for (auto neighbour : neighbourNodes) {
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
    // updateWarpfield();

    CombinedSolverParameters params;
    params.numIter       = 20;
    params.nonLinearIter = 15;
    params.linearIter    = 250;
    params.useOpt        = false;
    params.useOptLM      = true;
    params.earlyOut      = true;

    std::cout << "solving" << std::endl;
    CombinedSolver combinedSolver(*warpfield, params);
    combinedSolver.initializeProblemInstance(this->canonicalFrame, this->liveFrame);
    combinedSolver.solveAll();

    std::cout << "solved" << std::endl;

    std::vector<cv::Vec3f> canonicalVerticesWarpedToLive;

    int n = 0;
    for (auto vertex : canonicalFrame->getVertices()) {
        if (cv::norm(vertex) == 0) {
            canonicalVerticesWarpedToLive.push_back(vertex);
        } else {
            auto transformation   = warpfield->calcDQB(vertex);
            auto totalTranslation = transformation->getTranslation();

            auto vertexWarpedToLive = vertex + totalTranslation;
            canonicalVerticesWarpedToLive.push_back(vertexWarpedToLive);
            n++;
        }
    }

    std::cout << "no. of non-zero canonical vertices warped to live: " << n << std::endl;

    std::cout << "set the canonical vertices warped to live" << std::endl;

    DynFusion::nextFrameReady = true;

    canonicalWarpedToLive =
        std::make_shared<dynfu::Frame>(0, canonicalVerticesWarpedToLive, canonicalVerticesWarpedToLive);
}

void DynFusion::addLiveFrame(int frameID, kfusion::cuda::Cloud &vertices, kfusion::cuda::Normals &normals) {
    auto liveFrameVertices = matToVector(cloudToMat(vertices));
    auto liveFrameNormals  = matToVector(normalsToMat(normals));

    liveFrame = std::make_shared<dynfu::Frame>(frameID, liveFrameVertices, liveFrameNormals);
}

std::shared_ptr<dynfu::Frame> DynFusion::getCanonicalWarpedToLive() { return canonicalWarpedToLive; }

/* define the static field */
bool DynFusion::nextFrameReady = false;

static bool DynFusion::isNaN(kfusion::Point pt) {
    if (std::isnan(pt.x) || std::isnan(pt.y) || std::isnan(pt.z)) {
        return true;
    }
    return false;
}

cv::Mat DynFusion::cloudToMat(kfusion::cuda::Cloud cloud) {
    cv::Mat cloudHost;
    cloudHost.create(cloud.rows(), cloud.cols(), CV_32FC4);
    cloud.download(cloudHost.ptr<kfusion::Point>(), cloudHost.step);
    return cloudHost;
}

kfusion::cuda::Cloud DynFusion::matToCloud(cv::Mat matrix) {
    kfusion::cuda::Cloud cloud;
    cloud.create(matrix.rows, matrix.cols);
    cloud.upload(matrix.data, matrix.step, matrix.rows, matrix.cols);
    return cloud;
}

cv::Mat DynFusion::depthToMat(kfusion::cuda::Depth depths) {
    cv::Mat depthHost;
    depthHost.create(depths.rows(), depths.cols(), CV_32FC1);
    depths.download(depthHost.ptr<kfusion::Point>(), depthHost.step);
    return depthHost;
}

kfusion::cuda::Depth DynFusion::matToDepth(cv::Mat matrix) {
    kfusion::cuda::Depth depths;
    depths.create(matrix.rows, matrix.cols);
    depths.upload(matrix.data, matrix.step, matrix.rows, matrix.cols);
    return depths;
}

cv::Mat DynFusion::normalsToMat(kfusion::cuda::Normals normals) {
    cv::Mat normalsHost;
    normalsHost.create(normals.rows(), normals.cols(), CV_32FC4);
    normals.download(normalsHost.ptr<kfusion::Point>(), normalsHost.step);
    return normalsHost;
}

kfusion::cuda::Normals DynFusion::matToNormals(cv::Mat matrix) {
    kfusion::cuda::Normals normals;
    normals.create(matrix.rows, matrix.cols);
    normals.upload(matrix.data, matrix.step, matrix.rows, matrix.cols);
    return normals;
}

std::vector<cv::Vec3f> DynFusion::matToVector(cv::Mat matrix) {
    std::vector<cv::Vec3f> vector(matrix.cols * matrix.rows);
    for (int y = 0; y < matrix.rows; ++y) {
        for (int x = 0; x < matrix.cols; ++x) {
            auto point = matrix.at<kfusion::Point>(y, x);
            if (!(std::isnan(point.x) || std::isnan(point.y) || std::isnan(point.z))) {
                vector[x + matrix.cols * y] = cv::Vec3f(point.x, point.y, point.z);
            }
        }
    }
    return vector;
}

cv::Mat DynFusion::vectorToMat(std::vector<cv::Vec3f> vec) {
    int colLen = 640;
    int rowLen = 480;
    cv::Mat mat(rowLen, colLen, CV_32FC4);
    for (int y = 0; y < rowLen; ++y) {
        for (int x = 0; x < colLen; ++x) {
            int index = x + y * colLen;
            kfusion::Point p;
            if (index < vec.size() && (vec[index][0] || vec[index][1] || vec[index][2])) {
                p = kfusion::Point({{vec[index][0], vec[index][1], vec[index][2]}});
            } else {
                p = kfusion::Point({{std::nan(""), std::nan(""), std::nan("")}});
            }
            mat.at<kfusion::Point>(y, x) = p;
        }
    }
    return mat;
}
