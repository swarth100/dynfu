#include <dynfu/dyn_fusion.hpp>
/* pcl includes */
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
/* TODO: Add comment */
DynFusion::DynFusion() = default;

/* TODO: Add comment */
DynFusion::~DynFusion() = default;

/* initialise dynamicfusion with the initial vertices and normals */
void DynFusion::init(kfusion::cuda::Cloud &vertices, kfusion::cuda::Normals &normals) {
    global_counter    = 0;
    cv::Mat cloudHost = cloudToMat(vertices);
    std::vector<cv::Vec3f> canonicalVertices(cloudHost.rows * cloudHost.cols);
    std::cout << "no. of canonical vertices: " << cloudHost.cols * cloudHost.rows << std::endl;

    for (int y = 0; y < cloudHost.rows; ++y) {
        for (int x = 0; x < cloudHost.cols; ++x) {
            auto point = cloudHost.at<kfusion::Point>(y, x);
            if (!isNaN(point)) {
                canonicalVertices[x + cloudHost.cols * y] = cv::Vec3f(point.x, point.y, point.z);
            } else {
                canonicalVertices[x + cloudHost.cols * y] = cv::Vec3f(0.f, 0.f, 0.f);
            }
        }
    }

    cv::Mat normalHost = normalsToMat(normals);
    std::vector<cv::Vec3f> canonicalNormals(normalHost.rows * normalHost.cols);

    for (int y = 0; y < normalHost.rows; ++y) {
        for (int x = 0; x < normalHost.cols; ++x) {
            auto point = normalHost.at<kfusion::Normal>(y, x);

            if (!isNaN(point)) {
                canonicalNormals[x + normalHost.cols * y] = cv::Vec3f(point.x, point.y, point.z);
            } else {
                canonicalNormals[x + normalHost.cols * y] = cv::Vec3f(0.f, 0.f, 0.f);
            }
        }
    }

    initCanonicalFrame(canonicalVertices, canonicalNormals);

    int step                     = 50;
    auto &canonicalFrameVertices = canonicalFrame->getVertices();
    std::vector<std::shared_ptr<Node>> deformationNodes;
    for (int i = 0; i < canonicalFrameVertices.size(); i += step) {
        auto dq = std::make_shared<DualQuaternion<float>>(0.f, 0.f, 0.f, 0.f, 0.f, 0.f);
        deformationNodes.push_back(std::make_shared<Node>(canonicalFrameVertices[i], dq, 2.f));
    }

    //[> sample the deformation nodes <]
    // int noDeformationNodes = 8192;

    // std::vector<std::shared_ptr<Node>> deformationNodes;

    // int i = 1;
    // while (i <= noDeformationNodes) {
    // auto k = rand() % (canonicalVertices.size() - 1);

    // auto dq = std::make_shared<DualQuaternion<float>>(0.f, 0.f, 0.f, 0.f, 0.f, 0.f);
    // deformationNodes.push_back(std::make_shared<Node>(canonicalVertices[k], dq, 2.f));
    // i++;
    //}

    /* Initialise the warp field with the inital frames vertices */
    warpfield = std::make_shared<Warpfield>();
    warpfield->init(deformationNodes);

    std::cout << "initialised warpfield" << std::endl;
}

/* TODO: Add comment */
void DynFusion::initCanonicalFrame(std::vector<cv::Vec3f> &vertices, std::vector<cv::Vec3f> &normals) {
    this->canonicalFrame = std::make_shared<dynfu::Frame>(0, vertices, normals);
}

void DynFusion::setAffine(cv::Affine3f affine) { this->affine = affine; }

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
    int step = 5;
    for (int i = 0; i < unsupportedVertices.size(); i += step) {
        auto dg_se3 = warpfield->calcDQB(unsupportedVertices[i]);
        warpfield->addNode(std::make_shared<Node>(unsupportedVertices[i], dg_se3, 2.f));
    }

    std::cout << "finished updating the warpfield" << std::endl;
}

/* TODO: Add comment */
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
    auto canonicalWarped             = warpfield->warpToLive(canonicalFrame);
    auto canonicalNormals            = canonicalWarped->getNormals();
    auto canonicalVertices           = canonicalWarped->getVertices();
    auto liveFrameVertices           = liveFrame->getVertices();
    auto correspondingCanonicalFrame = findCorrespondingFrame(canonicalVertices, canonicalNormals, liveFrameVertices);
    savePointCloud(liveFrameVertices, "LiveFrame", global_counter);
    // savePointCloud(correspondingCanonicalFrame->getVertices(), "CanonicalCorresponding", global_counter);
    combinedSolver.initializeProblemInstance(correspondingCanonicalFrame, this->liveFrame);
    combinedSolver.solveAll();
    std::cout << "solved" << std::endl;

    canonicalWarpedToLive = warpfield->warpToLive(canonicalFrame);
    savePointCloud(canonicalWarpedToLive->getVertices(), "CanonicalWarpedToLive", global_counter);
    global_counter++;
}

std::shared_ptr<dynfu::Frame> DynFusion::findCorrespondingFrame(std::vector<cv::Vec3f> canonicalVertices,
                                                                std::vector<cv::Vec3f> canonicalNormals,
                                                                std::vector<cv::Vec3f> liveVertices) {
    /* Initialise KD-tree */
    auto kdCloud = std::make_shared<nanoflann::PointCloud>();
    kdCloud->pts = canonicalVertices;
    auto kdTree  = std::make_shared<kd_tree_t>(3, *kdCloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    kdTree->buildIndex();

    std::vector<cv::Vec3f> correspondingCanonicalVertices;
    std::vector<cv::Vec3f> correspondingCanonicalNormals;

    std::vector<float> outDistSqr(1);
    std::vector<size_t> retIndex(1);
    for (auto vertex : liveVertices) {
        if ((!vertex[0] && !vertex[1] && !vertex[2]) || std::isnan(vertex[0]) || std::isnan(vertex[1]) ||
            std::isnan(vertex[2])) {
            correspondingCanonicalVertices.push_back(vertex);
            correspondingCanonicalNormals.push_back(vertex);
        } else {
            std::vector<float> query = {vertex[0], vertex[1], vertex[2]};
            kdTree->knnSearch(&query[0], 1, &retIndex[0], &outDistSqr[0]);
            size_t index = retIndex[0];
            correspondingCanonicalVertices.push_back(canonicalVertices[index]);
            correspondingCanonicalNormals.push_back(canonicalNormals[index]);
        }
    }
    return std::make_shared<dynfu::Frame>(0, correspondingCanonicalVertices, correspondingCanonicalNormals);
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

static bool DynFusion::isNormalNaN(kfusion::Normal n) {
    if (std::isnan(n.x) || std::isnan(n.y) || std::isnan(n.z)) {
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
    normals.download(normalsHost.ptr<kfusion::Normal>(), normalsHost.step);
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

void DynFusion::savePointCloud(std::vector<cv::Vec3f> vertices, std::string filename, int i) {
    /* initialise the point cloud */
    auto cloud    = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    cloud->width  = vertices.size();
    cloud->height = 1;
    cloud->points.resize(cloud->width * cloud->height);

    /* iterate through vectors */
    for (size_t i = 0; i < vertices.size(); i++) {
        const cv::Vec3f &pt = vertices[i];
        cloud->points[i]    = pcl::PointXYZ(pt[0], pt[1], pt[2]);
    }

    /* save to PCL */
    std::string filenameStr = ("files/" + filename + std::to_string(i) + ".pcd");
    try {
        pcl::io::savePCDFileASCII(filenameStr, (*cloud));
    } catch (...) {
        std::cout << "Could not save to " + filenameStr << std::endl;
    }
}
