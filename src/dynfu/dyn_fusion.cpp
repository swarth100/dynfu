#include <dynfu/dyn_fusion.hpp>

/* TODO: Add comment */
DynFusion::DynFusion() = default;

/* TODO: Add comment */
DynFusion::~DynFusion() = default;

/* initialise dynamicfusion with the initial vertices and normals */
void DynFusion::init(kfusion::cuda::Cloud &vertices, kfusion::cuda::Normals &normals) {
    cv::Mat cloudHost = cloudToMat(vertices);
    std::vector<cv::Vec3f> canonicalVertices(cloudHost.rows * cloudHost.cols);

    cv::Mat normalHost = normalsToMat(normals);
    std::vector<cv::Vec3f> canonicalNormals(normalHost.rows * normalHost.cols);

    std::vector<cv::Vec3f> nonzeroCanonicalVertices;
    std::vector<cv::Vec3f> nonzeroCanonicalNormals;

    std::cout << "no. of canonical vertices: " << cloudHost.cols * cloudHost.rows << std::endl;

    /* assumes vertices and normals have the same size */
    for (int y = 0; y < cloudHost.rows; ++y) {
        for (int x = 0; x < cloudHost.cols; ++x) {
            auto ptVertex = cloudHost.at<kfusion::Point>(y, x);
            auto ptNormal = normalHost.at<kfusion::Normal>(y, x);

            if (!isNaN(ptVertex) && !isNaN(ptNormal)) {
                canonicalVertices[x + cloudHost.cols * y] = cv::Vec3f(ptVertex.x, ptVertex.y, ptVertex.z);
                canonicalNormals[x + normalHost.cols * y] = cv::Vec3f(ptNormal.x, ptNormal.y, ptNormal.z);

                if (!isZero(ptVertex) && !isZero(ptNormal)) {
                    nonzeroCanonicalVertices.emplace_back(cv::Vec3f(ptVertex.x, ptVertex.y, ptVertex.z));
                    nonzeroCanonicalNormals.emplace_back(cv::Vec3f(1.f, 1.f, 1.f));
                }

            } else {
                canonicalVertices[x + cloudHost.cols * y] = cv::Vec3f(0.f, 0.f, 0.f);
                canonicalNormals[x + normalHost.cols * y] = cv::Vec3f(0.f, 0.f, 0.f);
            }
        }
    }

    std::cout << "no. of non-zero vertices: " << nonzeroCanonicalVertices.size() << std::endl;

    initCanonicalFrame(canonicalVertices, canonicalNormals);
    initCanonicalMesh(nonzeroCanonicalVertices, nonzeroCanonicalNormals);

    /* TODO (dig15): implement better sampling of deformation nodes */
    auto &canonicalFrameVertices = canonicalFrame->getVertices();

    int step = 50;
    std::vector<std::shared_ptr<Node>> deformationNodes;

    for (int i = 0; i < canonicalFrameVertices.size(); i += step) {
        auto dq = std::make_shared<DualQuaternion<float>>(0.f, 0.f, 0.f, 0.f, 0.f, 0.f);
        deformationNodes.push_back(std::make_shared<Node>(canonicalFrameVertices[i], dq, 2.f));
    }

    /* initialise the warp field with the sampled deformation nodes */
    warpfield = std::make_shared<Warpfield>();
    warpfield->init(deformationNodes);

    std::cout << "initialised warpfield" << std::endl;
}

/* TODO: Add comment */
void DynFusion::initCanonicalFrame(std::vector<cv::Vec3f> &vertices, std::vector<cv::Vec3f> &normals) {
    this->canonicalFrame = std::make_shared<dynfu::Frame>(0, vertices, normals);
}

void DynFusion::initCanonicalMesh(std::vector<cv::Vec3f> &vertices, std::vector<cv::Vec3f> &normals) {
    std::cout << "constructing a polygon mesh" << std::endl;

    /* MARCHING CUBES RECONSTRUCTION */
    // init point clouds with vertices and normals
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudVertices(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::Normal>::Ptr cloudNormals(new pcl::PointCloud<pcl::Normal>);

    (*cloudVertices).width  = vertices.size();
    (*cloudVertices).height = 1;
    (*cloudVertices).points.resize((*cloudVertices).width * (*cloudVertices).height);

    (*cloudNormals).width  = normals.size();
    (*cloudNormals).height = 1;
    (*cloudNormals).points.resize((*cloudNormals).width * (*cloudNormals).height);

    // iterate through vectors with vertices and normals
    for (size_t i = 0; i < 8; i++) {
        const cv::Vec3f &ptVertices = vertices[i];
        pcl::PointXYZ pointVertex   = pcl::PointXYZ(ptVertices[0], ptVertices[1], ptVertices[2]);
        (*cloudVertices).points[i]  = pointVertex;

        const cv::Vec3f &ptNormals = normals[i];
        pcl::Normal pointNormal    = pcl::Normal(ptNormals[0], ptNormals[1], ptNormals[2]);
        (*cloudNormals).points[i]  = pointNormal;

        // std::cout << (*cloudVertices).points[i].x << " " << (*cloudVertices).points[i].y << " "
        //           << (*cloudVertices).points[i].z << std::endl;

        // std::cout << (*cloudNormals).points[i].data_c[0] << " " << (*cloudNormals).points[i].data_c[1] << " "
        //           << (*cloudNormals).points[i].data_c[2] << std::endl;
    }

    // put vertices and normals in one point cloud
    pcl::PointCloud<pcl::PointNormal>::Ptr cloudWithNormals(new pcl::PointCloud<pcl::PointNormal>);
    pcl::concatenateFields(*cloudVertices, *cloudNormals, *cloudWithNormals);

    std::cout << "begin marching cubes reconstruction" << std::endl;

    // perform reconstruction via marching cubes
    pcl::MarchingCubes<pcl::PointNormal> *mc;
    mc = new pcl::MarchingCubesRBF<pcl::PointNormal>();
    pcl::PolygonMesh::Ptr triangles(new pcl::PolygonMesh);

    mc->setInputCloud(cloudWithNormals);
    mc->reconstruct(*triangles);

    canonicalMesh = *triangles;

    // /* MINIMAL EXAMPLE */
    // pcl::PointCloud<pcl::PointXYZ>::Ptr cloudVertices(new pcl::PointCloud<pcl::PointXYZ>);
    //
    // (*cloudVertices).width  = 8;
    // (*cloudVertices).height = 1;
    // (*cloudVertices).points.resize((*cloudVertices).width * (*cloudVertices).height);
    //
    // for (size_t i = 0; i < 8; i++) {
    //     pcl::PointXYZ pointVertex  = pcl::PointXYZ(i + 1, i / 3, 7 * i);
    //     (*cloudVertices).points[i] = pointVertex;
    // }
    //
    // pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> ne;
    //
    // pcl::search::KdTree<pcl::PointXYZ>::Ptr tree1(new pcl::search::KdTree<pcl::PointXYZ>);
    // tree1->setInputCloud(cloudVertices);
    //
    // ne.setInputCloud(cloudVertices);
    // ne.setSearchMethod(tree1);
    // ne.setKSearch(20);
    //
    // pcl::PointCloud<pcl::Normal>::Ptr cloudNormals(new pcl::PointCloud<pcl::Normal>);
    // ne.compute(*cloudNormals);
    //
    // // for (size_t i = 0; i < 8; i++) {
    // //     std::cout << (*cloudNormals).points[i].data_c[0] << " " << (*cloudNormals).points[i].data_c[1] << " "
    // //               << (*cloudNormals).points[i].data_c[2] << std::endl;
    // //     std::cout << (*cloudVertices).points[i].x << " " << (*cloudVertices).points[i].y << " "
    // //               << (*cloudVertices).points[i].z << std::endl;
    // // }
    //
    // // Concatenate the XYZ and normal fields
    // pcl::PointCloud<pcl::PointNormal>::Ptr cloudWithNormals(new pcl::PointCloud<pcl::PointNormal>);
    // concatenateFields(*cloudVertices, *cloudNormals, *cloudWithNormals);
    //
    // // Create search tree
    // pcl::search::KdTree<pcl::PointNormal>::Ptr tree(new pcl::search::KdTree<pcl::PointNormal>);
    // tree->setInputCloud(cloudWithNormals);
    //
    // std::cout << "begin marching cubes reconstruction" << std::endl;
    //
    // // perform reconstruction via marching cubes
    // pcl::MarchingCubes<pcl::PointNormal> *mc;
    // mc = new pcl::MarchingCubesRBF<pcl::PointNormal>();
    // pcl::PolygonMesh::Ptr triangles(new pcl::PolygonMesh);
    //
    // mc->setInputCloud(cloudWithNormals);
    // mc->setSearchMethod(tree);
    // mc->reconstruct(*triangles);

    std::cout << triangles->polygons.size() << " triangles created" << std::endl;
}

void DynFusion::updateAffine(cv::Affine3f newAffine) { affineLiveToCanonical = affineLiveToCanonical * newAffine; }

cv::Affine3f DynFusion::getLiveToCanonicalAffine() { return affineLiveToCanonical; }

void DynFusion::updateWarpfield() {
    std::vector<cv::Vec3f> unsupportedVertices;

    for (auto vertex : canonicalFrame->getVertices()) {
        float currentDist = HUGE_VALF;
        float vertexMin   = currentDist;

        if (cv::norm(vertex) != 0) {
            for (auto neighbour : warpfield->findNeighbors(KNN, vertex)) {
                currentDist = cv::norm(vertex - neighbour->getPosition()) / neighbour->getRadialBasisWeight();

                if (currentDist < vertexMin) {
                    vertexMin = currentDist;
                }
            }

            if (vertexMin > 1) {
                unsupportedVertices.emplace_back(vertex);
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

    auto affineCanonicalToLive = affineLiveToCanonical.inv();

    auto canonicalWarped   = warpfield->warpToLive(affineCanonicalToLive, canonicalFrame);
    auto canonicalNormals  = canonicalWarped->getNormals();
    auto canonicalVertices = canonicalWarped->getVertices();
    auto liveFrameVertices = liveFrame->getVertices();

    auto correspondingCanonicalFrame = findCorrespondingFrame(canonicalVertices, canonicalNormals, liveFrameVertices);

    combinedSolver.initializeProblemInstance(correspondingCanonicalFrame, this->liveFrame);
    combinedSolver.solveAll();

    std::cout << "solved" << std::endl;

    canonicalWarpedToLive = warpfield->warpToLive(affineCanonicalToLive, canonicalFrame);
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

pcl::PolygonMesh DynFusion::getCanonicalMesh() { return canonicalMesh; }

std::shared_ptr<dynfu::Frame> DynFusion::getCanonicalWarpedToLive() { return canonicalWarpedToLive; }

/* define the static field */
bool DynFusion::nextFrameReady = false;

bool DynFusion::isNaN(kfusion::Point pt) { return (std::isnan(pt.x) || std::isnan(pt.y) || std::isnan(pt.z)); }

bool DynFusion::isZero(kfusion::Point pt) { return (cv::norm(cv::Vec3f(pt.x, pt.y, pt.z)) == 0); }

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

cv::Mat DynFusion::normalsToMat(kfusion::cuda::Normals normals) {
    cv::Mat normalsHost;
    normalsHost.create(normals.rows(), normals.cols(), CV_32FC4);
    normals.download(normalsHost.ptr<kfusion::Normal>(), normalsHost.step);
    return normalsHost;
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

            kfusion::Point point;

            if (index < vec.size() && (vec[index][0] || vec[index][1] || vec[index][2])) {
                point = kfusion::Point({{vec[index][0], vec[index][1], vec[index][2]}});
            } else {
                point = kfusion::Point({{static_cast<float>(std::nan("")), static_cast<float>(std::nan("")),
                                         static_cast<float>(std::nan(""))}});
            }

            mat.at<kfusion::Point>(y, x) = point;
        }
    }
    return mat;
}
