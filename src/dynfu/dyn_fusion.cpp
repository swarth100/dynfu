#include <dynfu/dyn_fusion.hpp>

#include <kfusion/internal.hpp>
#include <kfusion/precomp.hpp>

/* TODO: Add comment */
DynFusion::DynFusion(const kfusion::KinFuParams &params) : kfusion::KinFu::KinFu(params){};

/* TODO: Add comment */
DynFusion::~DynFusion() = default;

bool DynFusion::operator()(const kfusion::cuda::Depth &depth, const kfusion::cuda::Image & /*image*/) {
    std::cout << "frame no. " << frame_counter_ << std::endl;

    const kfusion::KinFuParams &p = params_;
    const int LEVELS              = icp_->getUsedLevelsNum();

    kfusion::cuda::computeDists(depth, dists_, p.intr);
    depthBilateralFilter(depth, curr_.depth_pyr[0], p.bilateral_kernel_size, p.bilateral_sigma_spatial,
                         p.bilateral_sigma_depth);

    if (p.icp_truncate_depth_dist > 0) {
        kfusion::cuda::depthTruncation(curr_.depth_pyr[0], p.icp_truncate_depth_dist);
    }

    for (int i = 1; i < LEVELS; ++i) {
        kfusion::cuda::depthBuildPyramid(curr_.depth_pyr[i - 1], curr_.depth_pyr[i], p.bilateral_sigma_depth);
    }

    for (int i = 0; i < LEVELS; ++i) {
#if defined USE_DEPTH
        kfusion::cuda::computeNormalsAndMaskDepth(p.intr(i), curr_.depth_pyr[i], curr_.normals_pyr[i]);
#else
        kfusion::cuda::computePointNormals(p.intr(i), curr_.depth_pyr[i], curr_.points_pyr[i], curr_.normals_pyr[i]);
#endif
    }

    kfusion::cuda::waitAllDefaultStream();

    // can't do more with the first frame
    if (frame_counter_ == 0) {
        volume_->integrate(dists_, poses_.back(), p.intr);

#if defined USE_DEPTH
        curr_.depth_pyr.swap(prev_.depth_pyr);
#else
        curr_.points_pyr.swap(prev_.points_pyr);
#endif
        curr_.normals_pyr.swap(prev_.normals_pyr);
        return ++frame_counter_, false;
    }

    /*
     * ITERATIVE CLOSET POINT
     */
    cv::Affine3f affine;  // current -> previous
    {
// ScopeTime time("icp");
#if defined USE_DEPTH
        bool ok = icp_->estimateTransform(affine, p.intr, curr_.depth_pyr, curr_.normals_pyr, prev_.depth_pyr,
                                          prev_.normals_pyr);
        updateAffine(affine);
#else
        bool ok = icp_->estimateTransform(affine, p.intr, curr_.points_pyr, curr_.normals_pyr, prev_.points_pyr,
                                          prev_.normals_pyr);
        updateAffine(affine);
#endif
        if (!ok) {
            return reset(), false;
        }
    }

    poses_.push_back(poses_.back() * affine);  // curr -> global

    /*
     * VOLUME INTEGRATION
     * we don't integrate volume if the camera doesn't move
     */
    float rnorm    = static_cast<float>(cv::norm(affine.rvec()));
    float tnorm    = static_cast<float>(cv::norm(affine.translation()));
    bool integrate = (rnorm + tnorm) / 2 >= p.tsdf_min_camera_movement;

    // if (integrate) {
    // ScopeTime time("tsdf");
    volume_->clear();
    volume_->integrate(dists_, poses_.back(), p.intr);
    //}

    /*
     * RAYCASTING
     */
    {
// ScopeTime time("ray-cast-all");
#if defined USE_DEPTH
        volume_->raycast(poses_.back(), p.intr, prev_.depth_pyr[0], prev_.normals_pyr[0]);
        for (int i = 1; i < LEVELS; ++i) {
            resizeDepthNormals(prev_.depth_pyr[i - 1], prev_.normals_pyr[i - 1], prev_.depth_pyr[i],
                               prev_.normals_pyr[i]);
        }
#else
        volume_->raycast(poses_.back(), p.intr, prev_.points_pyr[0], prev_.normals_pyr[0]);
        for (int i = 1; i < LEVELS; ++i) {
            resizePointsNormals(prev_.points_pyr[i - 1], prev_.normals_pyr[i - 1], prev_.points_pyr[i],
                                prev_.normals_pyr[i]);
        }
#endif

        kfusion::cuda::waitAllDefaultStream();
    }

    if (frame_counter_ == 1) {
        /* initialise the warpfield */
        init(prev_.points_pyr[0], prev_.normals_pyr[0]);

        return ++frame_counter_, false;
    }

    /* add new live frame to dynfu */
    addLiveFrame(frame_counter_, prev_.points_pyr[0], prev_.normals_pyr[0]);
    /* warp canonical frame to live frame */
    warpCanonicalToLiveOpt();
    /* get the canonical frame as warped to live */
    canonicalWarpedToLive = getCanonicalWarpedToLive();
    /* get the polygon mesh of the model */
    canonicalWarpedToLiveMesh = reconstructSurface();

    return ++frame_counter_, true;
}

/* initialise dynamicfusion with the initial vertices and normals */
void DynFusion::init(kfusion::cuda::Cloud &vertices, kfusion::cuda::Normals &normals) {
    cv::Mat cloudHost = cloudToMat(vertices);
    std::vector<cv::Vec3f> canonicalVertices(cloudHost.rows * cloudHost.cols);

    cv::Mat normalHost = normalsToMat(normals);
    std::vector<cv::Vec3f> canonicalNormals(normalHost.rows * normalHost.cols);

    /* assumes vertices and normals have the same size */
    for (int y = 0; y < cloudHost.rows; ++y) {
        for (int x = 0; x < cloudHost.cols; ++x) {
            auto ptVertex = cloudHost.at<kfusion::Point>(y, x);
            auto ptNormal = normalHost.at<kfusion::Normal>(y, x);

            if (!isNaN(ptVertex) && !isNaN(ptNormal)) {
                canonicalVertices[x + cloudHost.cols * y] = cv::Vec3f(ptVertex.x, ptVertex.y, ptVertex.z);
                canonicalNormals[x + normalHost.cols * y] = cv::Vec3f(ptNormal.x, ptNormal.y, ptNormal.z);

            } else {
                canonicalVertices[x + cloudHost.cols * y] = cv::Vec3f(0.f, 0.f, 0.f);
                canonicalNormals[x + normalHost.cols * y] = cv::Vec3f(0.f, 0.f, 0.f);
            }
        }
    }

    initCanonicalFrame(canonicalVertices, canonicalNormals);

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
void DynFusion::warpCanonicalToLive() {}

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

pcl::PolygonMesh DynFusion::reconstructSurface() {
    std::cout << "constructing a polygon mesh" << std::endl;

    auto vertices = canonicalWarpedToLive->getVertices();
    auto normals  = canonicalWarpedToLive->getNormals();

    /* FIXME (dig15): need to subsample the vertices not to run out of memory */
    std::vector<cv::Vec3f> verticesMesh;
    std::vector<cv::Vec3f> normalsMesh;

    for (int i = 0; i < vertices.size(); i += 25) {
        if (cv::norm(vertices[i]) != 0 && cv::norm(normals[i]) != 0) {
            verticesMesh.emplace_back(vertices[i]);
            normalsMesh.emplace_back(normals[i]);
        }
    }

    std::cout << "no. of non-zero vertices used to construct polygon mesh: " << verticesMesh.size() << std::endl;

    // init point clouds with vertices and normals
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudVertices(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::Normal>::Ptr cloudNormals(new pcl::PointCloud<pcl::Normal>);

    (*cloudVertices).width  = verticesMesh.size();
    (*cloudVertices).height = 1;
    (*cloudVertices).points.resize((*cloudVertices).width * (*cloudVertices).height);

    (*cloudNormals).width  = normalsMesh.size();
    (*cloudNormals).height = 1;
    (*cloudNormals).points.resize((*cloudNormals).width * (*cloudNormals).height);

    // iterate through vectors with vertices and normals
    for (size_t i = 0; i < verticesMesh.size(); i++) {
        const cv::Vec3f &ptVertices = verticesMesh[i];
        pcl::PointXYZ pointVertex   = pcl::PointXYZ(ptVertices[0], ptVertices[1], ptVertices[2]);
        (*cloudVertices).points[i]  = pointVertex;

        const cv::Vec3f &ptNormals = normalsMesh[i];
        pcl::Normal pointNormal    = pcl::Normal(ptNormals[0], ptNormals[1], ptNormals[2]);
        (*cloudNormals).points[i]  = pointNormal;
    }

    // put vertices and normals in one point cloud
    pcl::PointCloud<pcl::PointNormal>::Ptr cloudWithNormals(new pcl::PointCloud<pcl::PointNormal>);
    pcl::concatenateFields(*cloudVertices, *cloudNormals, *cloudWithNormals);

    // perform reconstruction via marching cubes
    pcl::MarchingCubes<pcl::PointNormal> *mc;
    mc = new pcl::MarchingCubesRBF<pcl::PointNormal>();
    mc->setInputCloud(cloudWithNormals);

    float iso_level                = 0.f;
    float extend_percentage        = 0.f;
    int grid_res                   = 50;
    float off_surface_displacement = 0.f;

    mc->setIsoLevel(iso_level);
    mc->setGridResolution(grid_res, grid_res, grid_res);
    mc->setPercentageExtendGrid(extend_percentage);

    pcl::PolygonMesh::Ptr triangles(new pcl::PolygonMesh);

    std::cout << "beginning marching cubes reconstruction" << std::endl;

    mc->reconstruct(*triangles);

    std::cout << triangles->polygons.size() << " triangles created" << std::endl;

    return *triangles;
}

std::shared_ptr<dynfu::Frame> DynFusion::getCanonicalWarpedToLive() { return canonicalWarpedToLive; }

pcl::PolygonMesh DynFusion::getCanonicalWarpedToLiveSurface() { return canonicalWarpedToLiveMesh; }

void DynFusion::renderCanonicalWarpedToLive(kfusion::cuda::Image &image, int flag) {
    const kfusion::KinFuParams &p = params_;
    image.create(p.rows, flag != 3 ? p.cols : p.cols * 2);
    auto vertices = matToCloud(vectorToMat(canonicalWarpedToLive->getVertices()));
    auto normals  = matToCloud(vectorToMat(canonicalWarpedToLive->getNormals()));

    if ((flag < 1 || flag > 3)) {
        kfusion::cuda::renderImage(vertices, normals, params_.intr, params_.light_pose, image);
    } else if (flag == 2) {
        kfusion::cuda::renderTangentColors(normals, image);
    } else /* if (flag == 3) */ {
        kfusion::device::DeviceArray2D<kfusion::RGB> i1(p.rows, p.cols, image.ptr(), image.step());
        kfusion::device::DeviceArray2D<kfusion::RGB> i2(p.rows, p.cols, image.ptr() + p.cols, image.step());

        kfusion::cuda::renderImage(vertices, normals, params_.intr, params_.light_pose, i1);
        kfusion::cuda::renderTangentColors(normals, i2);
    }
}

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
