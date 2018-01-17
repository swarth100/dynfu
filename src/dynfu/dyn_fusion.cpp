#include <dynfu/dyn_fusion.hpp>

#include <kfusion/internal.hpp>
#include <kfusion/precomp.hpp>

DynFuParams DynFuParams::defaultParams() {
    DynFuParams p;

    kfusion::KinFuParams kinfuParams = kfusion::KinFuParams::default_params();
    kinfuParams.volume_dims          = cv::Vec3i::all(256);  // number of voxels
    p.kinfuParams                    = kinfuParams;

    p.tukeyOffset = 4.652;

    p.lambda   = 200;   // regularisation parameter
    p.psi_data = 0.01;  // parameter to calculate tukey biweights
    p.psi_reg  = 1e-4;  // parameter to calculate huber weights

    p.L    = 4;  // no. of levels int the regularisation hierarchy
    p.beta = 4;  // ???

    p.epsilon = 0.015;  // decimation density, mm

    return p;
};

DynFusion::DynFusion(const DynFuParams &params) : kfusion::KinFu::KinFu(params.kinfuParams) { dynfuParams = params; };

DynFusion::~DynFusion() = default;

DynFuParams &DynFusion::params() { return dynfuParams; }

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
        kfusion::cuda::computePointNormals(p.intr(i), curr_.depth_pyr[i], curr_.points_pyr[i], curr_.normals_pyr[i]);
    }

    kfusion::cuda::waitAllDefaultStream();

    // can't do more with the first frame
    if (frame_counter_ == 0) {
        volume_->integrate(dists_, poses_.back(), p.intr);

        curr_.points_pyr.swap(prev_.points_pyr);
        curr_.normals_pyr.swap(prev_.normals_pyr);

        /* TODO (dig15); figure out why this decreases the no. of vertices */
        volume_->raycast(poses_.back(), p.intr, prev_.points_pyr[0], prev_.normals_pyr[0]);

        /* initialise the warpfield */
        init(prev_.points_pyr[0], prev_.normals_pyr[0]);
        /* construct polygon mesh for the canonical frame */
        // reconstructSurface();

        return ++frame_counter_, false;
    }

    /*
     * ITERATIVE CLOSET POINT
     */
    cv::Affine3f affine;  // current -> previous
    {
        // ScopeTime time("icp");

        bool ok = icp_->estimateTransform(affine, p.intr, curr_.points_pyr, curr_.normals_pyr, prev_.points_pyr,
                                          prev_.normals_pyr);

        if (!ok) {
            return reset(), false;
        }
    }

    poses_.push_back(poses_.back() * affine);  // curr -> global

    /*
     * VOLUME INTEGRATION--don't integrate volume if the camera doesn't move
     */
    {
        float rnorm    = static_cast<float>(cv::norm(affine.rvec()));
        float tnorm    = static_cast<float>(cv::norm(affine.translation()));
        bool integrate = (rnorm + tnorm) / 2 >= p.tsdf_min_camera_movement;
        volume_->clear();
        volume_->integrate(dists_, poses_.back(), p.intr);
    }

    /*
     * RAYCASTING
     */
    {
        // ScopeTime time("ray-cast-all");

        volume_->raycast(poses_.back(), p.intr, prev_.points_pyr[0], prev_.normals_pyr[0]);

        for (int i = 1; i < LEVELS; ++i) {
            resizePointsNormals(prev_.points_pyr[i - 1], prev_.normals_pyr[i - 1], prev_.points_pyr[i],
                                prev_.normals_pyr[i]);
        }

        kfusion::cuda::waitAllDefaultStream();
    }

    /* TODO (dig15): pass in the depths, not points; add new live frame to dynfu */
    addLiveFrame(frame_counter_, prev_.points_pyr[0], prev_.normals_pyr[0]);
    /* warp canonical frame to live frame */
    warpCanonicalToLiveOpt();
    /* construct a polygon mesh of the warped model via marching cubes */
    // reconstructSurface();

    return ++frame_counter_, true;
}

void DynFusion::init(kfusion::cuda::Cloud &vertices, kfusion::cuda::Normals &normals) {
    cv::Mat cloudHost = cloudToMat(vertices);
    pcl::PointCloud<pcl::PointXYZ> canonicalVertices;

    cv::Mat normalHost = normalsToMat(normals);
    pcl::PointCloud<pcl::Normal> canonicalNormals;

    /* assumes vertices and normals have the same size */
    for (int y = 0; y < cloudHost.rows; ++y) {
        for (int x = 0; x < cloudHost.cols; ++x) {
            auto ptVertex = cloudHost.at<kfusion::Point>(y, x);
            auto ptNormal = normalHost.at<kfusion::Normal>(y, x);

            if (!isNaN(ptVertex) && !isNaN(ptNormal) && !isZero(ptVertex) && !isNaN(ptNormal)) {
                canonicalVertices.push_back(pcl::PointXYZ(ptVertex.x, ptVertex.y, ptVertex.z));
                canonicalNormals.push_back(pcl::Normal(ptNormal.x, ptNormal.y, ptNormal.z));
            }
        }
    }

    initCanonicalFrame(canonicalVertices, canonicalNormals);

    auto &canonicalFrameVertices = canonicalFrame->getVertices();

    int step = 192;
    std::vector<std::shared_ptr<Node>> deformationNodes;

    for (int i = 0; i < canonicalFrameVertices.size(); i += step) {
        auto dg_v = canonicalFrameVertices[i];
        auto dq   = std::make_shared<DualQuaternion<float>>(0.f, 0.f, 0.f, 0.f, 0.f, 0.f);
        /* FIXME (dig15): set dg_w based on the sampling sparsity of the nodes */
        float dg_w = 2.f;

        deformationNodes.push_back(std::make_shared<Node>(dg_v, dq, dg_w));
    }

    /* initialise the warp field with the sampled deformation nodes */
    warpfield = std::make_shared<Warpfield>();
    warpfield->init(deformationNodes);

    std::cout << "initialised warpfield" << std::endl;
}

void DynFusion::initCanonicalFrame(pcl::PointCloud<pcl::PointXYZ> &vertices, pcl::PointCloud<pcl::Normal> &normals) {
    this->canonicalFrame             = std::make_shared<dynfu::Frame>(0, vertices, normals);
    this->canonicalFrameWarpedToLive = std::make_shared<dynfu::Frame>(0, vertices, normals);

    std::cout << "no. of canonical vertices: " << vertices.size() << std::endl;
}

void DynFusion::warpCanonicalToLiveOpt() {
    CombinedSolverParameters params;
    params.numIter       = 24;
    params.nonLinearIter = 16;
    params.linearIter    = 256;
    params.useOpt        = true;
    params.useOptLM      = false;
    params.earlyOut      = true;

    std::cout << "solving" << std::endl;

    CombinedSolver combinedSolver(*warpfield, params, dynfuParams.tukeyOffset, dynfuParams.psi_data, dynfuParams.lambda,
                                  dynfuParams.psi_reg);

    this->canonicalFrameWarpedToLive = warpfield->warpToLive(this->canonicalFrame);

    pcl::PointCloud<pcl::PointXYZ> canonicalFrameWarpedToLiveVertices = canonicalFrameWarpedToLive->getVertices();
    pcl::PointCloud<pcl::Normal> canonicalFrameWarpedToLiveNormals    = canonicalFrameWarpedToLive->getNormals();

    pcl::PointCloud<pcl::PointXYZ> liveFrameVertices = liveFrame->getVertices();

    std::shared_ptr<dynfu::Frame> correspondingCanonicalFrame = findCorrespondingFrame(
        canonicalFrameWarpedToLiveVertices, canonicalFrameWarpedToLiveNormals, liveFrameVertices);

    combinedSolver.initializeProblemInstance(correspondingCanonicalFrame, this->liveFrame);
    combinedSolver.solveAll();

    std::cout << "solved" << std::endl;
}

std::shared_ptr<dynfu::Frame> DynFusion::findCorrespondingFrame(pcl::PointCloud<pcl::PointXYZ> canonicalVertices,
                                                                pcl::PointCloud<pcl::Normal> canonicalNormals,
                                                                pcl::PointCloud<pcl::PointXYZ> liveVertices) {
    std::vector<cv::Vec3f> vertices;

    for (int i = 0; i < canonicalVertices.size(); i++) {
        vertices.emplace_back(cv::Vec3f(canonicalVertices[i].x, canonicalVertices[i].y, canonicalVertices[i].z));
    }

    /* Initialise KD-tree */
    auto kdCloud = std::make_shared<nanoflann::PointCloud>();
    kdCloud->pts = vertices;
    auto kdTree  = std::make_shared<kd_tree_t>(3, *kdCloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    kdTree->buildIndex();

    pcl::PointCloud<pcl::PointXYZ> correspondingCanonicalVertices;
    pcl::PointCloud<pcl::Normal> correspondingCanonicalNormals;

    std::vector<float> outDistSqr(1);
    std::vector<size_t> retIndex(1);
    for (auto vertex : liveVertices) {
        std::vector<float> query = {vertex.x, vertex.y, vertex.z};
        kdTree->knnSearch(&query[0], 1, &retIndex[0], &outDistSqr[0]);
        size_t index = retIndex[0];

        correspondingCanonicalVertices.push_back(canonicalVertices[index]);
        correspondingCanonicalNormals.push_back(canonicalNormals[index]);
    }

    return std::make_shared<dynfu::Frame>(0, correspondingCanonicalVertices, correspondingCanonicalNormals);
}

void DynFusion::addLiveFrame(int frameID, kfusion::cuda::Cloud &vertices, kfusion::cuda::Normals &normals) {
    auto liveFrameVertices = matToPointCloudVertices(cloudToMat(vertices));
    auto liveFrameNormals  = matToPointCloudNormals(normalsToMat(normals));

    liveFrame = std::make_shared<dynfu::Frame>(frameID, liveFrameVertices, liveFrameNormals);
}

void DynFusion::reconstructSurface() {
    std::cout << "constructing a polygon mesh" << std::endl;

    pcl::PointCloud<pcl::PointXYZ> vertices;
    pcl::PointCloud<pcl::Normal> normals;

    if (frame_counter_ == 0) {
        vertices = canonicalFrame->getVertices();
        normals  = canonicalFrame->getNormals();
    } else {
        vertices = canonicalFrameWarpedToLive->getVertices();
        normals  = canonicalFrameWarpedToLive->getNormals();
    }

    // put vertices and normals in one point cloud
    pcl::PointCloud<pcl::PointNormal>::Ptr cloudWithNormals(new pcl::PointCloud<pcl::PointNormal>);
    pcl::concatenateFields(vertices, normals, *cloudWithNormals);

    // downsample point cloud--otherwise std::bad_alloc
    pcl::PointCloud<pcl::PointNormal>::Ptr cloudWithNormalsDownsampled(new pcl::PointCloud<pcl::PointNormal>);

    pcl::VoxelGrid<pcl::PointNormal> sampler;
    sampler.setInputCloud(cloudWithNormals);
    sampler.setLeafSize(0.05, 0.05f, 0.05f);
    sampler.filter(*cloudWithNormalsDownsampled);

    std::cout << "no. of points used for reconstruction: " << cloudWithNormalsDownsampled->size() << std::endl;

    // perform reconstruction via marching cubes
    pcl::MarchingCubesRBF<pcl::PointNormal> *mc(new pcl::MarchingCubesRBF<pcl::PointNormal>());

    mc->setInputCloud(cloudWithNormalsDownsampled);

    float iso_level                = 0.f;
    float extend_percentage        = 0.f;
    int grid_res                   = 96;
    float off_surface_displacement = 1e-3f;

    mc->setIsoLevel(iso_level);
    mc->setGridResolution(grid_res, grid_res, grid_res);
    mc->setPercentageExtendGrid(extend_percentage);
    mc->setOffSurfaceDisplacement(off_surface_displacement);

    pcl::PolygonMesh::Ptr triangles(new pcl::PolygonMesh);

    std::cout << "beginning marching cubes reconstruction" << std::endl;
    mc->reconstruct(*triangles);
    std::cout << triangles->polygons.size() << " triangles created" << std::endl;

    canonicalWarpedToLiveMesh = *triangles;
}

std::shared_ptr<dynfu::Frame> DynFusion::getCanonicalWarpedToLive() { return this->canonicalFrameWarpedToLive; }

pcl::PolygonMesh DynFusion::getCanonicalWarpedToLiveSurface() { return this->canonicalWarpedToLiveMesh; }

void DynFusion::renderCanonicalWarpedToLive(kfusion::cuda::Image /* &image */, int /* flag */) {
    // const kfusion::KinFuParams &p = params_;
    // image.create(p.rows, flag != 3 ? p.cols : p.cols * 2);
    // auto vertices = matToCloud(vectorToMat(canonicalWarpedToLive->getVertices()));
    // auto normals  = matToCloud(vectorToMat(canonicalWarpedToLive->getNormals()));
    //
    // if ((flag < 1 || flag > 3)) {
    //     kfusion::cuda::renderImage(vertices, normals, params_.intr, params_.light_pose, image);
    // } else if (flag == 2) {
    //     kfusion::cuda::renderTangentColors(normals, image);
    // } else /* if (flag == 3) */ {
    //     kfusion::device::DeviceArray2D<kfusion::RGB> i1(p.rows, p.cols, image.ptr(), image.step());
    //     kfusion::device::DeviceArray2D<kfusion::RGB> i2(p.rows, p.cols, image.ptr() + p.cols, image.step());
    //
    //     kfusion::cuda::renderImage(vertices, normals, params_.intr, params_.light_pose, i1);
    //     kfusion::cuda::renderTangentColors(normals, i2);
    // }
}

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

pcl::PointCloud<pcl::PointXYZ> DynFusion::matToPointCloudVertices(cv::Mat matrix) {
    pcl::PointCloud<pcl::PointXYZ> vertices;

    for (int y = 0; y < matrix.rows; ++y) {
        for (int x = 0; x < matrix.cols; ++x) {
            auto ptVertex = matrix.at<kfusion::Point>(y, x);

            if (!isNaN(ptVertex) && !isZero(ptVertex)) {
                vertices.push_back(pcl::PointXYZ(ptVertex.x, ptVertex.y, ptVertex.z));
            }
        }
    }

    return vertices;
}

pcl::PointCloud<pcl::Normal> DynFusion::matToPointCloudNormals(cv::Mat matrix) {
    pcl::PointCloud<pcl::Normal> normals;

    for (int y = 0; y < matrix.rows; ++y) {
        for (int x = 0; x < matrix.cols; ++x) {
            auto ptNormal = matrix.at<kfusion::Point>(y, x);

            if (!isNaN(ptNormal) && !isZero(ptNormal)) {
                normals.push_back(pcl::Normal(ptNormal.x, ptNormal.y, ptNormal.z));
            }
        }
    }

    return normals;
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
