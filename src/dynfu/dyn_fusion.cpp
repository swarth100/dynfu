#include <dynfu/dyn_fusion.hpp>

#include <kfusion/internal.hpp>
#include <kfusion/precomp.hpp>

DynFuParams DynFuParams::defaultParams() {
    DynFuParams p;

    kfusion::KinFuParams kinfuParams = kfusion::KinFuParams::default_params();
    kinfuParams.volume_dims          = cv::Vec3i::all(128);  // number of voxels
    p.kinfuParams                    = kinfuParams;

    p.tukeyOffset = 4.652;

    /* regularisation parameter */
    p.lambda = 200;
    /* parameter to calculate tukey biweights */
    p.psi_data = 0.01;
    /* parameter to calculate huber weights */
    p.psi_reg = 1e-4;

    /* no. of levels in the regularisation hierarchy */
    p.L = 4;
    /* parameter for updating the regularisation graph */
    p.beta = 4;

    /* decimation density */
    p.epsilon = 0.1;

    return p;
};

DynFusion::DynFusion(const DynFuParams &params) : kfusion::KinFu::KinFu(params.kinfuParams) { dynfuParams = params; };

DynFusion::~DynFusion() = default;

DynFuParams &DynFusion::params() { return dynfuParams; }

/* PIPELINE
 *
 * 1. get canonical model vertices & normals via marching cubes
 * 2. match canonical model vertices to the live depth frame via kd-trees & icp
 * 3. estimate warpfield
 * 4. perform dense non-rigid surface fusion
 * 5. update warpfield
 *
 */
bool DynFusion::operator()(const kfusion::cuda::Depth &depth, const kfusion::cuda::Image & /*image*/) {
    std::cout << "frame no. " << frame_counter_ << std::endl;

    const kfusion::KinFuParams &p = params_;
    const int LEVELS              = icp_->getUsedLevelsNum();

    /* compute distances using the depth frame */
    kfusion::cuda::computeDists(depth, dists_, p.intr);
    /* apply the bilateral filter to the depth frame */
    depthBilateralFilter(depth, curr_.depth_pyr[0], p.bilateral_kernel_size, p.bilateral_sigma_spatial,
                         p.bilateral_sigma_depth);

    /* do depth truncation */
    if (p.icp_truncate_depth_dist > 0) {
        kfusion::cuda::depthTruncation(curr_.depth_pyr[0], p.icp_truncate_depth_dist);
    }

    kfusion::cuda::waitAllDefaultStream();

    /* can't do more with the first frame */
    if (frame_counter_ == 0) {
        /* volume integration */
        volume_->integrate(dists_, poses_.back(), p.intr);

        /* extract the canonical zero-level set of tsdf via marching cubes */
        kfusion::device::DeviceArray<pcl::PointXYZ> triangles_buffer_device_;
        kfusion::device::DeviceArray<pcl::PointXYZ> triangles_device = mc_->run(*volume_, triangles_buffer_device_);
        kfusion::cuda::waitAllDefaultStream();
        mesh_ptr_ = convertToMesh(triangles_device);

        std::cout << "no. of triangles in the mesh: " << triangles_device.size() << std::endl;

        /* FIXME (dig15): temporary workaround until normals are computed via mc */
        pcl::PointCloud<pcl::PointXYZ> pcVertices;

        pcVertices.width  = (int) triangles_buffer_device_.size();
        pcVertices.height = 1;
        triangles_device.download(pcVertices.points);

        pcl::PointCloud<pcl::Normal> pcNormals;
        pcl::copyPointCloud<pcl::PointXYZ, pcl::Normal>(pcVertices, pcNormals);

        /* store points and normals */
        curr_.points_pyr.swap(prev_.points_pyr);
        curr_.normals_pyr.swap(prev_.normals_pyr);

        /* initialise the warpfield */
        init(pcVertices, pcNormals);

        return ++frame_counter_, false;
    }

    /*
     * icp--not being done yet
     */

    cv::Affine3f affine;
    poses_.push_back(poses_.back() * affine); /* transformation live -> canonical */

    /*
     * volume integration
     *
     * FIXME (dig15): we're not updating the canonical model yet but we need a new volume to get out live vertices via
     * marching cubes; shouldn't have to do this
     */
    {
        volume_->clear();
        volume_->integrate(dists_, poses_.back(), p.intr);
    }

    /* extract points and normals from the live zero-level set of tsdf via marching cubes */
    kfusion::device::DeviceArray<pcl::PointXYZ> triangles_buffer_device_;
    kfusion::device::DeviceArray<pcl::PointXYZ> triangles_device = mc_->run(*volume_, triangles_buffer_device_);
    kfusion::cuda::waitAllDefaultStream();
    mesh_ptr_ = convertToMesh(triangles_device);

    std::cout << "no. of triangles in the mesh: " << triangles_device.size() << std::endl;

    /* FIXME (dig15): temporary workaround until normals are computed via mc */
    pcl::PointCloud<pcl::PointXYZ> pcVertices;

    pcVertices.width  = (int) triangles_buffer_device_.size();
    pcVertices.height = 1;
    triangles_device.download(pcVertices.points);

    pcl::PointCloud<pcl::Normal> pcNormals;
    pcl::copyPointCloud<pcl::PointXYZ, pcl::Normal>(pcVertices, pcNormals);

    /* add new live frame to dynfu */
    addLiveFrame(frame_counter_, pcVertices, pcNormals);

    /* warp canonical frame to live frame */
    warpCanonicalToLiveOpt(affine);
    /* update warpfield */
    warpfield->update(this->getCanonicalWarpedToLive());

    return ++frame_counter_, true;
}

void DynFusion::init(pcl::PointCloud<pcl::PointXYZ> &canonicalVertices,
                     pcl::PointCloud<pcl::Normal> &canonicalNormals) {
    initCanonicalFrame(canonicalVertices, canonicalNormals);

    int step = 192;                                      /* step size */
    std::vector<std::shared_ptr<Node>> deformationNodes; /* vector storing deformation nodes */

    for (int i = 0; i < canonicalVertices.size(); i += step) {
        auto dg_v  = canonicalVertices[i]; /* coordinates of deformation node */
        auto dq    = std::make_shared<DualQuaternion<float>>(0.f, 0.f, 0.f, 0.f, 0.f,
                                                          0.f); /* transformation stored in deformation node */
        float dg_w = 3 * dynfuParams.epsilon;                      /* radial basis weight stored in deformation node */

        deformationNodes.push_back(std::make_shared<Node>(dg_v, dq, dg_w));
    }

    /* initialise warpfield with the sampled deformation nodes */
    warpfield = std::make_shared<Warpfield>();
    warpfield->init(dynfuParams.epsilon, deformationNodes);

    std::cout << "initialised warpfield" << std::endl;
}

void DynFusion::initCanonicalFrame(pcl::PointCloud<pcl::PointXYZ> &vertices, pcl::PointCloud<pcl::Normal> &normals) {
    this->canonicalFrame             = std::make_shared<dynfu::Frame>(0, vertices, normals);
    this->canonicalFrameWarpedToLive = std::make_shared<dynfu::Frame>(0, vertices, normals);

    std::cout << "no. of canonical vertices: " << vertices.size() << std::endl;
}

void DynFusion::addLiveFrame(int frameID, pcl::PointCloud<pcl::PointXYZ> &vertices,
                             pcl::PointCloud<pcl::Normal> &normals) {
    liveFrame = std::make_shared<dynfu::Frame>(frameID, vertices, normals);
}

void DynFusion::warpCanonicalToLiveOpt(cv::Affine3f affine) {
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

    combinedSolver.initializeProblemInstance(correspondingCanonicalFrame, this->liveFrame, affine);
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

    /* init kd-tree */
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

std::shared_ptr<dynfu::Frame> DynFusion::getCanonicalWarpedToLive() { return this->canonicalFrameWarpedToLive; }
