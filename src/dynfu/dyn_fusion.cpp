#include <dynfu/dyn_fusion.hpp>

DynFusion::DynFusion() = default;
/* We initialise the dynamic fusion with the initals vertices and normals */
void DynFusion::init(kfusion::cuda::Cloud &vertices) {
    cv::Mat cloudHost = cloudToMat(vertices);
    std::vector<cv::Vec3f> canonical(cloudHost.rows * cloudHost.cols);

    for (int y = 0; y < cloudHost.cols; ++y) {
        for (int x = 0; x < cloudHost.rows; ++x) {
            auto point = cloudHost.at<kfusion::Point>(x, y);
            if (!(std::isnan(point.x) || std::isnan(point.y) || std::isnan(point.z))) {
                canonical[x + cloudHost.rows * y] = cv::Vec3f(point.x, point.y, point.z);
            }
        }
    }

    /* Sample the deformation nodes */
    int steps = 50;
    std::vector<std::shared_ptr<Node>> deformationNodes;
    for (int i = 0; i < canonical.size() - steps; i += steps) {
        auto dq = std::make_shared<DualQuaternion<float>>(0.f, 0.f, 0.f, 0.f, 0.f, 0.f);
        deformationNodes.push_back(std::make_shared<Node>(canonical[i], dq, 1.f));
    }
    /* Initialise the warp field with the inital frames vertices */
    warpfield = std::make_shared<Warpfield>();
    warpfield->init(deformationNodes);
}

/* TODO: Add comment */
DynFusion::~DynFusion() = default;

/* TODO: Add comment */
void DynFusion::initCanonicalFrame() {}

/* TODO: Add comment */
void DynFusion::warpCanonicalToLive() {
    // query the solver passing to it the canonicalFrame, liveFrame, and
    // prevwarpField
}

void DynFusion::addLiveFrame(int frameID, kfusion::cuda::Cloud &vertices, kfusion::cuda::Normals &normals) {
    auto liveFrameVertices = matToVector(cloudToMat(vertices));
    auto liveFrameNormals  = matToVector(normalsToMat(normals));

    liveFrame = std::make_shared<Frame>(frameID, liveFrameVertices, liveFrameNormals);
}

/* Calculate Dual Quaternion Blending */
/* Get the dg_se3 from each of the nodes, time it by the weight and calculate the sum */
/* Before returning, normalise the dual quaternion */
std::shared_ptr<DualQuaternion<float>> DynFusion::calcDQB(cv::Vec3f point) {
    /* From the warp field get the k (8) closest points */
    Warpfield warpfield;
    auto nearestNeighbors = warpfield.findNeighbors(KNN, point);
    /* Then for each of the Nodes compare the distance between the vector of the Node and the point */
    /* Apply the formula to get w(x) */
    DualQuaternion<float> transformationSum(0.f, 0.f, 0.f, 0.f, 0.f, 0.f);
    for (auto node : nearestNeighbors) {
        float nodeWeight = getWeight(node, point);

        DualQuaternion<float> dg_se3                  = *node->getTransformation();
        DualQuaternion<float> weighted_transformation = dg_se3 * nodeWeight;
        transformationSum += weighted_transformation;
    }
    /*Normalise the sum */
    DualQuaternion<float> dual_quaternion_blending = transformationSum.normalize();

    return std::make_shared<DualQuaternion<float>>(dual_quaternion_blending);
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

/* Calculate the weight using the position in the canonical frame and the radial weight of the node */
float DynFusion::getWeight(std::shared_ptr<Node> node, cv::Vec3f point) {
    return getWeightT<float>(node->getPosition(), node->getWeight(), point);
}
