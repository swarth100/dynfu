#include <dynfu/warp_field.hpp>

#define KNN_NEIGHBOURS 8

Warpfield::Warpfield() {
    typedef nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, PointCloud>, PointCloud, 3>
        kd_tree_t;

    kd_tree_t* index_;
    nanoflann::KNNResultSet<float>* resultSet_;
    PointCloud cloud;

    index_ = new kd_tree_t(3, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));

    /* Attempt to initiate a warp field with 9 vectors.
     * The vectors correspond to the 8 vertices of a cube + the origin
     */
    std::vector<cv::Vec3f> warp_init;
    warp_init.emplace_back(cv::Vec3f(1, 1, 1));
    warp_init.emplace_back(cv::Vec3f(1, 1, -1));
    warp_init.emplace_back(cv::Vec3f(1, -1, 1));
    warp_init.emplace_back(cv::Vec3f(1, -1, -1));
    warp_init.emplace_back(cv::Vec3f(-1, 1, 1));
    warp_init.emplace_back(cv::Vec3f(-1, 1, -1));
    warp_init.emplace_back(cv::Vec3f(-1, -1, 1));
    warp_init.emplace_back(cv::Vec3f(-1, -1, -1));
    warp_init.emplace_back(cv::Vec3f(0, 0, 0));

    cloud.pts = warp_init;

    std::vector<cv::Vec3f> canonical_vertices;
    canonical_vertices.emplace_back(cv::Vec3f(0, 0, 0));
    canonical_vertices.emplace_back(cv::Vec3f(-1, -1, -1));
    canonical_vertices.emplace_back(cv::Vec3f(1, 1, 1));
    canonical_vertices.emplace_back(cv::Vec3f(2, 2, 2));
    canonical_vertices.emplace_back(cv::Vec3f(3, 3, 3));

    index_->buildIndex();
    int num_results = 1;
    std::vector<float> out_dist_sqr(num_results);
    std::vector<size_t> ret_index(num_results);
    std::vector<float> query = {canonical_vertices[0][0], canonical_vertices[0][1], canonical_vertices[0][2]};
    num_results              = index_->knnSearch(&query[0], num_results, &ret_index[0], &out_dist_sqr[0]);
    std::cout << num_results << std::endl;
    std::cout << ret_index[0] << std::endl;

    // for(auto v : canonical_vertices)
    //{
    // index_->findNeighbors(*resultSet_, v.val, nanoflann::SearchParams(10));
    // for(int i = 0; i < KNN_NEIGHBOURS; i++)
    // std::cout<<ret_index_[i]<<" ";
    // std::cout<<std::endl;
    //}
}

/* Find the index of k closest neighbour for the given point */
std::vector<int> findNeighbors(int numNeighbour, cv::Vec3f point) {
    std::vector<float> outDistSqr(numNeighbour);
    std::vector<size_t> retIndex(numNeighbour);
    /* Unpack the Vec3f into vector */
    std::vector<float> query = {point[0], point[1], point[2]};
    int n                    = knnTree->knnSearch(&query[0], numNeighbour, &retIndex[0], &outDistSqr[0]);
    retIndex.resize(n);
    return retIndex
}
