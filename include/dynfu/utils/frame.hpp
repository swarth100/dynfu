#ifndef DYNFU_FRAME_HPP
#define DYNFU_FRAME_HPP

/* pcl includes */
#include <pcl/common/projection_matrix.h>
#include <pcl/point_types.h>

/* opencv includes */
#include <opencv2/core/affine.hpp>
#include <opencv2/core/core.hpp>

namespace dynfu {

/* */
class Frame {
public:
    /*
     * constructor for a frame
     * takes as input the frame id, the vertices, and the normals
     *
     */
    Frame(int id, pcl::PointCloud<pcl::PointXYZ> vertices, pcl::PointCloud<pcl::Normal> normals);
    ~Frame();

    int getId();
    pcl::PointCloud<pcl::PointXYZ> &getVertices();
    pcl::PointCloud<pcl::Normal> &getNormals();

private:
    int id;
    pcl::PointCloud<pcl::PointXYZ> vertices;
    pcl::PointCloud<pcl::Normal> normals;
};

}  // namespace dynfu
/* DYNFU_FRAME_HPP */
#endif
