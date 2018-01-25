#include <dynfu/utils/frame.hpp>

dynfu::Frame::Frame(int id, pcl::PointCloud<pcl::PointXYZ> vertices, pcl::PointCloud<pcl::Normal> normals) {
    this->id       = id;
    this->vertices = vertices;
    this->normals  = normals;
}

dynfu::Frame::~Frame() = default;

int dynfu::Frame::getId() { return this->id; }

pcl::PointCloud<pcl::PointXYZ>& dynfu::Frame::getVertices() { return this->vertices; }

pcl::PointCloud<pcl::Normal>& dynfu::Frame::getNormals() { return this->normals; }
