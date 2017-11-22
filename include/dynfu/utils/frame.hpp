#ifndef DYNFU_FRAME_HPP
#define DYNFU_FRAME_HPP

/* opencv includes */
#include <opencv2/core/affine.hpp>
#include <opencv2/core/core.hpp>

/* */
class Frame {
public:
    /*
     * constructor for a frame
     * takes as input the frame id, the vertices, and the normals
     */
    Frame(int id, std::vector<cv::Vec3f> vertices, std::vector<cv::Vec3f> normals);
    ~Frame();

    int getId();
    std::vector<cv::Vec3f> getVertices();
    std::vector<cv::Vec3f> getNormals();

private:
    int id;
    std::vector<cv::Vec3f> vertices;
    std::vector<cv::Vec3f> normals;
};

/* DYNFU_FRAME_HPP */
#endif
