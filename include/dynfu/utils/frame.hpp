#include <opencv2/core/affine.hpp>
#include <opencv2/core/core.hpp>

/* */
class Frame {
public:
    Frame(void *addIt);  // TODO figure out
    ~Frame();

    void getId();
    void getVertices();
    void getNormals();

private:
    int id;
    std::vector<cv::Vec3f> vertices;
    std::vector<cv::Vec3f> normals;
};
