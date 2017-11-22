/*
 *  Point cloud class for nanoflann. Used to calculate the closest neighbours
 *  Mostly based on implementation
 *  https://github.com/jlblancoc/nanoflann/blob/master/examples/utils.h
 */

#include <kfusion/types.hpp>
struct PointCloud {
    std::vector<cv::Vec3f> pts;
    /* Must return the number of data points */
    inline size_t kdtree_get_point_count() const { return pts.size(); }

    /*
     *  Returns the dim'th component of the idx'th point in the class:
     *  Since this is inlined and the "dim" argument is typically an immediate value, the
     *  "if/else's" are actually solved at compile time.
     */
    inline float kdtree_get_pt(const size_t idx, int dim) const {
        /*
         *  x = point[0];
         *  y = point[1];
         *  z = point[2];
         */
        if (dim == 0)
            return pts[idx][0];
        else if (dim == 1)
            return pts[idx][1];
        else
            return pts[idx][2];
    }

    /*
     *  Optional bounding-box computation: return false to default to a standard bbox computation loop.
     *  Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it
     * again. Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
     */
    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /* bb */) const {
        return false;
    }
};
