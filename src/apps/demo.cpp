/* kinfu includes */
#include <kfusion/kinfu.hpp>

/* opencv includes */
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

/* pcl includes */
#include <pcl/io/pcd_io.h>
#include <pcl/io/vtk_io.h>
#include <pcl/point_types.h>

/* sys headers */
#include <iostream>

struct DynFuApp {
    DynFuApp(std::string filePath, bool visualizer) : exit_(false), filePath_(filePath), visualizer_(visualizer) {
        KinFuParams params = KinFuParams::default_params();
        kinfu_             = KinFu::Ptr(new KinFu(params));
    }

    void show_raycasted(KinFu *kinfu, int i) {
        const int mode = 3;
        (*kinfu).renderImage(view_device_, mode);

        view_host_.create(view_device_.rows(), view_device_.cols(), CV_8UC4);
        view_device_.download(view_host_.ptr<void>(), view_host_.step);

        if (visualizer_) {
            cv::imshow("scene", view_host_);
            cvWaitKey(10);
        }

        std::string path = outPath_ + "/raycast" + std::to_string(i) + ".png";

        cv::cvtColor(view_host_, view_host_, CV_BGR2GRAY);
        cv::imwrite(path, view_host_);
    }

    /* data from the gpu */
    void show_canonical_warped_to_live(KinFu *kinfu, int i) {
        const int mode = 3;
        kinfu->renderCanonicalWarpedToLive(canonical_to_live_view_device_, mode);

        canonical_to_live_view_host_.create(canonical_to_live_view_device_.rows(),
                                            canonical_to_live_view_device_.cols(), CV_8UC4);
        canonical_to_live_view_device_.download(canonical_to_live_view_host_.ptr<void>(),
                                                canonical_to_live_view_host_.step);

        if (visualizer_) {
            cv::imshow("canonical warped to live", canonical_to_live_view_host_);
            cvWaitKey(10);
        }

        std::string path = outPath_ + "/canonical_to_live" + std::to_string(i) + ".png";

        cv::cvtColor(canonical_to_live_view_host_, canonical_to_live_view_host_, CV_BGR2GRAY);
        cv::imwrite(path, canonical_to_live_view_host_);
    }

    void save_polygon_mesh(KinFu *kinfu) {
        auto mesh = kinfu->canonicalMesh;
        pcl::io::saveVTKFile(outPath_ + "/canonicalModelMesh.vtk", mesh);

        std::cout << "saved canonical model mesh to .vtk" << std::endl;
    }

    void save_canonical_warped_to_live_point_cloud(KinFu *kinfu, int i) {
        auto vertices = kinfu->canonicalWarpedToLive->getVertices();

        /* initialise the point cloud */
        auto cloud    = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        cloud->width  = vertices.size();
        cloud->height = 1;
        cloud->points.resize(cloud->width * cloud->height);

        /* iterate through vertices */
        for (size_t i = 0; i < vertices.size(); i++) {
            const cv::Vec3f &pt = vertices[i];
            cloud->points[i]    = pcl::PointXYZ(pt[0], pt[1], pt[2]);
        }

        /* save to PCL */
        std::string filenameStr = outPath_ + "/pcl_canonical_to_live" + std::to_string(i) + ".pcd";
        try {
            pcl::io::savePCDFileASCII(filenameStr, (*cloud));
        } catch (...) {
            std::cout << "Could not save to " + filenameStr << std::endl;
        }
    }

    void take_cloud(KinFu *kinfu) {
        cuda::DeviceArray<Point> cloud = (*kinfu).tsdf().fetchCloud(cloud_buffer);
        cv::Mat cloud_host(1, static_cast<int>(cloud.size()), CV_32FC4);
        cloud.download(cloud_host.ptr<Point>());
    }

    void loadFiles(std::vector<cv::String> *depths, std::vector<cv::String> *images) {
        if (!boost::filesystem::exists(filePath_)) {
            std::cerr << "Error: Directory '" << filePath_ << "' does not exist. Exiting" << std::endl;
            exit(EXIT_FAILURE);
        }

        if (!boost::filesystem::exists(filePath_ + "/depth") || !boost::filesystem::exists(filePath_ + "/color")) {
            std::cerr << "Error: Directory should contain 'color' and 'depth' directories. Exiting" << std::endl;
            exit(EXIT_FAILURE);
        }

        cv::glob(filePath_ + "/depth", *depths);
        cv::glob(filePath_ + "/color", *images);

        std::sort((*depths).begin(), (*depths).end());
        std::sort((*images).begin(), (*images).end());
    }

    /* create a new directory /out if not already present inside the input folder */
    void createOutputDirectory() {
        outPath_ = filePath_ + "/out";
        boost::filesystem::path dir(outPath_);

        if (boost::filesystem::create_directory(dir)) {
            std::cout << "created output directory" << std::endl;
        }
    }

    bool execute() {
        KinFu &kinfu = *kinfu_;
        cv::Mat depth, image;

        double time_ms = 0;
        bool has_image = false;

        std::vector<cv::String> depths;
        std::vector<cv::String> images;

        loadFiles(&depths, &images);
        createOutputDirectory();

        for (int i = 0; i < depths.size(); ++i) {
            auto depth = cv::imread(depths[i], CV_LOAD_IMAGE_ANYDEPTH);
            auto image = cv::imread(images[i], CV_LOAD_IMAGE_COLOR);

            if (!image.data || !depth.data) {
                std::cerr << "Error: Image could not be read. Check for improper"
                          << " permissions, or invalid formats. Exiting..." << std::endl;
                exit(EXIT_FAILURE);
            }

            depth_device_.upload(depth.data, depth.step, depth.rows, depth.cols);

            {
                SampledScopeTime fps(time_ms);
                (void) fps;
                has_image = kinfu(depth_device_);
            }

            if (visualizer_ && i == 0) {
                cv::namedWindow("image", cv::WINDOW_AUTOSIZE);
                cv::namedWindow("depth", cv::WINDOW_AUTOSIZE);
            }

            if (visualizer_) {
                cv::imshow("image", image);
                cv::imshow("depth", depth);
                cv::waitKey(10);
            }

            if (visualizer_ && i == 1) {
                cv::namedWindow("scene", cv::WINDOW_AUTOSIZE);
                cv::namedWindow("canonical warped to live", cv::WINDOW_AUTOSIZE);
            }

            if (i == 1) {
                // save_polygon_mesh(&kinfu);
            }

            if (has_image) {
                show_raycasted(&kinfu, i);
                show_canonical_warped_to_live(&kinfu, i);

                save_canonical_warped_to_live_point_cloud(&kinfu, i);
            }

            // show_depth(depth);
        }

        return true;
    }

    KinFu::Ptr kinfu_;

    std::string filePath_;
    std::string outPath_;

    /* point cloud viz */
    std::shared_ptr<PointCloudViz> pointCloudViz;
    /* point cloud viz thread */
    std::shared_ptr<std::thread> vizThread;

    bool exit_, visualizer_;

    cv::Mat view_host_;
    cv::Mat canonical_to_live_view_host_;

    cuda::Image view_device_;
    cuda::Image canonical_to_live_view_device_;

    cuda::Depth depth_device_;
    cuda::DeviceArray<Point> cloud_buffer;
};

/* Parse the input flag and determine the file path and whether or not to enable visualizer
 * Any flags will be matched and the last argument which does not match the flag will be
 * treated as filepath.
 */
void parseFlags(std::vector<std::string> args, std::string *filePath, bool *visualizer) {
    std::vector<std::string> flags = {"-h", "--help", "--enable-viz"};
    for (auto arg : args) {
        if (std::find(std::begin(flags), std::end(flags), arg) != std::end(flags)) {
            if (arg == "-h" || arg == "--help") {
                std::cout << "USAGE: dynamicfusion [OPTIONS] <file path>" << std::endl;
                std::cout << "\t--help -h:    Display help" << std::endl;
                std::cout << "\t--enable-viz: Enable visualizer" << std::endl;
                std::exit(EXIT_SUCCESS);
            }
            if (arg == "--enable-viz") {
                *visualizer = true;
            }
        } else {
            *filePath = arg;
        }
    }
}

int main(int argc, char *argv[]) {
    int device = 0;
    cuda::setDevice(device);
    cuda::printShortCudaDeviceInfo(device);

    if (cuda::checkIfPreFermiGPU(device)) {
        std::cout << std::endl
                  << "Kinfu is not supported for pre-Fermi GPU "
                     "architectures, and not built for them by "
                     "default. Exiting..."
                  << std::endl;
        return 1;
    }

    /* program requires at least one argument--the path to the directory where the source files are */
    if (argc < 2) {
        return std::cerr << "Error: incorrect number of arguments. Please supply path to source data. Exiting..."
                         << std::endl,
               -1;
    }

    std::vector<std::string> args(argv + 1, argv + argc);
    std::string filePath;
    bool visualizer = false;
    parseFlags(args, &filePath, &visualizer);

    /* disable the visualiser when running over SSH */
    if (visualizer && getenv("SSH_CLIENT")) {
        return std::cerr << "Error: cannot run visualiser while in SSH environment. Please run locally or disable "
                            "visualiser. Exiting..."
                         << std::endl,
               -1;
    }

    DynFuApp app(filePath, visualizer);

    /* execute */
    try {
        app.execute();
    } catch (const std::bad_alloc & /*e*/) {
        std::cout << "Bad alloc" << std::endl;
    } catch (const std::exception & /*e*/) {
        std::cout << "Exception" << std::endl;
    }

    return 0;
}
