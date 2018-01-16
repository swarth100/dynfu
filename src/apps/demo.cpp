/* dynfu includes */
#include <dynfu/dyn_fusion.hpp>

/* opencv includes */
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

/* pcl includes */
#include <pcl/io/pcd_io.h>
#include <pcl/io/vtk_io.h>

/* sys headers */
#include <iostream>

struct DynFuApp {
    DynFuApp(std::string filePath, bool visualizer) : exit_(false), filePath_(filePath), visualizer_(visualizer) {
        DynFuParams params = DynFuParams::defaultParams();
        dynfu              = std::make_shared<DynFusion>(params);
    }

    void show_raycasted(std::shared_ptr<DynFusion> dynfu, int i) {
        const int mode = 3;
        (*dynfu).renderImage(view_device_, mode);

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
    void show_canonical_warped_to_live(std::shared_ptr<DynFusion> dynfu, int i) {
        const int mode = 3;
        dynfu->renderCanonicalWarpedToLive(canonical_to_live_view_device_, mode);

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

    void save_polygon_mesh(std::shared_ptr<DynFusion> dynfu, int i) {
        auto mesh = dynfu->getCanonicalWarpedToLiveSurface();
        pcl::io::saveVTKFile(outPath_ + "/" + std::to_string(i) + "_mesh.vtk", mesh);

        std::cout << "saved model mesh to .vtk" << std::endl;
    }

    void save_canonical_warped_to_live_point_cloud(std::shared_ptr<DynFusion> dynfu, int i) {
        auto vertices = dynfu->getCanonicalWarpedToLive()->getVertices();

        /* save to PCL */
        std::string filenameStr = outPath_ + "/pcl_canonical_to_live" + std::to_string(i) + ".pcd";
        try {
            pcl::io::savePCDFileASCII(filenameStr, vertices);
        } catch (...) {
            std::cout << "Could not save to " + filenameStr << std::endl;
        }
    }

    void take_cloud(std::shared_ptr<DynFusion> dynfu) {
        kfusion::cuda::DeviceArray<kfusion::Point> cloud = (*dynfu).tsdf().fetchCloud(cloud_buffer);
        cv::Mat cloud_host(1, static_cast<int>(cloud.size()), CV_32FC4);
        cloud.download(cloud_host.ptr<kfusion::Point>());
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
        dynfu = dynfu;
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
                kfusion::SampledScopeTime fps(time_ms);
                // (void) kfusion::fps;
                has_image = (*(dynfu))(depth_device_);
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

            save_polygon_mesh(dynfu, i);

            if (has_image) {
                show_raycasted(dynfu, i);
                show_canonical_warped_to_live(dynfu, i);

                save_canonical_warped_to_live_point_cloud(dynfu, i);
            }

            // show_depth(depth);
        }

        return true;
    }

    std::shared_ptr<DynFusion> dynfu;

    std::string filePath_;
    std::string outPath_;

    /* point cloud viz */
    std::shared_ptr<PointCloudViz> pointCloudViz;
    /* point cloud viz thread */
    std::shared_ptr<std::thread> vizThread;

    bool exit_, visualizer_;

    cv::Mat view_host_;
    cv::Mat canonical_to_live_view_host_;

    kfusion::cuda::Image view_device_;
    kfusion::cuda::Image canonical_to_live_view_device_;

    kfusion::cuda::Depth depth_device_;
    kfusion::cuda::DeviceArray<kfusion::Point> cloud_buffer;
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
    kfusion::cuda::setDevice(device);
    kfusion::cuda::printShortCudaDeviceInfo(device);

    if (kfusion::cuda::checkIfPreFermiGPU(device)) {
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
        std::cout << "std::bad_alloc" << std::endl;
    } catch (const std::exception & /*e*/) {
        std::cout << "std::exception" << std::endl;
    }

    return 0;
}
