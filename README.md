Dynfu
============
Dependencies:
* Fermi or Kepler or newer
* CUDA 5.0 or higher
* OpenCV 2.4.9 with new Viz module (only opencv_core, opencv_highgui, opencv_imgproc, opencv_viz modules required). Make sure that WITH_VTK flag is enabled in CMake during OpenCV configuration.
* OpenNI v1.5.4 (for Windows can download and install from http://pointclouds.org/downloads/windows.html)

Implicit dependency (needed by opencv_viz):
* VTK 5.8.0 or higher. (apt-get install on linux, for windows please download and compile from www.vtk.org)

## References
The KD-tree is created using [nanoflann](https://github.com/jlblancoc/nanoflann)
```
@misc{blanco2014nanoflann,
  title        = {nanoflann: a {C}++ header-only fork of {FLANN}, a library for Nearest Neighbor ({NN}) wih KD-trees},
  author       = {Blanco, Jose Luis and Rai, Pranjal Kumar},
  howpublished = {\url{https://github.com/jlblancoc/nanoflann}},
  year         = {2014}
}
```
