/**
 * @file    SensorsCalibApp.cpp
 *
 * @author  btran
 *
 */

#include <iostream>

#include <pcl/io/pcd_io.h>

#include <sensors_calib/sensors_calib.hpp>

namespace
{
using PointCloudType = pcl::PointXYZI;
using PointCloud = pcl::PointCloud<PointCloudType>;
using PointCloudPtr = PointCloud::Ptr;
}  // namespace

int main(int argc, char* argv[])
{
    if (argc != 2) {
        std::cerr << "Usage: [app] [path/to/calibration/handler/data]" << std::endl;
        return EXIT_FAILURE;
    }

    const std::string PARAM_PATH = argv[1];

    perception::CalibrationHandlerParam param = perception::getCalibrationHandlerParam(PARAM_PATH);
    perception::CalibrationHandler<PointCloudType>::Ptr calibrationHandler(
        new perception::CalibrationHandler<PointCloudType>(param));

    auto transform = calibrationHandler->optimize();
    const auto crlb = calibrationHandler->calculateCRLB(transform);
    const auto visualizedImgs = calibrationHandler->drawPointCloudOnImagePlane(transform);
    const auto projectedClouds = calibrationHandler->projectOnPointCloud(transform);

    const double radian = boost::math::double_constants::radian;
    printf("x: %f[m], y: %f[m], z: %f[m], r: %f[deg], p: %f[deg], y_deg: %f[deg]\n", transform(0), transform(1),
           transform(2), transform(3) * radian, transform(4) * radian, transform(5) * radian);
    // fork: the uncertainty of the estimate from the Cramer-Rao lower bound (eqs. 12-14)
    printf("CRLB std: x: %f[m], y: %f[m], z: %f[m], r: %f[deg], p: %f[deg], y: %f[deg]\n", std::sqrt(crlb(0, 0)),
           std::sqrt(crlb(1, 1)), std::sqrt(crlb(2, 2)), std::sqrt(crlb(3, 3)) * radian, std::sqrt(crlb(4, 4)) * radian,
           std::sqrt(crlb(5, 5)) * radian);

    for (std::size_t i = 0; i < visualizedImgs.size(); ++i) {
        const auto& curImg = visualizedImgs[i];
        cv::imwrite("img" + std::to_string(i) + ".png", curImg);
    }

    for (std::size_t i = 0; i < projectedClouds.size(); ++i) {
        const auto& curCloud = projectedClouds[i];
        pcl::io::savePCDFileASCII("cloud" + std::to_string(i) + ".pcd", *curCloud);
    }

    return EXIT_SUCCESS;
}
