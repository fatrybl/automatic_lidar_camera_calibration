/**
 * @file    MICostApp.cpp
 * Evaluate the mutual-information objective of Pandey et al. (AAAI 2012), in nats, at a given LiDAR->camera pose
 * (the initial guess of the parameter file), optionally at perturbed poses, without optimising. Every line also
 * reports how many points are co-observed by the laser scanner and the camera at that pose.
 * Perturbations are increments of the x y z roll pitch yaw parameters of the initial guess.
 * Usage: mi_cost_app <calibration_handler_param.json> [dyaw_deg dpitch_deg droll_deg dx dy dz]...
 */
#include <iostream>
#include <sensors_calib/sensors_calib.hpp>

namespace
{
using PointCloudType = pcl::PointXYZI;
}  // namespace

int main(int argc, char* argv[])
{
    if (argc < 2) {
        std::cerr << "Usage: [app] [path/to/calibration/handler/param.json] [dyaw dpitch droll dx dy dz]..." << std::endl;
        return EXIT_FAILURE;
    }
    perception::CalibrationHandlerParam param = perception::getCalibrationHandlerParam(argv[1]);
    perception::CalibrationHandler<PointCloudType>::Ptr handler(
        new perception::CalibrationHandler<PointCloudType>(param));
    perception::TransformInfo base = perception::getTransformInfo(param.pathToInitialGuess);

    const double deg = boost::math::double_constants::degree;
    const double baseCost = handler->calculateMICost(base);
    printf("pose x=%.4f y=%.4f z=%.4f r=%.3f p=%.3f y=%.3f [deg]  MI=%.6f  points=%d  (normalize_mi=%d, estimator=%d)\n",
           base(0), base(1), base(2), base(3) / deg, base(4) / deg, base(5) / deg, baseCost, handler->coObservedPoints(),
           param.normalizeMI, param.probabilityEstimatorType);
    for (int i = 2; i + 5 < argc; i += 6) {
        perception::TransformInfo t = base;
        t(5) += std::atof(argv[i]) * deg;
        t(4) += std::atof(argv[i + 1]) * deg;
        t(3) += std::atof(argv[i + 2]) * deg;
        t(0) += std::atof(argv[i + 3]);
        t(1) += std::atof(argv[i + 4]);
        t(2) += std::atof(argv[i + 5]);
        const double cost = handler->calculateMICost(t);
        printf("perturb dyaw=%s dpitch=%s droll=%s dx=%s dy=%s dz=%s  MI=%.6f  points=%d\n", argv[i], argv[i + 1],
               argv[i + 2], argv[i + 3], argv[i + 4], argv[i + 5], cost, handler->coObservedPoints());
    }
    return EXIT_SUCCESS;
}
