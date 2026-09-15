/**
 * @file    CalibrationHandler.ipp
 *
 * @author  btran
 *
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace perception
{
namespace detail
{
static constexpr double CRLB_MIN_DENSITY = 1e-12;

// |s^T s / s^T g| for the last step s and the change g of the unit gradient; the upper bound when there was no step
inline double barzilaiBorweinStep(const Eigen::Vector3d& lastStep, const Eigen::Vector3d& gradientChange,
                                  const double upper, const double epsilon)
{
    if (lastStep.norm() == 0) {
        return upper;
    }

    return std::fabs(lastStep.dot(lastStep) / (lastStep.dot(gradientChange) + epsilon));
}
}  // namespace detail

template <typename POINT_CLOUD_TYPE>
CalibrationHandler<POINT_CLOUD_TYPE>::CalibrationHandler(const CalibrationHandlerParam& param)
    : m_param(param)
    , m_initialGuess(TransformInfo::Identity())
{
    validate<CalibrationHandlerParam>(m_param);

    if (!param.pathToInitialGuess.empty()) {
        m_initialGuess = perception::getTransformInfo(m_param.pathToInitialGuess);
    }
    std::vector<std::string> imagePaths = parseMetaDataFile(m_param.pathToImages);
    std::vector<std::string> pointcloudPaths = parseMetaDataFile(m_param.pathToPointClouds);

    if (imagePaths.size() != pointcloudPaths.size()) {
        throw std::runtime_error("number of images and point clouds must be the same");
    }

    if (imagePaths.empty()) {
        throw std::runtime_error("empty calibration data");
    }

    const int numSamples = imagePaths.size();
    m_colorImgs.reserve(numSamples);
    m_grayImgs.reserve(numSamples);
    m_poinclouds.reserve(numSamples);

    PointCloudPtr inCloud(new PointCloud);

    for (int i = 0; i < numSamples; ++i) {
        // fork: fresh buffers per scan; cvtColor wrote every grey image into one reused buffer, so all stored grey
        // images were the last one and the multi-scan objective paired every scan with the last image
        cv::Mat colorImg = cv::imread(imagePaths[i]);
        cv::Mat grayImg;

        if (colorImg.empty()) {
            throw std::runtime_error("failed to read: " + imagePaths[i]);
        }

        if (pcl::io::loadPCDFile<PointCloudType>(pointcloudPaths[i], *inCloud) == -1) {
            throw std::runtime_error("failed to read: " + pointcloudPaths[i]);
        }

        if (m_param.filterInputImage) {
            cv::Mat filtered;
            cv::bilateralFilter(colorImg, filtered, m_param.filterDiameter, m_param.sigmaColor, m_param.sigmaSpace);
            cv::cvtColor(filtered, grayImg, cv::COLOR_BGR2GRAY);
        } else {
            cv::cvtColor(colorImg, grayImg, cv::COLOR_BGR2GRAY);
        }

        m_colorImgs.emplace_back(colorImg);
        m_grayImgs.emplace_back(grayImg);

        inCloud = perception::PointCloudFilter<PointCloudType>::filterXYZAxis(
            inCloud, m_param.xMin, m_param.xMax, m_param.yMin, m_param.yMax, m_param.zMin, m_param.zMax);

        // fork: the paper takes the reflectivity X_i in [0, 255] (section 2.2)
        for (const auto& point : inCloud->points) {
            if (!std::isfinite(point.intensity) || point.intensity < 0 || point.intensity > MAX_BINS - 1) {
                throw std::runtime_error("reflectivity outside [0, 255] in " + pointcloudPaths[i]);
            }
        }

        m_poinclouds.emplace_back(PointCloudPtr(new PointCloud(*inCloud)));
    }

    m_cameraInfo.reset(new CameraInfo(m_param.pathToCameraInfo));
    m_histogramHandler.reset(new perception::HistogramHandler(m_param.numBins));
    m_probabilityHandler.reset(new perception::ProbabilityHandler(m_param.numBins));
}

template <typename POINT_CLOUD_TYPE> CalibrationHandler<POINT_CLOUD_TYPE>::~CalibrationHandler()
{
}

template <typename POINT_CLOUD_TYPE>
double CalibrationHandler<POINT_CLOUD_TYPE>::calculateMICost(const TransformInfo& transform)
{
    this->estimateJointProbability(transform, m_param.probabilityEstimatorType);
    return m_probabilityHandler->calculateMICost(m_param.normalizeMI);
}

template <typename POINT_CLOUD_TYPE> int CalibrationHandler<POINT_CLOUD_TYPE>::coObservedPoints() const
{
    return m_histogramHandler->totalPoints();
}

template <typename POINT_CLOUD_TYPE>
cv::Mat CalibrationHandler<POINT_CLOUD_TYPE>::estimateJointProbability(const TransformInfo& transform,
                                                                      const int estimatorType)
{
    m_histogramHandler.reset(new perception::HistogramHandler(m_param.numBins));
    m_probabilityHandler.reset(new perception::ProbabilityHandler(m_param.numBins));

    m_histogramHandler->update<PointCloudType>(m_grayImgs, m_poinclouds, *m_cameraInfo,
                                               perception::toAffine(transform));

    if (estimatorType == ESTIMATOR_JAMES_STEIN) {
        m_probabilityHandler->estimateJS(m_histogramHandler);
    } else {
        m_probabilityHandler->estimateKDE(m_histogramHandler);
    }

    return m_probabilityHandler->jointProb();
}

template <typename POINT_CLOUD_TYPE>
typename CalibrationHandler<POINT_CLOUD_TYPE>::ParameterCovariance
CalibrationHandler<POINT_CLOUD_TYPE>::calculateCRLB(const TransformInfo& transform)
{
    // Fisher information of the kernel density estimate p(X, Y; Theta) (eq. 13): I_ij = sum dp/dTheta_i dp/dTheta_j / p
    // over the histogram cells, the derivatives of p by central differences with the configured gradient steps. The
    // bound (eq. 14) inverts the information of the n co-observed points, n I, which is why the paper's bound shrinks as
    // scans are added (Fig. 7).
    TransformInfo steps;
    steps << m_param.deltaTrans, m_param.deltaTrans, m_param.deltaTrans, m_param.deltaRotRad, m_param.deltaRotRad,
        m_param.deltaRotRad;

    const cv::Mat density = this->estimateJointProbability(transform, ESTIMATOR_KDE);
    const double observations = this->coObservedPoints();

    std::array<cv::Mat, 6> derivatives;
    for (int i = 0; i < 6; ++i) {
        TransformInfo forward = transform;
        TransformInfo backward = transform;
        forward(i) += steps(i);
        backward(i) -= steps(i);
        const cv::Mat forwardDensity = this->estimateJointProbability(forward, ESTIMATOR_KDE);
        const cv::Mat backwardDensity = this->estimateJointProbability(backward, ESTIMATOR_KDE);
        derivatives[i] = (forwardDensity - backwardDensity) / (2.0 * steps(i));
    }

    ParameterCovariance information = ParameterCovariance::Zero();
    for (int r = 0; r < density.rows; ++r) {
        for (int c = 0; c < density.cols; ++c) {
            const double probability = density.at<double>(r, c);
            if (probability <= detail::CRLB_MIN_DENSITY) {
                continue;
            }

            for (int i = 0; i < 6; ++i) {
                for (int j = i; j < 6; ++j) {
                    information(i, j) +=
                        derivatives[i].at<double>(r, c) * derivatives[j].at<double>(r, c) / probability;
                }
            }
        }
    }

    const ParameterCovariance totalInformation =
        observations * ParameterCovariance(information.template selfadjointView<Eigen::Upper>());
    const Eigen::FullPivLU<ParameterCovariance> decomposition(totalInformation);
    if (!decomposition.isInvertible()) {
        return ParameterCovariance::Constant(std::numeric_limits<double>::quiet_NaN());
    }

    return decomposition.inverse();
}

template <typename POINT_CLOUD_TYPE>
typename CalibrationHandler<POINT_CLOUD_TYPE>::DeltaTransformInfo
CalibrationHandler<POINT_CLOUD_TYPE>::step(const double prevCost, const TransformInfo& transform,
                                           const CalibrationHandlerParam& param)
{
    DeltaTransformInfo deltaTransformInfo = DeltaTransformInfo::Zero();
    TransformInfo deltaValues;
    deltaValues << param.deltaTrans, param.deltaTrans, param.deltaTrans, param.deltaRotRad, param.deltaRotRad,
        param.deltaRotRad;

    for (int i = 0; i < deltaValues.size(); ++i) {
        TransformInfo curTransform = transform;
        curTransform(i) += deltaValues(i);
        double curCost = this->calculateMICost(curTransform);
        deltaTransformInfo(i) = (curCost - prevCost) / deltaValues(i);
    }

    deltaTransformInfo.segment(0, 3).normalize();
    deltaTransformInfo.segment(3, 3).normalize();

    DEBUG_LOG("delta transform: x:%f, y:%f, z:%f, r:%f, p:%f, y:%f\n", deltaTransformInfo(0), deltaTransformInfo(1),
              deltaTransformInfo(2), deltaTransformInfo(3), deltaTransformInfo(4), deltaTransformInfo(5));
    return deltaTransformInfo;
}

template <typename POINT_CLOUD_TYPE> TransformInfo CalibrationHandler<POINT_CLOUD_TYPE>::optimize()
{
    // Barzilai-Borwein steepest gradient ascent on a numerical gradient (eqs. 9-11, Algorithm 1). Kept from upstream:
    // translation and rotation take separate unit gradients and step sizes, as their units differ, and a step that
    // lowers the objective is rejected with shrunken finite-difference and step bounds. fork: the bounds shrink on a
    // copy of the parameters, so m_param keeps the configured steps for the CRLB; the first step size is the upper
    // bound (the last transform used to start at zero, which fed the initial pose into eq. 11); the step size is the
    // magnitude of eq. 11, whose sign is negative on a concave objective and was clamped to the lower bound on every
    // iteration; and the search also stops when an accepted update falls below the threshold, as in Algorithm 1.
    CalibrationHandlerParam param = m_param;
    TransformInfo curTransform = m_initialGuess;
    TransformInfo prevTransform = curTransform;
    DeltaTransformInfo prevDeltaTransform = DeltaTransformInfo::Zero(), curDeltaTransform = DeltaTransformInfo::Zero();

    for (std::size_t i = 0; i < param.maxIter; ++i) {
        const double prevCost = this->calculateMICost(curTransform);
        curDeltaTransform = this->step(prevCost, curTransform, param);

        const double gammaTrans =
            std::clamp(detail::barzilaiBorweinStep(curTransform.segment(0, 3) - prevTransform.segment(0, 3),
                                                   curDeltaTransform.segment(0, 3) - prevDeltaTransform.segment(0, 3),
                                                   param.gammaTransU, param.epsilon),
                       param.gammaTransL, param.gammaTransU);
        const double gammaRot =
            std::clamp(detail::barzilaiBorweinStep(curTransform.segment(3, 3) - prevTransform.segment(3, 3),
                                                   curDeltaTransform.segment(3, 3) - prevDeltaTransform.segment(3, 3),
                                                   param.gammaRotU, param.epsilon),
                       param.gammaRotL, param.gammaRotU);

        prevTransform = curTransform;
        prevDeltaTransform = curDeltaTransform;
        curTransform.segment(0, 3) += gammaTrans * curDeltaTransform.segment(0, 3);
        curTransform.segment(3, 3) += gammaRot * curDeltaTransform.segment(3, 3);

        const double curCost = this->calculateMICost(curTransform);
        if (curCost < prevCost) {
            curTransform = prevTransform;
            param.deltaTrans /= param.deltaStepFactor;
            param.deltaRotRad /= param.deltaStepFactor;
            param.gammaRotU /= param.gammaStepFactor;
            param.gammaRotL /= param.gammaStepFactor;
            param.gammaTransU /= param.gammaStepFactor;
            param.gammaTransL /= param.gammaStepFactor;

            if (std::sqrt(param.deltaTrans * param.deltaTrans + param.deltaRotRad * param.deltaRotRad) <
                param.deltaThresh) {
                break;
            }
        } else if ((curTransform - prevTransform).norm() < param.deltaThresh) {
            break;
        }

        printf("iter: %zu, x: %f[m], y: %f[m], z: %f[m], r: %f[deg], p: %f[deg], y: %f[deg], cost: %f, gammaTrans: %f, "
               "gammaRot: %f\n",
               i, curTransform(0), curTransform(1), curTransform(2),
               curTransform(3) * boost::math::double_constants::radian,
               curTransform(4) * boost::math::double_constants::radian,
               curTransform(5) * boost::math::double_constants::radian, prevCost, gammaTrans, gammaRot);
    }

    m_transformation = curTransform;
    return curTransform;
}

template <typename POINT_CLOUD_TYPE>
std::vector<cv::Mat>
CalibrationHandler<POINT_CLOUD_TYPE>::drawPointCloudOnImagePlane(const TransformInfo& transform) const
{
    std::vector<cv::Mat> result;
    result.reserve(m_colorImgs.size());
    for (std::size_t i = 0; i < m_colorImgs.size(); ++i) {
        const auto& colorImg = m_colorImgs[i];
        const auto& curCloud = m_poinclouds[i];
        result.emplace_back(perception::drawPointCloudOnImagePlane<PointCloudType>(colorImg, curCloud, *m_cameraInfo,
                                                                                   perception::toAffine(transform)));
    }

    return result;
}

template <typename POINT_CLOUD_TYPE>
std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>
CalibrationHandler<POINT_CLOUD_TYPE>::projectOnPointCloud(const TransformInfo& transform) const
{
    std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> result;
    result.reserve(m_colorImgs.size());
    for (std::size_t i = 0; i < m_colorImgs.size(); ++i) {
        const auto& colorImg = m_colorImgs[i];
        const auto& curCloud = m_poinclouds[i];
        result.emplace_back(perception::projectOnPointCloud<PointCloudType>(colorImg, curCloud, *m_cameraInfo,
                                                                            perception::toAffine(transform)));
    }

    return result;
}
}  // namespace perception
