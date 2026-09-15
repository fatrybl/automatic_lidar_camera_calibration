/**
 * @file    ProbabilityHandler.cpp
 *
 * @author  btran
 *
 */

#include <algorithm>
#include <cmath>

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

#include <sensors_calib/ProbabilityHandler.hpp>
#include <sensors_calib/utils/utils.hpp>

namespace
{
constexpr double KERNEL_VARIANCE_FLOOR = 1e-2;  // squared bins; keeps the kernel invertible when a variable is constant
constexpr double KERNEL_RADIUS_IN_STDS = 4.0;

double entropy(const cv::Mat& probability)
{
    double value = 0.0;
    for (auto it = probability.begin<double>(); it != probability.end<double>(); ++it) {
        if (*it > 0.0) {
            value -= *it * std::log(*it);
        }
    }
    return value;
}
}  // namespace

namespace perception
{
ProbabilityHandler::ProbabilityHandler(int numBins)
    : m_numBins(numBins)
    , m_totalPoints(0)
{
    if (m_numBins <= 0 || m_numBins > perception::MAX_BINS) {
        throw std::runtime_error("invalid number of bins");
    }

    this->reset(0);
}

ProbabilityHandler::~ProbabilityHandler()
{
}

bool ProbabilityHandler::estimateKDE(const HistogramHandler::Ptr& histogram)
{
    this->reset(histogram->totalPoints());
    if (m_totalPoints < 2) {
        DEBUG_LOG("need at least two samples");
        return false;
    }

    const cv::Mat& counts = histogram->jointHist();
    const double n = m_totalPoints;
    double sumGray = 0, sumIntensity = 0, sumGrayGray = 0, sumIntensityIntensity = 0, sumGrayIntensity = 0;
    for (int i = 0; i < m_numBins; ++i) {
        for (int j = 0; j < m_numBins; ++j) {
            const double count = counts.at<double>(i, j);
            sumGray += count * i;
            sumIntensity += count * j;
            sumGrayGray += count * i * i;
            sumIntensityIntensity += count * j * j;
            sumGrayIntensity += count * i * j;
        }
    }

    // sample covariance of the observations in bins, ordered (grey value, reflectivity)
    Eigen::Matrix2d covariance;
    covariance(0, 0) = (sumGrayGray - sumGray * sumGray / n) / (n - 1);
    covariance(1, 1) = (sumIntensityIntensity - sumIntensity * sumIntensity / n) / (n - 1);
    covariance(0, 1) = (sumGrayIntensity - sumGray * sumIntensity / n) / (n - 1);
    covariance(1, 0) = covariance(0, 1);

    // kernel covariance Omega Omega^T for the bandwidth matrix Omega = n^(-1/6) Sigma^(1/2)
    const Eigen::Matrix2d kernelCovariance =
        std::pow(n, -1.0 / 3.0) * covariance + KERNEL_VARIANCE_FLOOR * Eigen::Matrix2d::Identity();
    const Eigen::Matrix2d precision = kernelCovariance.inverse();
    const int grayRadius = std::min(
        m_numBins - 1, static_cast<int>(std::ceil(KERNEL_RADIUS_IN_STDS * std::sqrt(kernelCovariance(0, 0)))));
    const int intensityRadius = std::min(
        m_numBins - 1, static_cast<int>(std::ceil(KERNEL_RADIUS_IN_STDS * std::sqrt(kernelCovariance(1, 1)))));

    cv::Mat kernel(2 * grayRadius + 1, 2 * intensityRadius + 1, CV_64FC1);
    for (int di = -grayRadius; di <= grayRadius; ++di) {
        for (int dj = -intensityRadius; dj <= intensityRadius; ++dj) {
            const Eigen::Vector2d offset(di, dj);
            kernel.at<double>(di + grayRadius, dj + intensityRadius) = std::exp(-0.5 * offset.dot(precision * offset));
        }
    }
    kernel /= cv::sum(kernel)[0];

    // the kernel is symmetric, so the correlation of filter2D is the convolution; probability mass that would leave
    // [0, 255] is reflected back at the border
    const cv::Mat frequencies = counts / n;
    cv::filter2D(frequencies, m_jointProb, CV_64F, kernel, cv::Point(-1, -1), 0.0, cv::BORDER_REFLECT);
    m_jointProb /= cv::sum(m_jointProb)[0];
    this->updateMarginals();

    return true;
}

bool ProbabilityHandler::estimateJS(const HistogramHandler::Ptr& histogram)
{
    this->reset(histogram->totalPoints());
    if (m_totalPoints < 2) {
        DEBUG_LOG("need at least two samples");
        return false;
    }

    // lambda = (1 - sum f^2) / ((n - 1) sum (1/K - f)^2) over the K = numBins^2 cells, with sum (1/K - f)^2 = sum f^2 - 1/K
    // fork: the upstream target was the identity matrix / numBins (perfectly correlated variables), applied after the
    // kernel smoothing, which inflated the mutual information by an amount that depends on n
    const double cells = static_cast<double>(m_numBins) * m_numBins;
    const cv::Mat frequencies = histogram->jointHist() / static_cast<double>(m_totalPoints);
    const double sumSquares = frequencies.dot(frequencies);
    const double distanceToTarget = sumSquares - 1.0 / cells;
    const double lambda = distanceToTarget > 0.0
                              ? std::clamp((1.0 - sumSquares) / ((m_totalPoints - 1) * distanceToTarget), 0.0, 1.0)
                              : 1.0;
    m_jointProb = frequencies * (1.0 - lambda) + cv::Scalar(lambda / cells);
    this->updateMarginals();

    return true;
}

double ProbabilityHandler::calculateMICost(const bool normalize) const
{
    // fork: natural logarithm, as in eqs. 2-4 (upstream: log2)
    const double grayEntropy = entropy(m_grayProb);
    const double intensityEntropy = entropy(m_intensityProb);
    const double mutualInformation = grayEntropy + intensityEntropy - entropy(m_jointProb);
    if (!normalize) {
        return mutualInformation;
    }

    const double marginalEntropy = grayEntropy + intensityEntropy;
    return marginalEntropy > 0.0 ? 2.0 * mutualInformation / marginalEntropy : 0.0;
}

void ProbabilityHandler::reset(const int totalPoints)
{
    m_totalPoints = totalPoints;
    m_grayProb = Probability::zeros(m_numBins, 1, CV_64FC1);
    m_intensityProb = Probability::zeros(1, m_numBins, CV_64FC1);
    m_jointProb = JointProbability::zeros(m_numBins, m_numBins, CV_64FC1);
}

void ProbabilityHandler::updateMarginals()
{
    cv::reduce(m_jointProb, m_grayProb, 1, cv::REDUCE_SUM, CV_64F);
    cv::reduce(m_jointProb, m_intensityProb, 0, cv::REDUCE_SUM, CV_64F);
}
}  // namespace perception
