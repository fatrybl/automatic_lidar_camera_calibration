/**
 * @file    HistogramHandler.cpp
 *
 * @author  btran
 *
 */

#include <algorithm>

#include <sensors_calib/HistogramHandler.hpp>
#include <sensors_calib/utils/Utility.hpp>

namespace perception
{
HistogramHandler::HistogramHandler(int numBins)
    : m_numBins(numBins)
    , m_totalPoints(0)
{
    if (m_numBins <= 0 || m_numBins > perception::MAX_BINS) {
        throw std::runtime_error("invalid number of bins");
    }

    m_jointHist = JointHistogram::zeros(m_numBins, m_numBins, CV_64FC1);
}

HistogramHandler::~HistogramHandler()
{
}

void HistogramHandler::addSample(const int grayValue, const double reflectivity)
{
    // fork: bins scale the [0, 255] values by numBins / MAX_BINS; the upstream integer bin fraction
    // MAX_BINS / numBins indexed past the histogram whenever numBins did not divide MAX_BINS
    const int grayBin = std::clamp(grayValue, 0, MAX_BINS - 1) * m_numBins / MAX_BINS;
    const int reflectivityBin = std::clamp(static_cast<int>(reflectivity), 0, MAX_BINS - 1) * m_numBins / MAX_BINS;

    m_jointHist.at<double>(grayBin, reflectivityBin)++;
    m_totalPoints++;
}
}  // namespace perception
