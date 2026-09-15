/**
 * @file    HistogramHandler.hpp
 *
 * @author  btran
 *
 */

#pragma once

#include <memory>
#include <vector>

#include <opencv2/opencv.hpp>

#include <pcl/common/transforms.h>
#include <pcl/point_cloud.h>

#include "Constants.hpp"
#include "Types.hpp"
#include "utils/utils.hpp"

namespace perception
{
/**
 *  @brief joint histogram of the grey value Y and the laser reflectivity X of the points co-observed by the laser
 *  scanner and the camera, accumulated over every scan-image pair (Pandey et al., AAAI 2012, section 2.2)
 */
class HistogramHandler
{
 public:
    using Ptr = std::shared_ptr<HistogramHandler>;

    using JointHistogram = cv::Mat;

    explicit HistogramHandler(int numBins);
    ~HistogramHandler();

    template <typename PointCloudType>
    bool update(const std::vector<cv::Mat>& grayImgs,
                const std::vector<typename pcl::PointCloud<PointCloudType>::Ptr>& inClouds,
                const CameraInfo& cameraInfo, const Eigen::Affine3d& affine = Eigen::Affine3d::Identity());

    template <typename PointCloudType>
    bool update(const cv::Mat& grayImg, const typename pcl::PointCloud<PointCloudType>::Ptr& inCloud,
                const CameraInfo& cameraInfo, const Eigen::Affine3d& affine = Eigen::Affine3d::Identity());

    /**
     *  @brief add one co-observation: a grey value and a reflectivity, both in [0, 255]
     */
    void addSample(int grayValue, double reflectivity);

    int numBins() const
    {
        return m_numBins;
    }

    int totalPoints() const
    {
        return m_totalPoints;
    }

    // rows: grey-value bin, columns: reflectivity bin
    const JointHistogram& jointHist() const
    {
        return m_jointHist;
    }

 private:
    int m_numBins;
    int m_totalPoints;

    JointHistogram m_jointHist;
};
}  // namespace perception
#include "impl/HistogramHandler.ipp"
