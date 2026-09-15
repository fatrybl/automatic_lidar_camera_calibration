/**
 * @file    ProbabilityHandler.hpp
 *
 * @author  btran
 *
 */

#pragma once

#include <memory>

#include "HistogramHandler.hpp"

namespace perception
{
/**
 *  @brief marginal and joint probabilities of the grey value Y and the reflectivity X, and their mutual information
 *  (Pandey et al., AAAI 2012, eqs. 1-4 and 7)
 */
class ProbabilityHandler
{
 public:
    using Ptr = std::shared_ptr<ProbabilityHandler>;
    using Probability = cv::Mat;
    using JointProbability = cv::Mat;

    explicit ProbabilityHandler(int numBins);
    ~ProbabilityHandler();

    /**
     *  @brief kernel density estimate of the normalised joint histogram (eq. 7): a Gaussian kernel whose bandwidth
     *  matrix is n^(-1/6) Sigma^(1/2), Sigma the sample covariance of the (grey, reflectivity) observations (Scott's
     *  normal reference rule in two dimensions); the marginals are the marginals of this estimate
     */
    bool estimateKDE(const HistogramHandler::Ptr& histogram);

    /**
     *  @brief James-Stein-type shrinkage of the maximum-likelihood cell frequencies towards the uniform distribution
     *  (Hausser and Strimmer 2009), the estimator the paper names as a drop-in alternative
     */
    bool estimateJS(const HistogramHandler::Ptr& histogram);

    /**
     *  @brief mutual information H(X) + H(Y) - H(X, Y) in nats; normalised: 2 MI / (H(X) + H(Y))
     */
    double calculateMICost(const bool normalize = false) const;

    const Probability& intensityProb() const
    {
        return m_intensityProb;
    }

    const Probability& grayProb() const
    {
        return m_grayProb;
    }

    const JointProbability& jointProb() const
    {
        return m_jointProb;
    }

    int totalPoints() const
    {
        return m_totalPoints;
    }

 private:
    void reset(int totalPoints);
    void updateMarginals();

 private:
    Probability m_grayProb;        // numBins x 1
    Probability m_intensityProb;   // 1 x numBins
    JointProbability m_jointProb;  // rows: grey-value bin, columns: reflectivity bin

    int m_numBins;
    int m_totalPoints;
};
}  // namespace perception
