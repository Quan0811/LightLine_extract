#pragma once

void gamma_correct(cv::Mat& img, cv::Mat& dst, double gamma);

enum ThinningTypes
{
    THINNING_ZHANGSUEN = 0, // Thinning technique of Zhang-Suen
    THINNING_GUOHALL = 1  // Thinning technique of Guo-Hall
};
static void thinningIteration(cv::Mat img, int iter, int thinningType);
void thinning(cv::InputArray input, cv::OutputArray output, int thinningType);

int Get8NeighborPt(cv::Mat img, cv::Point point, std::vector<cv::Point>& vPoint);
void FindEndPt(cv::Mat img, std::vector<cv::Point>& EndPoint);
void RemoveBurr(cv::Mat img, cv::Mat& dst, const int& nThresh);

double DistancePL(const cv::Point& p0, const cv::Point& p1, const cv::Point& p2);
void RecursionReduction(std::list<int>& listKeyPointIndex, const std::vector<cv::Point>& dataPoint, int firstindex, int endindex, const double& thrDistance);
bool DouglasPeuckerVacuate(std::vector<cv::Point>& VacuatedPoint, std::vector<cv::Point>& dataPoint, const double& thrDistance);

void KSample(std::vector<cv::Point> dataPoint, std::vector<cv::Point>& samplePoint, int& num_samples);
