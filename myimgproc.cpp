#include <iostream>
#include <opencv2/opencv.hpp>
#include "myimgproc.h"

/*
*  gamma�任����ͼ��Ҷ�ֵ���з����Ա任����ͼ����ع�ǿ�ȵ�������Ӧ��ø��ӽ����۸��ܵ���Ӧ������Ư�ף�����ع⣩��������عⲻ�㣩��ͼƬ�����н���
*       gamma>1, ����������Ҷȱ����죬�ϰ�������Ҷȱ�ѹ���ĸ�����ͼ������䰵��
*       gamma<1, ����������Ҷȱ�ѹ�����ϰ�������Ҷȱ�����Ľ�����ͼ�����������
*/
void gamma_correct(cv::Mat& img, cv::Mat& dst, double gamma)
{
    unsigned char LUT[256];
    for (int i = 0; i < 256; i++)
    {
        float f = (i + 0.5f) / 255;
        f = (float)(pow(f, gamma));
        LUT[i] = cv::saturate_cast<uchar>(f * 255.0f - 0.5f);
    }

    dst = img.clone();
    if (img.channels() == 1)
    {
        cv::MatIterator_<uchar> iterator = dst.begin<uchar>();
        cv::MatIterator_<uchar> iteratorEnd = dst.end<uchar>();
        for (; iterator != iteratorEnd; iterator++)
        {
            *iterator = LUT[(*iterator)];
        }
    }
    else
    {
        cv::MatIterator_<cv::Vec3b> iterator = dst.begin<cv::Vec3b>();
        cv::MatIterator_<cv::Vec3b> iteratorEnd = dst.end<cv::Vec3b>();
        for (; iterator != iteratorEnd; iterator++)
        {
            (*iterator)[0] = LUT[((*iterator)[0])];//b
            (*iterator)[1] = LUT[((*iterator)[1])];//g
            (*iterator)[2] = LUT[((*iterator)[2])];//r
        }
    }
}

/*
*  ͼ��ϸ�����Ǽ���ȡ��:��һ����ͨ����ϸ����һ�����صĿ��
*           �ṩZhang-Suen��GUOHALL�����㷨��opencv�㷨��ժ¼��
*/
static void thinningIteration(cv::Mat img, int iter, int thinningType) {
    cv::Mat marker = cv::Mat::zeros(img.size(), CV_8UC1);

    if (thinningType == THINNING_ZHANGSUEN) {
        for (int i = 1; i < img.rows - 1; i++)
        {
            for (int j = 1; j < img.cols - 1; j++)
            {
                uchar p2 = img.at<uchar>(i - 1, j);
                uchar p3 = img.at<uchar>(i - 1, j + 1);
                uchar p4 = img.at<uchar>(i, j + 1);
                uchar p5 = img.at<uchar>(i + 1, j + 1);
                uchar p6 = img.at<uchar>(i + 1, j);
                uchar p7 = img.at<uchar>(i + 1, j - 1);
                uchar p8 = img.at<uchar>(i, j - 1);
                uchar p9 = img.at<uchar>(i - 1, j - 1);

                int A = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) +
                    (p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) +
                    (p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
                    (p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);
                int B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
                int m1 = iter == 0 ? (p2 * p4 * p6) : (p2 * p4 * p8);
                int m2 = iter == 0 ? (p4 * p6 * p8) : (p2 * p6 * p8);

                if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0)
                    marker.at<uchar>(i, j) = 1;
            }
        }
    }
    if (thinningType == THINNING_GUOHALL) {
        for (int i = 1; i < img.rows - 1; i++)
        {
            for (int j = 1; j < img.cols - 1; j++)
            {
                uchar p2 = img.at<uchar>(i - 1, j);
                uchar p3 = img.at<uchar>(i - 1, j + 1);
                uchar p4 = img.at<uchar>(i, j + 1);
                uchar p5 = img.at<uchar>(i + 1, j + 1);
                uchar p6 = img.at<uchar>(i + 1, j);
                uchar p7 = img.at<uchar>(i + 1, j - 1);
                uchar p8 = img.at<uchar>(i, j - 1);
                uchar p9 = img.at<uchar>(i - 1, j - 1);

                int C = ((!p2) & (p3 | p4)) + ((!p4) & (p5 | p6)) +
                    ((!p6) & (p7 | p8)) + ((!p8) & (p9 | p2));
                int N1 = (p9 | p2) + (p3 | p4) + (p5 | p6) + (p7 | p8);
                int N2 = (p2 | p3) + (p4 | p5) + (p6 | p7) + (p8 | p9);
                int N = N1 < N2 ? N1 : N2;
                int m = iter == 0 ? ((p6 | p7 | (!p9)) & p8) : ((p2 | p3 | (!p5)) & p4);

                if ((C == 1) && ((N >= 2) && ((N <= 3)) & (m == 0)))
                    marker.at<uchar>(i, j) = 1;
            }
        }
    }

    img &= ~marker;
}
void thinning(cv::InputArray input, cv::OutputArray output, int thinningType) {
    cv::Mat processed = input.getMat().clone();
    CV_CheckTypeEQ(processed.type(), CV_8UC1, "");
    // Enforce the range of the input image to be in between 0 - 255
    processed /= 255;

    cv::Mat prev = cv::Mat::zeros(processed.size(), CV_8UC1);
    cv::Mat diff;

    do {
        thinningIteration(processed, 0, thinningType);
        thinningIteration(processed, 1, thinningType);
        absdiff(processed, prev, diff);
        processed.copyTo(prev);
    } while (countNonZero(diff) > 0);

    processed *= 255;

    output.assign(processed);
}

/*
*  ȥë���㷨:��ϸ����ͼ��ȥ��ë��
*       1.����ͼ���ҵ�ϸ��ͼ��Ķ˵㣬�˵�İ����������ֻ��һ������㣻
*       2.����ÿһ���˵㣬ֱ���������ֲ�㣬������仯���ȣ���С���趨��ֵ������������֧���㣻�ֲ��İ������ڵ��������ڵ���3��
*/
int Get8NeighborPt(cv::Mat img, cv::Point point, std::vector<cv::Point>& vPoint)
{
    if (img.at<uchar>(point.y, point.x) == 0)
    {
        return 0;
    }
    vPoint.clear();
    cv::Point point2, point3, point4, point5, point6, point7, point8, point9;

    point2.x = point.x;
    point2.y = point.y - 1;
    if (img.at<uchar>(point2.y, point2.x) == 255)
    {
        vPoint.push_back(point2);
    }
    point3.x = point.x + 1;
    point3.y = point.y - 1;
    if (img.at<uchar>(point3.y, point3.x) == 255)
    {
        vPoint.push_back(point3);
    }
    point4.x = point.x + 1;
    point4.y = point.y;
    if (img.at<uchar>(point4.y, point4.x) == 255)
    {
        vPoint.push_back(point4);
    }
    point5.x = point.x + 1;
    point5.y = point.y + 1;
    if (img.at<uchar>(point5.y, point5.x) == 255)
    {
        vPoint.push_back(point5);
    }
    point6.x = point.x;
    point6.y = point.y + 1;
    if (img.at<uchar>(point6.y, point6.x) == 255)
    {
        vPoint.push_back(point6);
    }
    point7.x = point.x - 1;
    point7.y = point.y + 1;
    if (img.at<uchar>(point7.y, point7.x) == 255)
    {
        vPoint.push_back(point7);
    }
    point8.x = point.x - 1;
    point8.y = point.y;
    if (img.at<uchar>(point8.y, point8.x) == 255)
    {
        vPoint.push_back(point8);
    }
    point9.x = point.x - 1;
    point9.y = point.y - 1;
    if (img.at<uchar>(point9.y, point9.x) == 255)
    {
        vPoint.push_back(point9);
    }

    int vsize = vPoint.size();
    return vsize;
}
void FindEndPt(cv::Mat img, std::vector<cv::Point>& EndPoint)
{
    for (int i = 1; i < img.rows; i++)
    {
        for (int j = 1; j < img.cols; j++)
        {
            cv::Point point;
            std::vector<cv::Point> vPoint;
            point.x = j;
            point.y = i;
            if (Get8NeighborPt(img, point, vPoint) == 1)
            {
                EndPoint.push_back(point);
            }
        }
    }
}
void RemoveBurr(cv::Mat img, cv::Mat& dst, const int& nThresh)
{
    dst = img.clone();
    std::vector<cv::Point> EndPoint;
    FindEndPt(img, EndPoint);
    cv::Point LastPt, NextPt;
    int Lenth = 0;
    bool bFlagEnd = true;

    for (int index = 0; index < EndPoint.size(); index++)
    {
        LastPt = EndPoint[index];
        NextPt = EndPoint[index];
        Lenth = 0;
        bFlagEnd = true;
        std::vector<cv::Point> vBurrPt;
        while (bFlagEnd)
        {
            std::vector<cv::Point> vPoint;
            if (Get8NeighborPt(img, NextPt, vPoint) == 1)
            {
                vBurrPt.push_back(NextPt);
                LastPt = NextPt;
                NextPt = vPoint[0];
                Lenth++;
            }
            else if (Get8NeighborPt(img, NextPt, vPoint) == 2)
            {
                vBurrPt.push_back(NextPt);
                if (LastPt != vPoint[0])
                {
                    LastPt = NextPt;
                    NextPt = vPoint[0];
                }
                else
                {
                    LastPt = NextPt;
                    NextPt = vPoint[1];
                }
                Lenth++;
            }
            else if (Get8NeighborPt(img, NextPt, vPoint) >= 3)
            {
                bFlagEnd = false;
            }
            if (Lenth > nThresh)
            {
                bFlagEnd = false;
            }
        }
        if (Lenth < nThresh)
        {
            for (int IndexBurr = 0; IndexBurr < vBurrPt.size(); IndexBurr++)
            {
                dst.at<uchar>(vBurrPt[IndexBurr].y, vBurrPt[IndexBurr].x) = 0;
            }
        }
    }
}

/*
*  ���߳�ϡ:����ֵͼ���ϵ����߰��ض��㷨��ȡ������
*      ������˹-�տ��㷨��Douglas-Peucker algorithm, DP����
*                1.��������β���������һ��ֱ��,���������㵽��ֱ�ߵľ��룻
*                2.ѡ�����������ֵ��Ƚ�,��������ֵ,�����ֱ�߾������ĵ㱣��,����ֱ�����˵�����ȫ����ȥ��
*                3.�����������ĵ�,����֪���߷ֳ������ִ���,�ظ���1��2����������������,ֱ���޵����ȥ�����õ�������������޲�����ߵ����ꡣ
*      ���Ȳ����㷨��
*                1.�����������߳��ȣ�
*                2.�趨�ȷ�ֵ������ȷֳ��ȣ�
*                3.�ݹ����ȷֳ��ȵ㡣
*/
//������˹-�տ��㷨
double DistancePL(const cv::Point& p0, const cv::Point& p1, const cv::Point& p2)
{
    double area = std::abs(0.5 * (p1.x * p2.y + p2.x * p0.y + p0.x * p1.y - p2.x * p1.y - p0.x * p2.y - p1.x * p0.y));
    double base = std::sqrt(std::pow(p1.x - p2.x, 2) + std::pow(p1.y - p2.y, 2));
    double height = area / base * 2;
    return height;
}
void RecursionReduction(std::list<int>& listKeyPointIndex, const std::vector<cv::Point>& dataPoint, int firstindex, int endindex, const double& thrDistance)
{
    int maxindex = -3;
    double maxdistance = -3;
    for (int i = firstindex; i < endindex; i++)
    {
        double dis = DistancePL(dataPoint[i], dataPoint[firstindex], dataPoint[endindex]);
        if (dis > maxdistance)
        {
            maxdistance = dis;
            maxindex = i;
        }
    }
    if ((maxdistance > thrDistance) && (maxindex != 0))
    {
        listKeyPointIndex.push_back(maxindex);

        RecursionReduction(listKeyPointIndex, dataPoint, firstindex, maxindex, thrDistance);
        RecursionReduction(listKeyPointIndex, dataPoint, maxindex, endindex, thrDistance);
    }
}
bool DouglasPeuckerVacuate(std::vector<cv::Point>& VacuatedPoint, std::vector<cv::Point>& dataPoint, const double& thrDistance)
{
    try
    {
        if (dataPoint.size() < 3) { return false; }
        VacuatedPoint.clear();

        int firstindex = 0;
        int endindex = dataPoint.size() - 1;

        double xf = dataPoint[firstindex].x;
        double yf = dataPoint[firstindex].y;
        double xl = dataPoint[endindex].x;
        double yl = dataPoint[endindex].y;

        while ((abs(xf - xl) < 1e-15) && (abs(yf - yl) < 1e-15))
        {
            endindex--;
        }

        std::list<int> listKeyPointIndex;
        listKeyPointIndex.push_back(firstindex);
        listKeyPointIndex.push_back(endindex);

        RecursionReduction(listKeyPointIndex, dataPoint, firstindex, endindex, thrDistance);

        listKeyPointIndex.sort();
        std::list<int>::iterator iter;
        for (iter = listKeyPointIndex.begin(); iter != listKeyPointIndex.end(); iter++)
        {
            VacuatedPoint.push_back(dataPoint[*iter]);
        }
        return true;
    }
    catch (const std::exception&)
    {
        return false;
    }
}
//���Ȳ����㷨
void KSample(std::vector<cv::Point> dataPoint, std::vector<cv::Point>& samplePoint, int& num_samples)
{
    std::vector<double> dist(dataPoint.size());
    for (int i = 1; i < dataPoint.size(); i++)
    {
        double distance = sqrt(pow(dataPoint[i].x - dataPoint[i - 1].x, 2) + pow(dataPoint[i].y - dataPoint[i - 1].y, 2));
        dist[i] = dist[i - 1] + distance;
    }
    for (int i = 1; i < dataPoint.size(); i++)
    {
        dist[i] /= dist.back();
    }

    std::vector<double> s(num_samples);
    for (int i = 0; i < num_samples; i++)
    {
        s[i] = i * (1.0 / (num_samples - 1));
    }
    samplePoint.resize(num_samples);
    for (int i = 0; i < num_samples; i++)
    {
        std::vector<double>::iterator it = std::lower_bound(dist.begin(), dist.end(), s[i]);
        int j = it - dist.begin();
        if (j > 0 && (j == dist.size() || s[i] - dist[j - 1] < dist[j] - s[i]))
        {
            j--;
        }
        samplePoint[i] = dataPoint[j];
    }
    samplePoint.push_back(dataPoint.back());
}