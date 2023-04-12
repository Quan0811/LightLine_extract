/*
* 连通域分析：基于预处理、阈值二值化、开运算、边缘检测操作
*/


#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/imgproc_c.h"

using namespace std;
using namespace cv;

Mat img, smallImg, gray, bw;
vector<Vec4i> hierarchy;
vector<vector<Point> > contours;
int threshval = 220;
Rect r;
Rect maxrect, brect;
int idx, n;
const static Scalar colors[15] = {
    CV_RGB(0,  0,128),
    CV_RGB(0,128,  0),
    CV_RGB(0,128,128),
    CV_RGB(128,  0,  0),
    CV_RGB(128,  0,128),
    CV_RGB(128,128,  0),
    CV_RGB(128,128,128),
    CV_RGB(160,160,160),
    CV_RGB(0,  0,255),
    CV_RGB(0,255,  0),
    CV_RGB(0,255,255),
    CV_RGB(255,  0,  0),
    CV_RGB(255,  0,255),
    CV_RGB(255,255,  0),
    CV_RGB(255,255,255),
};
Scalar color;
//gamma 变换 ，提升图像暗部细节
//Gamma变换是对输入图像灰度值进行的非线性操作，使输出图像灰度值与输入图像灰度值呈指数关系
void gamma_correct(Mat& img, Mat& dst, double gamma) 
{
    unsigned char LUT[256];
    for (int i = 0; i < 256; i++)
    {
        float f = (i + 0.5f) / 255;
        f = (float)(pow(f, gamma));
        LUT[i] = saturate_cast<uchar>(f * 255.0f - 0.5f);
    }

    dst = img.clone();
    if (img.channels() == 1)
    {
        MatIterator_<uchar> iterator = dst.begin<uchar>();
        MatIterator_<uchar> iteratorEnd = dst.end<uchar>();
        for (; iterator != iteratorEnd; iterator++)
        {
            *iterator = LUT[(*iterator)];
        }
    }
    else
    {
        MatIterator_<Vec3b> iterator = dst.begin<Vec3b>();
        MatIterator_<Vec3b> iteratorEnd = dst.end<Vec3b>();
        for (; iterator != iteratorEnd; iterator++)
        {
            (*iterator)[0] = LUT[((*iterator)[0])];//b
            (*iterator)[1] = LUT[((*iterator)[1])];//g
            (*iterator)[2] = LUT[((*iterator)[2])];//r
        }
    }
}


int main() {
    namedWindow("display", 1);
    img = imread("E:/VSProgramData/cv/ShaftCalibration/img/left_draw/left12.bmp", 1);//忽略alpha通道
    //取图片的中间部分其中宽度为0.8，高度为0.66
    r.x = 0;//img.cols / 10
    r.y = 1 * img.rows / 5;//img.rows / 3
    r.width = img.cols * 9 / 10;//img.cols * 8 / 10
    r.height = 1 * img.rows / 5;//img.rows * 2 / 3
    smallImg = img(r);
    cvtColor(smallImg, gray, CV_BGR2GRAY);
    //  medianBlur(gray,gray,5);
    //直方图均衡化，提高图像对比度
    equalizeHist(gray, gray);
    gamma_correct(gray, gray, 4.0);
    imshow("预处理", gray);
    waitKey(0);

    bw = (gray > threshval);
    imshow("阈值化", bw);
    waitKey(0);

    Mat	Structure0 = getStructuringElement(MORPH_RECT, Size(3, 3));//矩形核
    erode(bw, bw, Structure0, Point(-1, -1));
    Mat	Structure1 = getStructuringElement(MORPH_RECT, Size(6, 6));
    dilate(bw, bw, Structure1, Point(-1, -1));
    imshow("开运算", bw);
    waitKey(0);

    findContours(bw, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    if (!contours.empty() && !hierarchy.empty()) {
        idx = 0;
        n = 0;
        vector<Point> approx;
        for (; idx >= 0; idx = hierarchy[idx][0]) {
            color = colors[idx % 15];
            //          drawContours(smallImg,contours,idx,color,1,8,hierarchy);
            approxPolyDP(Mat(contours[idx]), approx, arcLength(Mat(contours[idx]), true) * 0.001, true);//0.005为将毛边拉直的系数
            const Point* p = &approx[0];
            int m = (int)approx.size();
            polylines(smallImg, &p, &m, 1, true, color);
            circle(smallImg, Point(p[0].x, p[0].y), 3, color);
            circle(smallImg, Point(p[1].x, p[1].y), 2, color);
            for (int i = 2; i < m; i++) circle(smallImg, Point(p[i].x, p[i].y), 1, color);
            n++;
            if (1 == n) {
                maxrect = boundingRect(Mat(contours[idx]));
            }
            else {
                brect = boundingRect(Mat(contours[idx]));
                CvRect mr, br;
                mr.x = maxrect.x;
                br.x = brect.x;
                mr.y = maxrect.y;
                br.y = brect.y;
                mr.width = maxrect.width;
                br.width = brect.width;
                mr.height = maxrect.height;
                br.height = brect.height;
                maxrect = cvMaxRect(&mr, &br);
            }
        }
        circle(smallImg, Point(maxrect.x + maxrect.width / 2, maxrect.y + maxrect.height / 2), 2, CV_RGB(255, 0, 0));
    }
    imshow("display", smallImg);
    waitKey(0);
    destroyWindow("display");
    return 0;
}