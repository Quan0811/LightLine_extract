/*
* 进行直方图均衡化、gamma变换预处理，使用鼠标选择ROI、阈值化、开运算、计算形心保留最长的线条、图像细化（骨架提取）、曲线去毛刺、均匀采样曲线点
* 左键选择多边形控制点
* 右键生成控制点构建的闭合多边形掩码
* 中键取消控制点集合，重新选取
*/

#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui_c.h>

#include "myimgproc.h"

cv::Mat src, dst;
cv::Mat tempsrc, gray, bw;
int threshval = 220;
cv::Point point, Whitepoint;
std::vector<cv::Point> points, Whitepoints;

int tempflag = 0;

void Mouse(int event, int x, int y, int flags, void* param);//鼠标操作函数

int main()
{
	src = cv::imread("E:/VSProgramData/cv/ShaftCalibration/img/left_draw/left04.bmp", 1);
	if (src.empty()) { std::cout << "image is empty!" << std::endl; return -1; }

	/*
	*   ROI初步提取
	*/
	cv::namedWindow("程序窗口");
	cv::setMouseCallback("程序窗口", Mouse, (void*)&src);

	while (1)
	{
		tempsrc = src.clone();//防止绘制图像粘连 要将src的值克隆给tempsrc
		if (tempflag == 1)//左键
		{
			cv::circle(tempsrc, point, 5, cv::Scalar(0, 255, 0), 2);
			for (int i = 0; i < points.size() - 1; i++)
			{
				cv::circle(tempsrc, points[i], 5, cv::Scalar(0, 255, 0), 2);
				cv::line(tempsrc, points[i], points[i + 1], cv::Scalar(0, 0, 255), 2);
			}
		}
		if (tempflag == 2)//右键
		{
			//cv::polylines(tempsrc, points, true, cv::Scalar(0, 0, 255), 2);
			tempsrc = cv::Mat(tempsrc.rows, tempsrc.cols, CV_8UC1, cv::Scalar::all(0));
			cv::fillPoly(tempsrc, points, cv::Scalar(255));
		}
		if (tempflag == 3)//中键
		{
			if (!points.empty()) { std::cout << "未清除" << std::endl; }
		}
		imshow("程序窗口", tempsrc);
		if (cv::waitKey(10) == 27)
		{
			cv::destroyWindow("程序窗口");
			break;
		}
	}

	cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
	//直方图均衡化，提高图像对比度
	cv::equalizeHist(gray, gray);
	gamma_correct(gray, gray, 4.0);
	//cv::imshow("预处理", gray);
	//cv::waitKey(0);

	//按位与
	cv::bitwise_and(gray, tempsrc, dst);
	//cv::imshow("按位与", dst);
	//cv::waitKey(0);

	//阈值化
	bw = (dst > threshval);
	//cv::imshow("阈值二值化", bw);
	//cv::waitKey(0);

	//开运算
	cv::Mat	Structure0 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(6, 6));//矩形核
	erode(bw, bw, Structure0, cv::Point(-1, -1));
	cv::Mat	Structure1 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(6, 6));
	dilate(bw, bw, Structure1, cv::Point(-1, -1));
	//imshow("开运算", bw);
	//cv::waitKey(0);

	//根据轮廓计算形心，筛选亮线
	std::vector<float> Vec_max;
	std::vector<cv::Point> Pcs;
	std::vector<cv::Vec4i> hierarchy;
	std::vector<std::vector<cv::Point> > contours;
	cv::findContours(bw, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);//只检测外围轮廓，只保存拐点到contours中
	if (!contours.empty() && !hierarchy.empty())
	{
		for (int i = 0; i < contours.size(); i++)
		{
			float max = 0;
			cv::Moments M = cv::moments(contours[i]);
			cv::Point Pc(M.m10 / M.m00, M.m01 / M.m00);
			Pcs.push_back(Pc);
			for (int j = 0; j < contours[i].size(); j++)
			{
				cv::Point temp_p = Pc - contours[i][j];
				float tmp = sqrt(temp_p.x * temp_p.x + temp_p.y * temp_p.y);
				if (tmp > max) { max = tmp; }
			}
			Vec_max.push_back(max);
		}
		int maxPosition = max_element(Vec_max.begin(), Vec_max.end()) - Vec_max.begin();
		bw = cv::Mat(bw.rows, bw.cols, CV_8UC1, cv::Scalar::all(0));
		cv::drawContours(bw, contours, maxPosition, cv::Scalar(255), cv::FILLED);
	}
	//imshow("形心", bw);
	//cv::waitKey(0);

	/*
	*  ROI精细化
	*/
	thinning(bw, bw, THINNING_GUOHALL);

	//cv::imwrite("E:/VSProgramData/cv/ShaftCalibration/img/细化图像.jpg", bw);
	//imshow("细化", bw);
	//cv::waitKey(0);

	/*
	*  曲线去毛刺
	*/
	cv::Mat dst;
	RemoveBurr(bw,dst,30);
	//cv::imwrite("E:/VSProgramData/cv/ShaftCalibration/img/去毛刺.jpg", dst);
	//imshow("去毛刺", dst);
	//cv::waitKey(0);

	for (int i = 0; i < dst.cols; i++)
	{
		for (int j = 0; j < dst.rows; j++)
		{
			if (dst.at<uchar>(j, i) == 255) { Whitepoint.x = i; Whitepoint.y = j; Whitepoints.push_back(Whitepoint); }
		}
	}
	/*
	*  曲线点抽稀
	*/

	//道格拉斯-普克算法
	//std::vector<cv::Point>VacuatedPoint;
	//double thrDistance = 0.5;
	//DouglasPeuckerVacuate(VacuatedPoint, Whitepoints, thrDistance);

	//等弧长法
	std::vector<cv::Point>VacuatedPoint;
	int K = 50;
	KSample(Whitepoints, VacuatedPoint, K);

	for (int i = 0; i < VacuatedPoint.size() - 1; i++)
	{
		cv::circle(dst, VacuatedPoint[i], 5, cv::Scalar(255), 2);
	}
	imshow("抽稀", dst);
	cv::waitKey(0);

	return 0;
}

void Mouse(int event, int x, int y, int flags, void* param)
{
	//int event -- 中断事件 并不是简单的自定义的int形式的变量名 而是CV_EVENT_*变量之一，一般为setMouseCallBack传过来固定数据，常用的几个数据为

	//EVENT_MOUSEMOVE 滑动		EVENT_LBUTTONDOWN 左击
	//EVENT_RBUTTONDOWN 右击		EVENT_MBUTTONDOWN 中键点击
	//EVENT_LBUTTONUP 左键放开		EVENT_RBUTTONUP 右键放开
	//EVENT_MBUTTONUP 中键放开		EVENT_LBUTTONDBLCLK 左键双击
	//EVENT_RBUTTONDBLCLK 右键双击		EVENT_MBUTTONDBLCLK 中键双击

	//todoevent中有x和y两个坐标变量
	//为鼠标当前所在位置的x坐标  y为鼠标当前所在位子的y坐标

	//todoevent中有一个标志位flag

	//todoevent中的param用来接收setMouseCallback传递过来的用户数据，为void指针类型

	printf("当前鼠标位置x=%d y=%d \r", x, y);

	cv::Mat& src = *(cv::Mat*)param;//将param强制转换为Mat指针，*(Mat*)=Mat，就如*（int*）=int一样 void*为万能智能指针 可以指向任何形式的参数 而param只是指针名
	switch (event)
	{
	case (cv::EVENT_LBUTTONDOWN)://如果鼠标左键按下,记录当前点位
	{
		tempflag = 1;
		point.x = x;
		point.y = y;
		points.push_back(point);
	}
	break;

	case cv::EVENT_RBUTTONDOWN://如果鼠标右键按下
	{
		tempflag = 2;
	}
	break;

	case cv::EVENT_MBUTTONDOWN://如果鼠标中键按下,清楚所有元素
	{
		tempflag = 3;
		points.clear();
	}
	break;
	}
}

