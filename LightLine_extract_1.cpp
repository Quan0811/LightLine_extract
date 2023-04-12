/*
* ����ֱ��ͼ���⻯��gamma�任Ԥ����ʹ�����ѡ��ROI����ֵ���������㡢�������ı������������ͼ��ϸ�����Ǽ���ȡ��������ȥë�̡����Ȳ������ߵ�
* ���ѡ�����ο��Ƶ�
* �Ҽ����ɿ��Ƶ㹹���ıպ϶��������
* �м�ȡ�����Ƶ㼯�ϣ�����ѡȡ
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

void Mouse(int event, int x, int y, int flags, void* param);//����������

int main()
{
	src = cv::imread("E:/VSProgramData/cv/ShaftCalibration/img/left_draw/left04.bmp", 1);
	if (src.empty()) { std::cout << "image is empty!" << std::endl; return -1; }

	/*
	*   ROI������ȡ
	*/
	cv::namedWindow("���򴰿�");
	cv::setMouseCallback("���򴰿�", Mouse, (void*)&src);

	while (1)
	{
		tempsrc = src.clone();//��ֹ����ͼ��ճ�� Ҫ��src��ֵ��¡��tempsrc
		if (tempflag == 1)//���
		{
			cv::circle(tempsrc, point, 5, cv::Scalar(0, 255, 0), 2);
			for (int i = 0; i < points.size() - 1; i++)
			{
				cv::circle(tempsrc, points[i], 5, cv::Scalar(0, 255, 0), 2);
				cv::line(tempsrc, points[i], points[i + 1], cv::Scalar(0, 0, 255), 2);
			}
		}
		if (tempflag == 2)//�Ҽ�
		{
			//cv::polylines(tempsrc, points, true, cv::Scalar(0, 0, 255), 2);
			tempsrc = cv::Mat(tempsrc.rows, tempsrc.cols, CV_8UC1, cv::Scalar::all(0));
			cv::fillPoly(tempsrc, points, cv::Scalar(255));
		}
		if (tempflag == 3)//�м�
		{
			if (!points.empty()) { std::cout << "δ���" << std::endl; }
		}
		imshow("���򴰿�", tempsrc);
		if (cv::waitKey(10) == 27)
		{
			cv::destroyWindow("���򴰿�");
			break;
		}
	}

	cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
	//ֱ��ͼ���⻯�����ͼ��Աȶ�
	cv::equalizeHist(gray, gray);
	gamma_correct(gray, gray, 4.0);
	//cv::imshow("Ԥ����", gray);
	//cv::waitKey(0);

	//��λ��
	cv::bitwise_and(gray, tempsrc, dst);
	//cv::imshow("��λ��", dst);
	//cv::waitKey(0);

	//��ֵ��
	bw = (dst > threshval);
	//cv::imshow("��ֵ��ֵ��", bw);
	//cv::waitKey(0);

	//������
	cv::Mat	Structure0 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(6, 6));//���κ�
	erode(bw, bw, Structure0, cv::Point(-1, -1));
	cv::Mat	Structure1 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(6, 6));
	dilate(bw, bw, Structure1, cv::Point(-1, -1));
	//imshow("������", bw);
	//cv::waitKey(0);

	//���������������ģ�ɸѡ����
	std::vector<float> Vec_max;
	std::vector<cv::Point> Pcs;
	std::vector<cv::Vec4i> hierarchy;
	std::vector<std::vector<cv::Point> > contours;
	cv::findContours(bw, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);//ֻ�����Χ������ֻ����յ㵽contours��
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
	//imshow("����", bw);
	//cv::waitKey(0);

	/*
	*  ROI��ϸ��
	*/
	thinning(bw, bw, THINNING_GUOHALL);

	//cv::imwrite("E:/VSProgramData/cv/ShaftCalibration/img/ϸ��ͼ��.jpg", bw);
	//imshow("ϸ��", bw);
	//cv::waitKey(0);

	/*
	*  ����ȥë��
	*/
	cv::Mat dst;
	RemoveBurr(bw,dst,30);
	//cv::imwrite("E:/VSProgramData/cv/ShaftCalibration/img/ȥë��.jpg", dst);
	//imshow("ȥë��", dst);
	//cv::waitKey(0);

	for (int i = 0; i < dst.cols; i++)
	{
		for (int j = 0; j < dst.rows; j++)
		{
			if (dst.at<uchar>(j, i) == 255) { Whitepoint.x = i; Whitepoint.y = j; Whitepoints.push_back(Whitepoint); }
		}
	}
	/*
	*  ���ߵ��ϡ
	*/

	//������˹-�տ��㷨
	//std::vector<cv::Point>VacuatedPoint;
	//double thrDistance = 0.5;
	//DouglasPeuckerVacuate(VacuatedPoint, Whitepoints, thrDistance);

	//�Ȼ�����
	std::vector<cv::Point>VacuatedPoint;
	int K = 50;
	KSample(Whitepoints, VacuatedPoint, K);

	for (int i = 0; i < VacuatedPoint.size() - 1; i++)
	{
		cv::circle(dst, VacuatedPoint[i], 5, cv::Scalar(255), 2);
	}
	imshow("��ϡ", dst);
	cv::waitKey(0);

	return 0;
}

void Mouse(int event, int x, int y, int flags, void* param)
{
	//int event -- �ж��¼� �����Ǽ򵥵��Զ����int��ʽ�ı����� ����CV_EVENT_*����֮һ��һ��ΪsetMouseCallBack�������̶����ݣ����õļ�������Ϊ

	//EVENT_MOUSEMOVE ����		EVENT_LBUTTONDOWN ���
	//EVENT_RBUTTONDOWN �һ�		EVENT_MBUTTONDOWN �м����
	//EVENT_LBUTTONUP ����ſ�		EVENT_RBUTTONUP �Ҽ��ſ�
	//EVENT_MBUTTONUP �м��ſ�		EVENT_LBUTTONDBLCLK ���˫��
	//EVENT_RBUTTONDBLCLK �Ҽ�˫��		EVENT_MBUTTONDBLCLK �м�˫��

	//todoevent����x��y�����������
	//Ϊ��굱ǰ����λ�õ�x����  yΪ��굱ǰ����λ�ӵ�y����

	//todoevent����һ����־λflag

	//todoevent�е�param��������setMouseCallback���ݹ������û����ݣ�Ϊvoidָ������

	printf("��ǰ���λ��x=%d y=%d \r", x, y);

	cv::Mat& src = *(cv::Mat*)param;//��paramǿ��ת��ΪMatָ�룬*(Mat*)=Mat������*��int*��=intһ�� void*Ϊ��������ָ�� ����ָ���κ���ʽ�Ĳ��� ��paramֻ��ָ����
	switch (event)
	{
	case (cv::EVENT_LBUTTONDOWN)://�������������,��¼��ǰ��λ
	{
		tempflag = 1;
		point.x = x;
		point.y = y;
		points.push_back(point);
	}
	break;

	case cv::EVENT_RBUTTONDOWN://�������Ҽ�����
	{
		tempflag = 2;
	}
	break;

	case cv::EVENT_MBUTTONDOWN://�������м�����,�������Ԫ��
	{
		tempflag = 3;
		points.clear();
	}
	break;
	}
}

