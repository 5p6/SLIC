#pragma once
//超像素均值聚类
#define NEWS NEWS

#include <opencv2/opencv.hpp>
#include<iostream>
#include<vector>
struct center
{
	int x;//column
	int y;//row
	int L;
	int A;
	int B;
	int label;
};


//input parameters:
//imageLAB:    the source image in Lab color space
//DisMask:       it save the shortest distance to the nearest center
//labelMask:   it save every pixel's label  
//centers:       clustering center
//len:         the super pixls will be initialize to len*len
//m:           a parameter witch adjust the weights of the spacial and color space distance 
//
//output:

int clustering(const cv::Mat& imageLAB, cv::Mat& DisMask, cv::Mat& labelMask,
	std::vector<center>& centers, int len, int m)   //更新标签和距离矩阵
{
	if (imageLAB.empty())
	{
		std::cout << "clustering :the input image is empty!\n";
		return -1;
	}

	double* disPtr = NULL;//disMask type: 64FC1
	double* labelPtr = NULL;//labelMask type: 64FC1
	const uchar* imgPtr = NULL;//imageLAB type: 8UC3

	//disc = std::sqrt(pow(L - cL, 2)+pow(A - cA, 2)+pow(B - cB,2))
	//diss = std::sqrt(pow(x-cx,2) + pow(y-cy,2));
	//dis = sqrt(disc^2 + (diss/len)^2 * m^2)
	double dis = 0, disc = 0, diss = 0;
	//cluster center's cx, cy,cL,cA,cB;
	int cx, cy, cL, cA, cB, clabel;
	//imageLAB's  x, y, L,A,B
	int x, y, L, A, B;


	//注：这里的图像坐标以左上角为原点，水平向右为x正方向,水平向下为y正方向，与opencv保持一致
	//      从矩阵行列角度看，i表示行，j表示列，即(i,j) = (y,x)
	for (int ck = 0; ck < centers.size(); ck++)
	{
		cx = centers[ck].x;
		cy = centers[ck].y;
		cL = centers[ck].L;
		cA = centers[ck].A;
		cB = centers[ck].B;
		clabel = centers[ck].label;

		for (int i = cy - len; i < cy + len; i++)
		{
			if (i < 0 || i >= imageLAB.rows) continue;
			//pointer point to the ith row
			imgPtr = imageLAB.ptr<uchar>(i);
			disPtr = DisMask.ptr<double>(i);
			labelPtr = labelMask.ptr<double>(i);
			for (int j = cx - len; j < cx + len; j++)
			{
				if (j < 0 || j >= imageLAB.cols) continue;
				L = *(imgPtr + j * 3);
				A = *(imgPtr + j * 3 + 1);
				B = *(imgPtr + j * 3 + 2);

				disc = std::sqrt(pow(L - cL, 2) + pow(A - cA, 2) + pow(B - cB, 2));
				diss = std::sqrt(pow(j - cx, 2) + pow(i - cy, 2));
				dis = sqrt(pow(disc, 2) + m * pow(diss, 2));

				if (dis < *(disPtr + j))
				{
					*(disPtr + j) = dis;
					*(labelPtr + j) = clabel;
				}//end if
			}//end for
		}
	}//end for (int ck = 0; ck < centers.size(); ++ck)


	return 0;
}



//input parameters:
//imageLAB:    the source image in Lab color space
//labelMask:    it save every pixel's label
//centers:       clustering center
//len:         the super pixls will be initialize to len*len
//
//output:

int updateCenter(cv::Mat& imageLAB, cv::Mat& labelMask, std::vector<center>& centers, int len) //升级聚类中心
{
	double* labelPtr = NULL;//labelMask type: 64FC1
	const uchar* imgPtr = NULL;//imageLAB type: 8UC3
	int cx, cy;

	for (int ck = 0; ck < centers.size(); ++ck)
	{
		double sumx = 0, sumy = 0, sumL = 0, sumA = 0, sumB = 0, sumNum = 0;
		cx = centers[ck].x;
		cy = centers[ck].y;
		for (int i = cy - len; i < cy + len; i++)
		{
			if (i < 0 || i >= imageLAB.rows) continue;
			//pointer point to the ith row
			imgPtr = imageLAB.ptr<uchar>(i);
			labelPtr = labelMask.ptr<double>(i);
			for (int j = cx - len; j < cx + len; j++)
			{
				if (j < 0 || j >= imageLAB.cols) continue;

				if (*(labelPtr + j) == centers[ck].label)
				{
					sumL += *(imgPtr + j * 3);
					sumA += *(imgPtr + j * 3 + 1);
					sumB += *(imgPtr + j * 3 + 2);
					sumx += j;
					sumy += i;
					sumNum += 1;
				}//end if
			}
		}
		//update center
		if (sumNum == 0) sumNum = 0.000000001;
		centers[ck].x = sumx / sumNum;
		centers[ck].y = sumy / sumNum;
		centers[ck].L = sumL / sumNum;
		centers[ck].A = sumA / sumNum;
		centers[ck].B = sumB / sumNum;

	}//end for

	return 0;
}


int showSLICResult(const cv::Mat& image, cv::Mat& labelMask, std::vector<center>& centers, int len)  //展示 
{
	cv::Mat dst = cv::Mat::zeros(image.size(), image.type());
	cv::cvtColor(dst, dst, cv::COLOR_BGR2Lab);
	double* labelPtr = NULL;//labelMask type: 32FC1
	uchar* imgPtr = NULL;//image type: 8UC3

	int cx, cy;
	double sumx = 0, sumy = 0, sumL = 0, sumA = 0, sumB = 0, sumNum = 0.00000001;
	for (int ck = 0; ck < centers.size(); ++ck)
	{
		cx = centers[ck].x;
		cy = centers[ck].y;

		for (int i = cy - len; i < cy + len; i++)
		{
			if (i < 0 || i >= image.rows) continue;
			//pointer point to the ith row
			imgPtr = dst.ptr<uchar>(i);
			labelPtr = labelMask.ptr<double>(i);
			for (int j = cx - len; j < cx + len; j++)
			{
				if (j < 0 || j >= image.cols) continue;

				if (*(labelPtr + j) == centers[ck].label)
				{
					*(imgPtr + j * 3) = centers[ck].L;
					*(imgPtr + j * 3 + 1) = centers[ck].A;
					*(imgPtr + j * 3 + 2) = centers[ck].B;
				}//end if
			}
		}
	}//end for

	cv::cvtColor(dst, dst, cv::COLOR_Lab2BGR);
	cv::namedWindow("showSLIC", 0);
	cv::imshow("showSLIC", dst);
	cv::waitKey(1);

	return 0;
}


int showSLICResult2(const cv::Mat& image, cv::Mat& labelMask, std::vector<center>& centers, int len)  //展示二
{
	cv::Mat dst = image.clone();
	//cv::cvtColor(dst, dst, cv::COLOR_Lab2BGR);
	double* labelPtr = NULL;//labelMask type: 32FC1
	double* labelPtr_nextRow = NULL;//labelMask type: 32FC1
	uchar* imgPtr = NULL;//image type: 8UC3

	for (int i = 0; i < labelMask.rows - 1; i++)
	{
		labelPtr = labelMask.ptr<double>(i);
		imgPtr = dst.ptr<uchar>(i);
		for (int j = 0; j < labelMask.cols - 1; j++)
		{
			//if left pixel's label is different from the right's 
			if (*(labelPtr + j) != *(labelPtr + j + 1))
			{
				*(imgPtr + 3 * j) = 0;
				*(imgPtr + 3 * j + 1) = 0;
				*(imgPtr + 3 * j + 2) = 0;
			}

			//if the upper pixel's label is different from the bottom's 
			labelPtr_nextRow = labelMask.ptr<double>(i + 1);
			if (*(labelPtr_nextRow + j) != *(labelPtr + j))
			{
				*(imgPtr + 3 * j) = 0;
				*(imgPtr + 3 * j + 1) = 0;
				*(imgPtr + 3 * j + 2) = 0;
			}
		}
	}

	//show center
	for (int ck = 0; ck < centers.size(); ck++)
	{
		imgPtr = dst.ptr<uchar>(centers[ck].y);
		*(imgPtr + centers[ck].x * 3) = 100;
		*(imgPtr + centers[ck].x * 3 + 1) = 100;
		*(imgPtr + centers[ck].x * 3 + 1) = 10;
	}

	cv::namedWindow("showSLIC2", 0);
	cv::imshow("showSLIC2", dst);
	cv::waitKey(1);
	return 0;
}


int initilizeCenters(cv::Mat& imageLAB, std::vector<center>& centers, int len)  //初始化聚类中心
{
	if (imageLAB.empty())
	{
		std::cout << "In itilizeCenters:     image is empty!\n";
		return -1;
	}

	uchar* ptr = NULL;
	center cent;
	int num = 0;
	for (int i = 0; i < imageLAB.rows; i += len)
	{
		cent.y = i + len / 2;
		if (cent.y >= imageLAB.rows) continue;
		ptr = imageLAB.ptr<uchar>(cent.y);
		for (int j = 0; j < imageLAB.cols; j += len)
		{
			cent.x = j + len / 2;
			if ((cent.x >= imageLAB.cols)) continue;
			cent.L = *(ptr + cent.x * 3);
			cent.A = *(ptr + cent.x * 3 + 1);
			cent.B = *(ptr + cent.x * 3 + 2);
			cent.label = ++num;
			centers.push_back(cent);
		}
	}
	return 0;
}


//if the center locates in the edges, fitune it's location.




//input parameters:
//image:    the source image in RGB color space
//resultLabel:     it save every pixel's label
//len:         the super pixls will be initialize to len*len
//m:           a parameter witch adjust the weights of diss 
//output:

int SLIC(cv::Mat& image, cv::Mat& resultLabel, std::vector<center>& centers, int len, int m)  //主要运行函数
{
	if (image.empty())
	{
		std::cout << "in SLIC the input image is empty!\n";
		return -1;

	}

	int MAXDIS = 999999;


	//convert color
	cv::Mat imageLAB;
	cv::cvtColor(image, imageLAB, cv::COLOR_BGR2Lab);

	//initiate
	//std::vector<center> centers;
	//disMask save the distance of the pixels to center;
	cv::Mat disMask;
	//labelMask save the label of the pixels
	cv::Mat labelMask = cv::Mat::zeros(image.size(), CV_64FC1);

	//initialize centers,  get centers
	initilizeCenters(imageLAB, centers, len);
	for (auto it = centers.begin(); it != centers.end(); it++)
	{
		std::cout << it->label << " " << it->L << " " << it->A << " " <<
			it->B << " " << it->x << " " << it->y << " " << "\n";
	}
	system("pause");
	system("cls");
	//update cluster 10 times 
	for (int time = 0; time < 15; time++)
	{
		//clustering
		
		disMask = cv::Mat(image.size(), CV_64FC1, cv::Scalar(MAXDIS));
		clustering(imageLAB, disMask, labelMask, centers, len, m);
		for (auto it = centers.begin(); it != centers.end(); it++)
		{
			std::cout << it->label << " " << it->L << " " << it->A << " " <<
				it->B << " " << it->x << " " << it->y << " " << "\n";
		}
		system("pause");
		system("cls");


		//update
		updateCenter(imageLAB, labelMask, centers, len);

	}

	for (auto it = centers.begin(); it != centers.end(); it++)
	{
		std::cout << it->label << " " << it->L << " " << it->A << " " <<
			it->B << " " << it->x << " " << it->y << " " << "\n";
	}



	resultLabel = labelMask;

	return 0;
}


/*int SLIC_Demo()
{

	std::string imagePath = "D:/picure/class1/moon.tif";
	cv::Mat image = cv::imread(imagePath);
	//	cv::Mat image(cv::Size(10,10),CV_8UC3);
	//	cv::randn(image, 150, 50);
	cv::Mat labelMask;//save every pixel's label
	cv::Mat dst;//save the shortest distance to the nearest centers
	std::vector<center> centers;//clustering centers

	int len = 15;//the scale of the superpixel ,len*len
	int m = 10;//a parameter witch adjust the weights of spacial distance and the color space distance
	SLIC(image, labelMask, centers, len, m);

	cv::namedWindow("image", 1);
	cv::imshow("image", image);
	showSLICResult(image, labelMask, centers, len);
	showSLICResult2(image, labelMask, centers, len);

	cv::waitKey(0);
	return 0;
}
*/

