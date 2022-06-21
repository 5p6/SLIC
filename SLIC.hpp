#pragma once


//仅限于RGB三通道空间的图像聚类使用



#include<opencv2/opencv.hpp>
#include<iostream>
#include<vector>
#define MySLIC MySLIC
#define MaxDis 9999999
struct center//超像素中心
{
	int x;//x坐标
	int y;//y坐标
	int L;//第一分量
	int A;//第二分量
	int B;//第三分量
	int label;
};
class SLIC
{
public:
	cv::Mat label;//图像标签矩阵
	cv::Mat dis;//图像距离矩阵
	SLIC(){}
	SLIC(cv::Mat& image);
	
	
	int Initseed(cv::Mat& imageLAB, std::vector<center>& centers,  int size);//初始化超体素种子
	int update_pixel(cv::Mat& image, std::vector<center>& centers, int size, int weight);//更新每个像素的标签和距离
	int updatecenter(cv::Mat& image, std::vector<center>& centers, int size);//更新超体素中心
};


//SLIC类将子函数聚焦；
// 1.初始化聚类中心
// 2.更新每个体素的聚类标签
// 3.更新聚类中心





SLIC::SLIC(cv::Mat& image)  //初始化距离矩阵和标签矩阵
{
	dis = cv::Mat(image.size(), CV_64FC1, cv::Scalar(MaxDis));
	label = cv::Mat(image.size(), CV_64FC1,cv::Scalar(-1));
}



int SLIC::Initseed(cv::Mat& imageLAB,std::vector<center>& centers, int size)
{
	//size为每个聚类的边长
	if (imageLAB.empty())
	{
		std::cout << "In itilizeCenters:     image is empty!\n";
		return -1;
	}
	cv::Vec<uchar,3>* p=NULL;
	center cent;
	int num = 0; //记录标签
	for (int i = 0; i < imageLAB.rows; i += size)
	{
		cent.x = i + size / 2;
		if (cent.x >= imageLAB.rows) continue;
		p = imageLAB.ptr<cv::Vec<uchar,3>>(cent.x);

		for (int j = 0; j < imageLAB.cols; j += size)
		{
			cent.y = j + size / 2;
			if (cent.y >= imageLAB.cols) continue;
			cent.L = p[cent.y][0];
			cent.A =p[cent.y][1];
			cent.B =p[cent.y][2];
			cent.label = num++;
			centers.push_back(cent);
		}
    }
	return 1;
}



inline double getdistance(cv::Mat& image, const std::vector<center>& centers, int centerindex, int i, int j,int m)
{
	//image为图像，centers为超体素中心，centerindex为超体素标签，i，j为图像体素的行列值
	cv::Vec<uchar,3>* pixel = image.ptr<cv::Vec<uchar,3>>(i); //以三个分量的体素提取
	float dl = centers[centerindex].L - pixel[j][0];
	float da = centers[centerindex].A - pixel[j][1];
	float db = centers[centerindex].B - pixel[j][2];
	float dx = centers[centerindex].x - i;
	float dy = centers[centerindex].y - j;
	int h_distance = dl * dl + da * da + db * db;
	int xy_distance = dx * dx + dy * dy;
	return h_distance + xy_distance * m;
}


int SLIC::update_pixel(cv::Mat& image, std::vector<center>& centers,int size,int weight) //修改像素的属于区域
 {
	if (image.empty() || image.channels() != 3)
	{
		return -1;
	}
//centers's x y l a b label
	int cx, cy, cl, ca, cb, clabel;

	//image's pixel x y l a b
	int x, y, l, a, b;

	//像素与超像素之间的距离
	

	//distance label pixel pointer
	uchar* pixel=NULL;
	double* disptr=NULL;
	double* labptr=NULL;

	for (int ck = 0; ck < centers.size(); ck++)//超像素遍历
	{
		double distance=0;
		cx = centers[ck].x;
		cy = centers[ck].y;
		cl  = centers[ck].L;
		ca = centers[ck].A;
		cb = centers[ck].B;
		clabel = centers[ck].label;

		for (int i = cx-size; i <size+cx; i++)//超像素的x区域
		{
			if (i<0 || i>=image.rows) continue;
		
			  pixel= image.ptr<uchar>(i);//像素
			  disptr = dis.ptr<double>(i);//距离
			  labptr = label.ptr<double>(i);//标签


			for (int j = cy - size; j < size + cy; j++)//超像素的y区域
			{
				if (j<0 || j>=image.cols) continue;

			        l =  * (pixel + j * 3);
			        a = * (pixel + j * 3 + 1);
			        b = * (pixel + j * 3 + 2);

					distance = getdistance(image, centers, ck, i, j,weight);

			        if (distance<disptr[j])//如果距离较近则修改
			        {
			        	disptr[j] = distance;
			        	labptr[j] = ck;
			        }
			 }
			
		}
	}
	return 1;
}



int  SLIC::updatecenter(cv::Mat& image, std::vector<center>& centers, int size) //更新中心体素
 {

	if (image.empty())
	{
		return -1;
	}

    // labelmask's pointer
	double* labptr = NULL;//label's type:CV_64FC1
	//image's pointer
	const uchar* imgptr = NULL;
	//centerpixel's x y;
	int cx, cy;
	for (int ck = 0; ck < centers.size(); ck++)
	{
		double sumx = 0, sumy = 0, suml = 0, suma = 0, sumb = 0,sumnum=0;//全部值.
		cx = centers[ck].x;
		cy = centers[ck].y;

		//centers area  scan
		for (int i = cx - size; i < cx + size; i++)
		{
			if (i<0 || i>=image.rows)continue;

			imgptr = image.ptr<uchar>(i);
			labptr = label.ptr<double>(i);

			for (int j = cy - size; j < cy + size; j++)
			{
				if (j<0 || j>=image.cols)  continue;

			    if (*(labptr + j)== centers[ck].label)//if pixel's label equal to center's label
			    {
			    	suml += *(imgptr + j * 3 );
			    	suma += *(imgptr + j * 3 + 1);
			    	sumb += *(imgptr + j * 3 + 2);
			    	sumx += i;
			    	sumy += j;
			    	sumnum+=1;
			    }

			}
		}

		//centers update
		if (sumnum == 0) continue;

		centers[ck].x = sumx / sumnum;
		centers[ck].y = sumy / sumnum;
		centers[ck].L= suml / sumnum;
		centers[ck].A = suma / sumnum;
		centers[ck].B = sumb/ sumnum;

	} 
	return 1;
}


int showSLIC(cv::Mat& image,cv::Mat& labelMask, std::vector<center>& centers,int size)
{

	cv::Mat dst = cv::Mat::zeros(image.size(),image.type());
	cv::cvtColor(dst, dst, cv::COLOR_BGR2Lab);
	if (labelMask.empty())
	{
		return -1;
	}

	uchar* pixel = NULL;
	double* labptr = NULL;
	int cx, cy;


	for (int ck = 0; ck < centers.size(); ++ck)
	{
		cx = centers[ck].x;
		cy = centers[ck].y;
		for (int i = cx-size; i < cx+size; i++)
		{
			if (i<0 || i>=dst.rows)continue;
			pixel = dst.ptr<uchar>(i);
			labptr = labelMask.ptr<double>(i);

			for (int j = cy-size; j < cy + size; j++)
			{
				if (j<0 || j>=dst.cols) continue;
				if (*(labptr+j) == centers[ck].label)
				{
					*(pixel + j * 3) = centers[ck].L;
					*(pixel + j * 3 +1) = centers[ck].A;
					*(pixel + j * 3 + 2) = centers[ck].B;
				}
			}
		}
	}
	cv::cvtColor(dst, dst, cv::COLOR_Lab2BGR);
	cv::imshow("调试", dst);
	cv::waitKey(1);
	return 1;
}

void SLICMeu(cv::Mat& image, int size, int weight)  //主要运行函数
{
	std::vector<center> centers;
	SLIC* s = new SLIC(image);
	cv::Mat imagelab;
	cv::cvtColor(image, imagelab, cv::COLOR_BGR2Lab);
	s->Initseed(imagelab, centers, size);

	for (int times = 0; times < 20; times++)
	{
		s->update_pixel(imagelab, centers, size, weight);
		
		s->updatecenter(imagelab, centers, size);
	}

	showSLIC(image, s->label, centers, size);

	delete s;

}

