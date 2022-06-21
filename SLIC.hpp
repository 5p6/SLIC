#pragma once


//������RGB��ͨ���ռ��ͼ�����ʹ��



#include<opencv2/opencv.hpp>
#include<iostream>
#include<vector>
#define MySLIC MySLIC
#define MaxDis 9999999
struct center//����������
{
	int x;//x����
	int y;//y����
	int L;//��һ����
	int A;//�ڶ�����
	int B;//��������
	int label;
};
class SLIC
{
public:
	cv::Mat label;//ͼ���ǩ����
	cv::Mat dis;//ͼ��������
	SLIC(){}
	SLIC(cv::Mat& image);
	
	
	int Initseed(cv::Mat& imageLAB, std::vector<center>& centers,  int size);//��ʼ������������
	int update_pixel(cv::Mat& image, std::vector<center>& centers, int size, int weight);//����ÿ�����صı�ǩ�;���
	int updatecenter(cv::Mat& image, std::vector<center>& centers, int size);//���³���������
};


//SLIC�ཫ�Ӻ����۽���
// 1.��ʼ����������
// 2.����ÿ�����صľ����ǩ
// 3.���¾�������





SLIC::SLIC(cv::Mat& image)  //��ʼ���������ͱ�ǩ����
{
	dis = cv::Mat(image.size(), CV_64FC1, cv::Scalar(MaxDis));
	label = cv::Mat(image.size(), CV_64FC1,cv::Scalar(-1));
}



int SLIC::Initseed(cv::Mat& imageLAB,std::vector<center>& centers, int size)
{
	//sizeΪÿ������ı߳�
	if (imageLAB.empty())
	{
		std::cout << "In itilizeCenters:     image is empty!\n";
		return -1;
	}
	cv::Vec<uchar,3>* p=NULL;
	center cent;
	int num = 0; //��¼��ǩ
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
	//imageΪͼ��centersΪ���������ģ�centerindexΪ�����ر�ǩ��i��jΪͼ�����ص�����ֵ
	cv::Vec<uchar,3>* pixel = image.ptr<cv::Vec<uchar,3>>(i); //������������������ȡ
	float dl = centers[centerindex].L - pixel[j][0];
	float da = centers[centerindex].A - pixel[j][1];
	float db = centers[centerindex].B - pixel[j][2];
	float dx = centers[centerindex].x - i;
	float dy = centers[centerindex].y - j;
	int h_distance = dl * dl + da * da + db * db;
	int xy_distance = dx * dx + dy * dy;
	return h_distance + xy_distance * m;
}


int SLIC::update_pixel(cv::Mat& image, std::vector<center>& centers,int size,int weight) //�޸����ص���������
 {
	if (image.empty() || image.channels() != 3)
	{
		return -1;
	}
//centers's x y l a b label
	int cx, cy, cl, ca, cb, clabel;

	//image's pixel x y l a b
	int x, y, l, a, b;

	//�����볬����֮��ľ���
	

	//distance label pixel pointer
	uchar* pixel=NULL;
	double* disptr=NULL;
	double* labptr=NULL;

	for (int ck = 0; ck < centers.size(); ck++)//�����ر���
	{
		double distance=0;
		cx = centers[ck].x;
		cy = centers[ck].y;
		cl  = centers[ck].L;
		ca = centers[ck].A;
		cb = centers[ck].B;
		clabel = centers[ck].label;

		for (int i = cx-size; i <size+cx; i++)//�����ص�x����
		{
			if (i<0 || i>=image.rows) continue;
		
			  pixel= image.ptr<uchar>(i);//����
			  disptr = dis.ptr<double>(i);//����
			  labptr = label.ptr<double>(i);//��ǩ


			for (int j = cy - size; j < size + cy; j++)//�����ص�y����
			{
				if (j<0 || j>=image.cols) continue;

			        l =  * (pixel + j * 3);
			        a = * (pixel + j * 3 + 1);
			        b = * (pixel + j * 3 + 2);

					distance = getdistance(image, centers, ck, i, j,weight);

			        if (distance<disptr[j])//�������Ͻ����޸�
			        {
			        	disptr[j] = distance;
			        	labptr[j] = ck;
			        }
			 }
			
		}
	}
	return 1;
}



int  SLIC::updatecenter(cv::Mat& image, std::vector<center>& centers, int size) //������������
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
		double sumx = 0, sumy = 0, suml = 0, suma = 0, sumb = 0,sumnum=0;//ȫ��ֵ.
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
	cv::imshow("����", dst);
	cv::waitKey(1);
	return 1;
}

void SLICMeu(cv::Mat& image, int size, int weight)  //��Ҫ���к���
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

