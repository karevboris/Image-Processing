#ifndef TQUADREGION_H_
#define TQUADREGION_H_

#include <opencv2/opencv.hpp>
#include <stdlib.h>
#include <cmath>
#include <conio.h>
#include <iostream>


using namespace std;
using namespace cv;


class TQuadRegion
{
public:

    Mat mat;
    int xl, yl, xr, yr;
    bool isUsed;
    int min, max, avg;

    TQuadRegion(void)
    {
    }

    int intensity(Vec<uchar, 3> pixel)
    {
        return 0.299*pixel[2] + 0.587*pixel[1] + 0.114*pixel[0];
    }

    TQuadRegion(Mat m, int xL, int yL, int xR, int yR)
    {
        isUsed = true;
        mat = m;
        xl = xL;
        yl = yL;
        xr = xR;
        yr = yR;
        min = max = intensity(mat.at<Vec3b>(0, 0));
        avg = 0;
        for (int i = 0; i < mat.rows; i++)
            for (int j = 0; j < mat.cols; j++)
            {
                if (intensity(mat.at<Vec3b>(i, j))>max) max = intensity(mat.at<Vec3b>(i, j));
                if (intensity(mat.at<Vec3b>(i, j))<min) min = intensity(mat.at<Vec3b>(i, j));
                avg = avg + intensity(mat.at<Vec3b>(i, j));
            }
        avg = avg / (mat.rows*mat.cols);
        avg = (max + min) / 2;//---------
    }

    bool Uni_qreg(int T, int percent)
    {
        bool f;

        /*int count=0;
        f=true;
        for( int i = 0; i < mat.rows; i++ )
        for( int j = 0; j < mat.cols; j++ )
        if ( abs(intensity(mat.at<Vec3b>(i,j))-avg)>=T ) count++;//f=false;
        if (count>mat.rows*mat.cols*percent/100) f=false;*/

        f = false; if (max - min <= T) f = true;

        return f;
    }

    void set(Mat m, int xL, int yL, int xR, int yR)
    {
        isUsed = true;
        mat = m;
        xl = xL;
        yl = yL;
        xr = xR;
        yr = yR;
        min = max = intensity(mat.at<Vec3b>(0, 0));
        avg = 0;
        for (int i = 0; i < mat.rows; i++)
            for (int j = 0; j < mat.cols; j++)
            {
                if (intensity(mat.at<Vec3b>(i, j))>max) max = intensity(mat.at<Vec3b>(i, j));
                if (intensity(mat.at<Vec3b>(i, j))<min) min = intensity(mat.at<Vec3b>(i, j));
                avg = avg + intensity(mat.at<Vec3b>(i, j));
            }
        avg = avg / (mat.rows*mat.cols);
        avg = (max + min) / 2;//---------
    }

    void split_reg(TQuadRegion &r1, TQuadRegion &r2, TQuadRegion &r3, TQuadRegion &r4)
    {
        int width, height, halfwidth, halfheight;
        width = xr - xl;
        height = yr - yl;
        halfwidth = (xr - xl) / 2;
        halfheight = (yr - yl) / 2;
        if ((width>1) && (height>1))
        {
            Rect roi1(Point_<int>(0, 0), Point_<int>(halfwidth, halfheight)); //верхн¤¤ лева¤ и права¤ нижн¤¤ точки
            Mat lefttop = mat(roi1);
            r1.set(lefttop, xl, yl, xl + halfwidth, yl + halfheight);

            Rect roi2(Point_<int>(0, halfheight), Point_<int>(halfwidth, height)); //верхн¤¤ лева¤ и права¤ нижн¤¤ точки
            Mat leftbot = mat(roi2);
            r2.set(leftbot, xl, yl + halfheight, xl + halfwidth, yr);

            Rect roi3(Point_<int>(halfwidth, 0), Point_<int>(width, halfheight)); //верхн¤¤ лева¤ и права¤ нижн¤¤ точки
            Mat righttop = mat(roi3);
            r3.set(righttop, xl + halfwidth, yl, xr, yl + halfheight);

            Rect roi4(Point_<int>(halfwidth, halfheight), Point_<int>(width, height)); //верхн¤¤ лева¤ и права¤ нижн¤¤ точки
            Mat rightbot = mat(roi4);
            r4.set(rightbot, xl + halfwidth, yl + halfheight, xr, yr);
        }
        else
        {
            if (width == 1)
            {
                Rect roi3(Point_<int>(halfwidth, 0), Point_<int>(width, halfheight)); //верхн¤¤ лева¤ и права¤ нижн¤¤ точки
                Mat righttop = mat(roi3);
                r3.set(righttop, xl + halfwidth, yl, xr, yl + halfheight);

                Rect roi4(Point_<int>(halfwidth, halfheight), Point_<int>(width, height)); //верхн¤¤ лева¤ и права¤ нижн¤¤ точки
                Mat rightbot = mat(roi4);
                r4.set(rightbot, xl + halfwidth, yl + halfheight, xr, yr);

                r1.isUsed = r2.isUsed = false;
            }
            if (height == 1)
            {
                Rect roi2(Point_<int>(0, halfheight), Point_<int>(halfwidth, height)); //верхн¤¤ лева¤ и права¤ нижн¤¤ точки
                Mat leftbot = mat(roi2);
                r2.set(leftbot, xl, yl + halfheight, xl + halfwidth, yr);

                Rect roi4(Point_<int>(halfwidth, halfheight), Point_<int>(width, height)); //верхн¤¤ лева¤ и права¤ нижн¤¤ точки
                Mat rightbot = mat(roi4);
                r4.set(rightbot, xl + halfwidth, yl + halfheight, xr, yr);

                r1.isUsed = r3.isUsed = false;
            }
        }
    }

    bool NeighbourCheck(const TQuadRegion &reg2)
    {
        bool neighbour = 0;
        if ((xl >= reg2.xl) && (xr <= reg2.xr))
            if ((yr == reg2.yl) || (yl == reg2.yr))
                neighbour = 1;
        if ((yl <= reg2.yl) && (yr >= reg2.yr))
            if ((xr == reg2.xl) || (xl == reg2.xr))
                neighbour = 1;
        if ((reg2.xl >= xl) && (reg2.xr <= xr))
            if ((reg2.yr == yl) || (reg2.yl == yr))
                neighbour = 1;
        if ((reg2.yl <= yl) && (reg2.yr >= yr))
            if ((reg2.xr == xl) || (reg2.xl == xr))
                neighbour = 1;
        return neighbour;
    }

};
#endif