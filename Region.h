#ifndef REGION
#define REGION

#include <opencv2/opencv.hpp>
#include <stdlib.h>
#include <cmath>
#include <conio.h>
#include <iostream>
#include <list> 
#include "TQuadRegion.h"

using namespace std;
using namespace cv;

class TRegion
{
    int min, max, avg;
    list<TQuadRegion> regions;

public:

    TRegion(TQuadRegion reg)
    {
        regions.push_front(reg);
        min = reg.min;
        max = reg.max;
        avg = reg.avg;
    }

    void Push_reg(TQuadRegion reg)
    {
        regions.push_back(reg);
        if (min>reg.min) min = reg.min;
        if (max<reg.max) max = reg.max;
        avg = (avg + reg.avg) / 2;
        avg = (max + min) / 2; //------------
    }

    bool Uni_reg(const TQuadRegion &reg, int T, int percent)
    {
        bool f;

        /*int count=0, size=0;
        f=true;
        int m=min, M=max; //------------
        if (m>reg.min) m=reg.min; //
        if (M<reg.max) M=reg.max; //--------
        int newavg=(avg+reg.avg)/2;
        newavg=(M+m)/2;//------------
        list <TQuadRegion> :: iterator curr;
        curr=regions.begin();
        do
        {
        for( int i = 0; i < curr->mat.rows; i++ )
        for( int j = 0; j < curr->mat.cols; j++ )
        if ( abs(intensity(curr->mat.at<Vec3b>(i,j))-newavg)>=T ) count++;//f=false;
        size=size+curr->mat.rows*curr->mat.cols;
        curr++;
        }
        while ( (curr!=regions.end())/*&&(f==true)*/ //);
        /*for( int i = 0; i < reg.mat.rows; i++ )
        for( int j = 0; j < reg.mat.cols; j++ )
        if ( abs(intensity(reg.mat.at<Vec3b>(i,j))-newavg)>=T ) count++;//f=false;
        size=size+reg.mat.rows*reg.mat.cols;
        if (count>size*percent/100) f=false;*/

        f = false;
        int m = min, M = max;
        if (m>reg.min) m = reg.min;
        if (M<reg.max) M = reg.max;
        if (M - m <= T) f = true;

        return f;
    }

    void paint(int r, int g, int b)
    {
        list <TQuadRegion> ::iterator curr;
        curr = regions.begin();
        do
        {
            for (int i = 0; i < curr->mat.rows; i++)
                for (int j = 0; j < curr->mat.cols; j++)
                {
                    curr->mat.at<Vec3b>(i, j)[0] = b;
                    curr->mat.at<Vec3b>(i, j)[1] = g;
                    curr->mat.at<Vec3b>(i, j)[2] = r;
                }
            curr++;
        } while (curr != regions.end());
    }

    bool NeighbourCheck(TRegion reg2)
    {
        bool result = false;
        list <TQuadRegion> ::iterator cur1, cur2;
        for (cur1 = regions.begin(); cur1 != regions.end(); cur1++)
            for (cur2 = reg2.regions.begin(); cur2 != reg2.regions.end(); cur2++)
            {
                if (cur1->NeighbourCheck(*cur2)) result = true;
            }
        return result;
    }

    bool NeighbourCheck(const TQuadRegion &reg2)
    {
        bool result = false;
        list <TQuadRegion> ::iterator cur1;
        for (cur1 = regions.begin(); cur1 != regions.end(); cur1++)
        {
            if (cur1->NeighbourCheck(reg2)) result = true;
            //break;
        }
        return result;
    }

    int intensity(Vec<uchar, 3> pixel)
    {
        return 0.299*pixel[2] + 0.587*pixel[1] + 0.114*pixel[0];
    }

};
#endif