#include <iostream>
#include <fstream>
#include <ctime>
#include <string>
#include <vector>
#include <iterator>
#include <omp.h>
#include <opencv2/opencv.hpp>
#include <stack>
#include <iterator>
#include "Region.h"
#include "TQuadRegion.h"

#define WIDTH 1024
#define HEIGHT 768
#define DEEP 1
#define SIZE WIDTH*HEIGHT*DEEP
#define LEVEL_NUM 8 

#define UNIFORM 1

using namespace std;
using namespace cv;

void uniformQuantization(uchar* data, int width, int height, int deep, int clustNum){
    for (int z = 0; z < deep; z++)
        for (int i = 0; i < height; i++)
            for (int j = 0; j < width; j++){
                data[i*width + j + z*width*height] /= (256.0f / clustNum);
                //data[i*width + j + z*width*height] *= (256.0f / clustNum);
            }
}

void quantizationGreyLevel(uchar* data, int width, int height, int deep, int clustNum, int methodType){
    switch (methodType){
        case UNIFORM:{
            uniformQuantization(data, width, height, deep, clustNum);
        }
        default:
            break;
    }
}

void glcm(uchar* data, int width, int height, int deep, int clustNum, int methodType, float *glcm){
    quantizationGreyLevel(data, width, height, deep, clustNum, methodType);

    for (int i = 0; i < clustNum*clustNum; i++) glcm[i] = 0;

    for (int z = 0; z < deep; z++)
        for (int i = 0; i < height; i++)
            for (int j = 0; j < width - 1; j++){
                int cur = (int)data[i*width + j + z*width*height];
                int next = (int)data[i*width + j + z*width*height + 1];
                glcm[next*clustNum + cur]++;
                glcm[cur*clustNum + next]++;
            }
    /*for (int i = 0; i < clustNum; i++){
        for (int j = 0; j < clustNum; j++){
            cout << " " << glcm[i*clustNum + j] << " ";
        }
        cout << endl;
    }*/
}

void glcmNormalize(float *glcm, int levelNum){
    double sum = 0;
    for (int i = 0; i < levelNum; i++)
        for (int j = 0; j < levelNum; j++){
            sum += glcm[i*levelNum + j];
        }
    if (sum != 0){
        for (int i = 0; i < levelNum; i++)
            for (int j = 0; j < levelNum; j++){
                glcm[i*levelNum + j] /= sum;
            }
    }
}

double getEnergy(float *glcm, int levelNum){
    double energy = 0;
    for (int i = 0; i < levelNum; i++)
        for (int j = 0; j < levelNum; j++){
            energy += glcm[i*levelNum + j] * glcm[i*levelNum + j];
        }
    return energy;
}

double getContrast(float *glcm, int levelNum){
    double contrast = 0;
    for (int i = 0; i < levelNum; i++)
        for (int j = 0; j < levelNum; j++){
            contrast += (i-j)*(i-j)*glcm[i*levelNum + j];
        }
    return contrast;
}

double getHomogeneity(float *glcm, int levelNum){
    double homogeneity = 0;
    for (int i = 0; i < levelNum; i++)
        for (int j = 0; j < levelNum; j++){
            homogeneity += 1 / (1 + (i - j)*(i - j))*glcm[i*levelNum + j];
        }
    return homogeneity;
}

void binarization(){
    uchar* data = new uchar[HEIGHT*WIDTH];
    FILE * file;

    file = fopen("test.tif", "rb");
    fread(data, sizeof(uchar), HEIGHT*WIDTH, file);
    fclose(file);

    Mat img = Mat(HEIGHT, WIDTH, CV_8UC1, data);

    int histogram[256];
    for (int i = 0; i < 256; i++) histogram[i] = 0;
    for (int i = 0; i < img.rows; i++)
        for (int j = 0; j < img.cols; j++)
            histogram[img.at<uchar>(i, j)]++;

    int max = histogram[0], imax = 0;
    for (int i = 1; i < 256; i++)
        if (max < histogram[i])
        {
            max = histogram[i];
            imax = i;
        }

    int min = max, imin = 0;
    for (int i = 0; i<imax; i++)
        if (histogram[i] != 0 && histogram[i+1] != 0)
        {
            min = histogram[i];
            imin = i;
            break;
        }

    float d = 0, dmax = 0;
    int T = 0;
    float sqr = (imax - imin)*(imax - imin) + (max - min)*(max - min);
    for (int i = imin; i < imax; i++)
    {
        d = abs((max - min)*i - (imax - imin)*histogram[i] + imax*min-max*imin) / sqr;
        if (dmax < d) {
            dmax = d;
            T = i;
        }
    }

    for (int i = 0; i < img.rows; i++)
        for (int j = 0; j < img.cols; j++){
            if (img.at<uchar>(i, j) < T) img.at<uchar>(i, j) = 0;
            else img.at<uchar>(i, j) = 255;
        }

    namedWindow("binar", WINDOW_NORMAL);
    imshow("binar", img);
    waitKey();
    delete[] data;
}

void binarization2(){
    uchar* data = new uchar[HEIGHT*WIDTH];
    FILE * file;

    file = fopen("test.tif", "rb");
    fread(data, sizeof(uchar), HEIGHT*WIDTH, file);
    fclose(file);

    Mat img = Mat(HEIGHT, WIDTH, CV_8UC1, data);

    int T = 120;
    for (int i = 0; i < img.rows; i++)
        for (int j = 0; j < img.cols; j++){
            if (img.at<uchar>(i, j) < T) img.at<uchar>(i, j) = 0;
            else img.at<uchar>(i, j) = 255;
        }

    namedWindow("binar2", WINDOW_NORMAL);
    imshow("binar2", img);
    waitKey();
    delete[] data;
}

void segmentation(){
    int Tsplit = 10, Tmerge = 80, percent = 15;

    uchar* data = new uchar[HEIGHT*WIDTH];
    FILE * file;

    file = fopen("test.tif", "rb");
    fread(data, sizeof(uchar), HEIGHT*WIDTH, file);
    fclose(file);

    Mat img = Mat(HEIGHT, WIDTH, CV_8UC1, data);

    stack <TQuadRegion> st_reg;
    list <TQuadRegion> lst_reg;
    list <TRegion> end_reg;
    TQuadRegion f_img(img, 0, 0, img.cols - 1, img.rows - 1);

    st_reg.push(f_img);
    while (!st_reg.empty())
    {
        TQuadRegion r1, r2, r3, r4;
        if (!st_reg.top().Uni_qreg(Tsplit, percent))
        {
            st_reg.top().split_reg(r1, r2, r3, r4);
            st_reg.pop();
            if (r1.isUsed) { if (r1.Uni_qreg(Tsplit, percent)) lst_reg.push_front(r1); else st_reg.push(r1); }
            if (r2.isUsed) { if (r2.Uni_qreg(Tsplit, percent)) lst_reg.push_front(r2); else st_reg.push(r2); }
            if (r3.isUsed) { if (r3.Uni_qreg(Tsplit, percent)) lst_reg.push_front(r3); else st_reg.push(r3); }
            if (r4.isUsed) { if (r4.Uni_qreg(Tsplit, percent)) lst_reg.push_front(r4); else st_reg.push(r4); }
        }
        else
        {
            lst_reg.push_front(st_reg.top());
            st_reg.pop();
        }
    }

    list <TQuadRegion> ::iterator curr;   int i = 0;
    while (!lst_reg.empty())
    {
        int k = 1;
        curr = lst_reg.begin();
        TRegion tmp(*curr);
        //while (k>0)
        {
            //cout<<i<<'\n'; i++;/////////////////////
            k = 0;
            curr = lst_reg.begin();
            curr++;
            while (curr != lst_reg.end())
            {
                if ( /*(tmp.NeighbourCheck(*curr))&&*/(tmp.Uni_reg(*curr, Tmerge, percent)))
                {//cout<<lst_reg.size()<<'\n';
                    tmp.Push_reg(*curr);
                    curr = lst_reg.erase(curr);
                    k++;
                    //curr=lst_reg.begin();
                    //curr++;
                }
                else curr++;
            }
        }
        end_reg.push_front(tmp);
        lst_reg.erase(lst_reg.begin());
    }

    int r, g, b;
    r = g = b = 0;
    list <TRegion> ::iterator finlstcur;
    for (finlstcur = end_reg.begin(); finlstcur != end_reg.end(); finlstcur++)
    {
        r += 120;
        if (r>255) { r = r % 256; g += 120; }
        if (g>255) { g = g % 256; b += 120; }
        if (b>255) b = b % 256;
        finlstcur->paint(r, g, b);
    }

    imshow("My Window", img);
    waitKey();
    delete[] data;
}

void imageProcessing(Mat orig){
    binarization();
    binarization2();
    //segmentation();

    Mat dst;
    medianBlur(orig, dst, 3);
    imshow("Median filter", dst);
    waitKey();

    GaussianBlur(orig, dst, Size(3, 3), 0, 0);
    imshow("Gaussian filter", dst);
    waitKey();
}

int main(){

    uchar* memblock = new uchar[SIZE];
    uchar* quant = new uchar[SIZE];
    float *horMat = new float[LEVEL_NUM*LEVEL_NUM];

    FILE * file;

    file = fopen("testing.tif", "rb");
    if (file == NULL) return -1;
    fread(memblock, sizeof(uchar), SIZE, file);
    fclose(file);

    file = fopen("testing.tif", "rb");
    if (file == NULL) return -1;
    fread(quant, sizeof(uchar), SIZE, file);
    fclose(file);

    cout << "Start" << endl;

    glcm(quant, WIDTH, HEIGHT, DEEP, LEVEL_NUM, UNIFORM, horMat);

    cout << "End" << endl;

    Mat orig = Mat(HEIGHT, WIDTH, CV_8UC1, memblock);
    namedWindow("orig", WINDOW_NORMAL);
    imshow("orig", orig);

    Mat quantum = Mat(HEIGHT, WIDTH, CV_8UC1, quant);
    namedWindow("quantum", WINDOW_NORMAL);
    imshow("quantum", quantum);
    
    waitKey();

    //imageProcessing(orig);

    delete[] memblock, quant;
    return 0;
}