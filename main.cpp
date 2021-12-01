
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <map>
#include <set>
#include <vector>

using namespace std;
using namespace cv;

struct prop
{
    static Mat colorHSL;

    prop * parent;
    //int r;
    //int g;
    //int b;
    Vec3b bgr;
    //int j;

    prop()
    {
        if(rand()%100 < 30)
        {
            //r = g = b = 255;
            bgr = Vec3b(255, 255, 255);
        }
        else
        {
            int h = colorHSL.at<Vec3b>(0,0)[0];
            int l = colorHSL.at<Vec3b>(0,0)[1] + (rand()%100 < 50? -1 : 1)*(rand()%11);
            //int l = colorHSL.at<Vec3b>(0,0)[1];
            int s = colorHSL.at<Vec3b>(0,0)[2] + (rand()%100 < 50? -1 : 1)*(rand()%11);
            //int s = colorHSL.at<Vec3b>(0,0)[2];
            //Mat hls(1, 1, CV_32FC3, Scalar(h, l, s));
            Mat bgr;
            Mat hls(1, 1, CV_8UC3, Scalar(h, l, s));
            cvtColor(hls, bgr, COLOR_HLS2BGR);
            this->bgr = bgr.at<Vec3b>(0,0);

            //r = rgb.at<Vec3b>(0,0)[0];
            //g = rgb.at<Vec3b>(0,0)[1];
            //b = rgb.at<Vec3b>(0,0)[2];

            //Mat cl = Mat(100, 100, CV_8UC3, Scalar(b, g, r));
            //imshow("", cl);
            //waitKey();

            //r = 0 + rand()%100 < 30 ? 255 : 33 + (rand()%100 < 30? -1 : 1)*(rand()%10);
            //g = 0 + rand()%100 < 30 ? 255 : 70 + (rand()%100 < 30? -1 : 1)*(rand()%10);
            //b = 0 + rand()%100 < 30 ? 255 : 123 + (rand()%100 < 30? -1 : 1)*(rand()%10);
            //this->color = rand()%100 < 30 ? CV_RGB(255,255,255) :CV_RGB(33+r,70+g,123+b);
        }

        parent = 0;
    }

    //prop(int color, int j)
    prop(int r, int g, int b)
    {
        //this->r = r;
        //this->g = g;
        //this->b = b;
        //this->j = j;
        bgr = Vec3b();
        bgr[0] = r;
        bgr[1] = g;
        bgr[2] = b;

        parent = 0;
    }

    prop* par()
    {
        if(parent == 0)
            return this;

        return parent->par();
    }

    bool sib(prop *other)
    {
        return par() == other->par();
    }

    bool zero()
    {
        return (bgr[0]==0 && bgr[1]==0 && bgr[2]==0);
    }
};

Mat prop::colorHSL;

int bintresh = 100;
int w = 500;
int h = 300;
double d = 1000.;
double sep = 200.;
double m1 = 100.;
double m2 = 100.;
int basez = 120;
int divz = 80;
//int grow = 5;
int grow = 0;
//int reduce = 3;
int reduce = 0;
//int smooth = 20;
int smooth = 0;

vector<vector<prop*> > arr;
vector<vector<int> > z;
vector<vector<int> > lz;
vector<vector<int> > rz;

double zbox(double z);
double zcomp(double z);
void propagateLeft(int i, int j);
void propagateRight(int i, int j, int maxz);

set<int> ss;

int main(int argc, char* argv)
{
    Mat colorRGB(1, 1, CV_8UC3, Scalar(123, 70, 33));
    cvtColor(colorRGB, prop::colorHSL, CV_BGR2HLS);

    //Mat test(100, 100, CV_8UC3, Scalar(123, 70, 33));
    //imshow("", test);
    //waitKey();

    //IplImage *depth = cvLoadImage("E:\\Israel\\Trabajos\\Auronix\\Solutions\\Autostereogram\\auronix.jpg", false);
    IplImage *depth = cvLoadImage("E:\\Israel\\Trabajos\\Auronix\\Solutions\\Autostereogram\\auronix3_inv.bmp", false);

    //cvShowImage("depth", depth);
    //cvWaitKey();

    CvSize size = cvGetSize(depth);
    w = size.width;
    h = size.height;

    z = vector<vector<int> >(h, vector<int>(w, basez));

    vector<vector<double> > cpy = vector<vector<double> >(h, vector<double>(w));
    vector<vector<double> > tmp = vector<vector<double> >(h, vector<double>(w));

    for(int i=0; i<h; ++i)
    {
        for(int j=0; j<w; ++j)
        {
            tmp[i][j] = CV_IMAGE_ELEM(depth, uchar, i, j) < bintresh ? 0 : 255;
        }
    }

    for(int k=0; k<grow; k++)
    {
        cpy = tmp;
        for(int i=0; i<h; ++i)
        {
            for(int j=0; j<w; ++j)
            {
                int vt = 0;
                int vb = 0;
                int vl = 0;
                int vr = 0;
                if(i > 0)
                    vt = cpy[i-1][j];
                if(i < h-1)
                    vb = cpy[i+1][j];
                if(j > 0)
                    vl = cpy[i][j-1];
                if(j < w-1)
                    vr = cpy[i][j+1];

                if(vt>0 || vb>0 || vl>0 || vr>0)
                    tmp[i][j] = 255;
            }
        }
    }

    for(int k=0; k<grow; k++)
    {
        cpy = tmp;
        for(int i=0; i<h; ++i)
        {
            for(int j=0; j<w; ++j)
            {
                int vt = 0;
                int vb = 0;
                int vl = 0;
                int vr = 0;
                if(i > 0)
                    vt = cpy[i-1][j];
                if(i < h-1)
                    vb = cpy[i+1][j];
                if(j > 0)
                    vl = cpy[i][j-1];
                if(j < w-1)
                    vr = cpy[i][j+1];

                if(vt==0 || vb==0 || vl==0 || vr==0)
                    tmp[i][j] = 0;
            }
        }
    }

    for(int k=0; k<smooth; k++)
    {
        cpy = tmp;
        for(int i=0; i<h; ++i)
        {
            for(int j=0; j<w; ++j)
            {
                int v = cpy[i][j];
                int vt = 0;
                int vb = 0;
                int vl = 0;
                int vr = 0;
                if(i > 0)
                    vt = cpy[i-1][j];
                if(i < h-1)
                    vb = cpy[i+1][j];
                if(j > 0)
                    vl = cpy[i][j-1];
                if(j < w-1)
                    vr = cpy[i][j+1];

                double vv = v;
                if(vt != vb)
                    vv = (vt + vb)/2;
//                {
//                    double minvv = min(vt, vb);
//                    double maxvv = max(vt, vb);
//                    vv = minvv + (maxvv-minvv)*pow((maxvv - minvv)/maxvv, 1/4.);
//                }

                double vh = v;
                if(vl != vr)
                    vh = (vl + vr)/2;
//                {
//                    double minvh = min(vl, vr);
//                    double maxvh = max(vl, vr);
//                    vh = minvh + (maxvh-minvh)*pow((maxvh - minvh)/maxvh, 1/4.);
//                }

                double c = (vv + vh)/2.;
                //double c = min(vv, vh);
                //double c = vv;

                tmp[i][j] = c;
            }
        }
    }

//    for(int i=0; i<h; ++i)
//    {
//        for(int j=0; j<w; ++j)
//        {
//            double c = tmp[i][j];
//            tmp[i][j] = 255*pow(c/255., 1/2.);
//        }
//    }

    for(int i=0; i<h; ++i)
    {
        for(int j=0; j<w; ++j)
        {
            int s = basez + tmp[i][j]/divz;
            z[i][j] = s;

            ss.insert(s);
        }
    }

    int maxz = 0;
    int minz = 255;
    for(set<int>::iterator it=ss.begin(); it!=ss.end(); ++it)
    {
        cout << *it << endl;
        if(*it > maxz)
            maxz = *it;
        if(*it < minz)
            minz = *it;
    }

    rz = vector<vector<int> >(h, vector<int>(w + maxz, 0));
    lz = vector<vector<int> >(h, vector<int>(w + maxz, maxz));

    for(int i=0; i<h; ++i)
    {
        for(int j=0; j<w+maxz; ++j)
        {
            //int s = basez + CV_IMAGE_ELEM(depth, uchar, i, j)/100;
            //z[i][j] = s;
            int s = j<w? z[i][j] : minz;
            rz[i][j] = s;
            //if(j+maxz < w+maxz)
            //    lz[i][j+maxz] = s;
            if(j+s < w+maxz)
                lz[i][j+s] = min(lz[i][j+s], s);
        }
    }


    //IplImage *lzi = cvCreateImage(cvSize(w, h), IPL_DEPTH_8U, 1);
    Mat lzi = Mat(h, w+maxz, CV_8U, 255);
    //IplImage *rzi = cvCreateImage(cvSize(w, h), IPL_DEPTH_8U, 1);
    Mat rzi = Mat(h, w+maxz, CV_8U, 255);

    for(int i=0; i<h; ++i)
    {
        for(int j=0; j<w+maxz; ++j)
        {
//            //if(arr[i][j]->par()->color==0)
//            {
//                //arr[i][j] = new prop(1+rand()%255, j);
//                //propagateRight(i,j);
//                //propagateLeft(i,j);
//                //arr[i][j] = new prop(0, j);
//                arr[i][j] = new prop(0, j);
//            }

            //CV_IMAGE_ELEM(lzi, char, i, j) = (lz[i][j]-basez)*divz;
            lzi.at<uchar>(i,j) = (lz[i][j]-basez)*divz;
            //CV_IMAGE_ELEM(rzi, char, i, j) = (rz[i][j]-basez)*divz;
            rzi.at<uchar>(i,j) = (rz[i][j]-basez)*divz;
        }
    }

    imshow("lzi", lzi);
    imshow("rzi", rzi);
    //waitKey();

    arr = vector<vector<prop*> >(h, vector<prop*>(w+maxz, (prop*)0));

    //cvSet(image, 255);
    //cvSet(lzi, 255);
    //cvSet(rzi, 255);

    for(int i=0; i<h; ++i)
    {
        for(int j=0; j<w+maxz; ++j)
        {
            //lz[i][j] = rz[i][j] = 128;
            //z[i][j] += basez;
            //arr[i][j] = new prop(rand()%256, j);
            arr[i][j] = new prop(0, 0, 0);
        }
    }

    for(int i=0; i<h; ++i)
    {
        for(int j=0; j<w && j<maxz; ++j)
        {
            //if(z[i][j]<=basez)
            //{
                propagateRight(i,j, maxz);
                //propagateLeft(i,j);
                //arr[i][j] = new prop(0);
            //}
        }
    }

    for(int i=0; i<h; ++i)
    {
        for(int j=0; j<w+maxz; ++j)
        {
            if(arr[i][j]->par()->zero())
            {
                arr[i][j] = new prop();
                propagateRight(i,j, maxz);
            }
        }
    }

//    //prop left
//    for(int i=0; i<h; ++i)
//    {
//        for(int j=0; j<w && j<maxz; ++j)
//        {
//            //if(z[i][j]<=basez)
//            //{
//                //propagateRight(i,j);
//                propagateLeft(i,w+maxz-j-1);
//                //arr[i][j] = new prop(0);
//            //}
//        }
//    }

//    //prop both
//    for(int i=0; i<h; ++i)
//    {
//        for(int j=0; j<w; ++j)
//        {
//            if(z[i][j]<=basez)
//            {
//                propagateRight(i,j);
//                propagateLeft(i,j);
//                //arr[i][j] = new prop(0, j);
//            }
//        }
//    }

    //IplImage *image = cvCreateImage(cvSize(w, h), IPL_DEPTH_8S, 3);
    Mat image = Mat(h, w+maxz, CV_8UC3, Scalar(255, 255, 255));

    for(int i=0; i<h; ++i)
    {
        for(int j=0; j<w+maxz; ++j)
        {
            //CV_IMAGE_ELEM(image, char, i, 3*j) = arr[i][j]->par()->b;
            //CV_IMAGE_ELEM(image, char, i, 3*j+1) = arr[i][j]->par()->g;
            //CV_IMAGE_ELEM(image, char, i, 3*j+2) = arr[i][j]->par()->r;
            image.at<Vec3b>(i,j) = arr[i][j]->par()->bgr;
        }
    }

    imwrite("autostereogram.jpg", image);
    imwrite("autostereogram.png", image);

    imshow("foo", image);
    waitKey();



    return 0;
}

double zcomp(double z)
{
    return d+m2+256.-z;
}

double zbox(double z)
{
    return m2+256.-z;
}

void propagateLeft(int i, int j )
{
    int li = i;
    int lj = j;

    do
    {
        prop *ct = arr[li][lj];

        const int &zz = lz[li][lj];
        //int s = (int)(zbox(zz)*sep/zcomp(zz));
        int s = zz;
        //ri = ri;
        lj -= s;

        if(lj>=0)
        {
            prop *pt = arr[li][lj];

            if(!ct->sib(pt))
            {
                prop *p = new prop();
                //prop *p = new prop(255);
                ct->par()->parent = pt->par()->parent = p;
            }
        }
    }
    while(lj>=0);
}

void propagateRight(int i, int j, int maxz)
{
    int ri = i;
    int rj = j;

    do
    {
        prop *ct = arr[ri][rj];

        const int &zz = rz[ri][rj];
        //int s = (int)(zbox(zz)*sep/zcomp(zz));
        int s = zz;
        //ri = ri;
        rj += s;

        if(rj<w+maxz)
        {
            prop *pt = arr[ri][rj];

            //if(ct->j > pt->par()->j)
            //    continue;

            if(!ct->sib(pt))
            {
                //prop *p = new prop(1+rand()%255, ct->j);
                prop *p = new prop();
                //prop *p = new prop(255);
                ct->par()->parent = pt->par()->parent = p;
            }
        }
    }
    while(rj<w);
}
