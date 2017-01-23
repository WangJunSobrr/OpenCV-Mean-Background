/*
 * opencvSegmentation.cpp
 *
 *  Created on: 7 nov. 2016
 *      Author: tux
 */

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void Morpho(Mat);
const string Date();
const string Heure();

int main()
{
    Mat img0, frame, fgmask, fgimg, bgimg;

	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	vector<int> nbCentroid;

	Point pt1, pt2;

    bool update_bg_model = true;

    string vid = "cars.mp4"; //video et chemin a modifier 
    VideoCapture cap(vid);

    cap >> frame;

    //algo remove background
    Ptr<BackgroundSubtractor> bg_model;
    bg_model = createBackgroundSubtractorMOG2(50);

    while(true)
    {
        cap >> img0;

        resize(img0, frame, Size(640, 640*img0.rows/img0.cols), INTER_LINEAR);
        fgimg.create(frame.size(), frame.type());

        bg_model->apply(frame, fgmask, update_bg_model ? -1 : 0);

        fgimg = Scalar::all(0);
        frame.copyTo(fgimg, fgmask);

        Morpho(fgmask); //segmentation

        pt1 = Point(0, frame.rows - 150);
        pt2 = Point(frame.cols - 1, frame.rows - 100);
        //Rect roi = Rect(pt1, pt2);
        rectangle(frame, pt1, pt2, Scalar(0,0,255), 1);

        //detection des contours
	    findContours(fgmask, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);

		vector<vector<Point> > contours_poly(contours.size());
		vector<Rect> boundRect(contours.size());
		vector<Point2f>center(contours.size());
		vector<float>radius(contours.size());

        //affichage des zones detectees
		for(int i = 0; i < contours.size(); i++)
		{
			approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
			boundRect[i] = boundingRect(Mat(contours_poly[i]));
			minEnclosingCircle((Mat)contours_poly[i], center[i], radius[i]);

			rectangle(frame, boundRect[i].tl(), boundRect[i].br(), Scalar(0,255,0), 2, 8, 0 );
			circle(frame, center[i], 1, Scalar(0,0,255), 2);
		}

        //affichage des informations
		rectangle(frame, Point(1, 5), Point(110,45), Scalar(0,0,0), -2);
		putText(frame, Date(),Point(3,20),1,1,Scalar(255,255,255),1);
		putText(frame, Heure(),Point(3,40),1,1,Scalar(255,255,255),1);

        bg_model->getBackgroundImage(bgimg);
        update_bg_model = !update_bg_model; //mise a jour du modele

        imshow("image", frame);
        imshow("foreground mask", fgmask);
        imshow("foreground image", fgimg);
        imshow("mean background image", bgimg);
        waitKey(32);
    }
    destroyAllWindows();
    return 0;
}

void Morpho(Mat a)
{
	Mat erodeElement = getStructuringElement(MORPH_RECT,Size(5,5)); //a modifier selon la detection voulue
	Mat dilateElement = getStructuringElement(MORPH_RECT,Size(15,15));

	erode(a, a, erodeElement);
	dilate(a, a, dilateElement);
	erode(a, a, erodeElement);
	dilate(a, a, dilateElement);

	threshold(a, a, 150, 255, CV_THRESH_BINARY); //seuil de detection a modifier si besoin
	medianBlur(a, a, 5);
}

const string Date()
{
    time_t     now = time(0);
    struct tm  tstruct;
    char       buf[80];
    tstruct = *localtime(&now);

    strftime(buf, sizeof(buf), "%d/%m/%Y", &tstruct);

    return buf;
}

const string Heure()
{
    time_t     now = time(0);
    struct tm  tstruct;
    char       buf[80];
    tstruct = *localtime(&now);

    strftime(buf, sizeof(buf), "%X", &tstruct);

    return buf;
}



