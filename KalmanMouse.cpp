// http://opencvexamples.blogspot.com/2014/01/kalman-filter-implementation-tracking.html
// (slightly cleaned up, restructured and patched to use opencv's gui functions)

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/video/tracking.hpp"

using namespace cv;
using namespace std;

Point mousePos;

// save mouse position in the global mousePos.
void saveMousePosCallback(int event, int x, int y, int flags, void* userdata) {
     if (event == EVENT_MOUSEMOVE) {
         mousePos.x = x;
         mousePos.y = y;
     }
}

#define ADDNOISE 1

// measures the mouse position by reading from mousePos and adding some
// artificial noise.
Mat_<float> measure() {
    Mat_<float> measurement(2,1);
    Mat_<float> measurementNoise(2,1);

    measurement(0) = mousePos.x;
    measurement(1) = mousePos.y;

#if ADDNOISE == 1
    Mat mean  = Mat::zeros(1,1,CV_64FC1);
    Mat sigma = Mat::ones(1,1,CV_64FC1) * 5;
    randn(measurementNoise, mean, sigma);
    measurement += measurementNoise;
#endif

    return measurement;
}


// draw a cross
void drawCross(Mat img, Point center, Scalar color, int d) {
    line(img, Point(center.x - d, center.y - d),
              Point(center.x + d, center.y + d), color, 2, CV_AA, 0);
    line(img, Point(center.x + d, center.y - d),
              Point(center.x - d, center.y + d), color, 2, CV_AA, 0);
}

Mat img(600, 800, CV_8UC3);
vector<Point> mousev, kalmanv;

void plot() {
        img = Scalar::all(0);


        Point statePt = kalmanv.back();
        Point measPt = mousev.back();
        drawCross(img, statePt, Scalar(255,255,255), 5);
        drawCross(img, measPt, Scalar(0,0,255), 5);


        for (int i = 0; i < mousev.size()-1; i++)
            line(img, mousev[i], mousev[i+1], Scalar(255,255,0), 1);

        for (int i = 0; i < kalmanv.size()-1; i++)
            line(img, kalmanv[i], kalmanv[i+1], Scalar(0,155,255), 1);
}



int main() {
    namedWindow("mouse kalman", 1);
    setMouseCallback("mouse kalman", saveMousePosCallback, NULL);



    // 4 state dimensions: x, y, dx, dy
    // 2 measurement dimensions: x, y
    KalmanFilter KF(4, 2, 0);

    // transition matrix models: x' = x + dx, y' = y + dy, dx' = dx, dy' = dy
    KF.transitionMatrix = *(Mat_<float>(4, 4) << 1,0,1,0,   0,1,0,1,  0,0,1,0,  0,0,0,1);

    setIdentity(KF.measurementMatrix);
    setIdentity(KF.processNoiseCov, Scalar::all(1e-3));
    setIdentity(KF.measurementNoiseCov, Scalar::all(10));
    setIdentity(KF.errorCovPost, Scalar::all(.1));



    while (waitKey(10) < 0) {
        // First predict, to update the internal statePre variable
        Mat prediction = KF.predict();


        // Measure
        Mat_<float> measurement = measure();


        // Update
        Mat_<float> estimated = KF.correct(measurement);


        // Save history
        Point statePt(estimated(0),estimated(1));
        Point measPt(measurement(0),measurement(1));
        mousev.push_back(measPt);
        kalmanv.push_back(statePt);


        // Plot
        plot();
        imshow("mouse kalman", img);
    }
}
