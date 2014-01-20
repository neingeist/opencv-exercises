#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>

using namespace cv;

int main()
{
    // Data for visual representation
    int width = 512, height = 512;
    Mat image = Mat::zeros(height, width, CV_8UC3);

    // Set up training data
    Mat labelsMat = (Mat_<float>(9, 1) << 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0);
    Mat trainingDataMat = (Mat_<float>(9, 2) <<
      501, 10, 255, 255, 255, 305, 10, 1, 10, 500, 290, 290, 180, 290, 200, 200, 400, 400);

    assert(labelsMat.rows == trainingDataMat.rows);

    // Set up SVM's parameters
    CvSVMParams params;
    params.svm_type    = CvSVM::C_SVC;
    params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000, 1e-6);
    params.kernel_type = CvSVM::RBF;
    params.gamma = .0001; // for poly/rbf/sigmoid

    params.C = 7; // for CV_SVM_C_SVC, CV_SVM_EPS_SVR and CV_SVM_NU_SVR
    params.nu = 0.0; // for CV_SVM_NU_SVC, CV_SVM_ONE_CLASS, and CV_SVM_NU_SVR
    params.p = 0.0; // for CV_SVM_EPS_SVR

    // Train the SVM
    CvSVM SVM;
    SVM.train(trainingDataMat, labelsMat, Mat(), Mat(), params);

    Vec3b whiteish(200,200,200), blackish (55,55,55);
    // Show the decision regions given by the SVM
    for (int i = 0; i < image.rows; ++i)
        for (int j = 0; j < image.cols; ++j)
        {
            Mat sampleMat = (Mat_<float>(1,2) << j,i);
            float response = SVM.predict(sampleMat);

            if (response == 1)
                image.at<Vec3b>(i,j)  = whiteish;
            else if (response == -1)
                 image.at<Vec3b>(i,j)  = blackish;
        }

    // Show the training data
    int thickness = -1;
    int lineType = 8;
    for (int i = 0; i < trainingDataMat.rows; i++) {
      const CvScalar color = (labelsMat.at<float>(i) == 1) ?
        CV_RGB(255, 255, 255) : CV_RGB(0, 0, 0);
      circle(image, Point(trainingDataMat.at<float>(i, 0), trainingDataMat.at<float>(i, 1)), 5,
          color, thickness, lineType);
    }

    // Show support vectors
    thickness = 2;
    lineType  = 8;
    int c     = SVM.get_support_vector_count();

    for (int i = 0; i < c; ++i)
    {
        const float* v = SVM.get_support_vector(i);
        circle( image,  Point( (int) v[0], (int) v[1]),   6,  Scalar(0, 0, 128), thickness, lineType);
    }

    imwrite("result.png", image);        // save the image

    imshow("SVM Non-Linear Example", image); // show it to the user
    waitKey(0);

}
