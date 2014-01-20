#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace cv;

void
Sharpen (const Mat & myImage, Mat & Result)
{
  CV_Assert (myImage.depth () == CV_8U);	// accept only uchar images

  Result.create (myImage.size (), myImage.type ());
  const int nChannels = myImage.channels ();

  for (int j = 1; j < myImage.rows - 1; ++j)
    {
      const uchar *previous = myImage.ptr < uchar > (j - 1);
      const uchar *current = myImage.ptr < uchar > (j);
      const uchar *next = myImage.ptr < uchar > (j + 1);

      uchar *output = Result.ptr < uchar > (j);

      for (int i = nChannels; i < nChannels * (myImage.cols - 1); ++i)
	{
	  *output++ = saturate_cast < uchar > (5 * current[i]
					       - current[i - nChannels] -
					       current[i + nChannels] -
					       previous[i] - next[i]);
	}
    }

  Result.row (0).setTo (Scalar (0));
  Result.row (Result.rows - 1).setTo (Scalar (0));
  Result.col (0).setTo (Scalar (0));
  Result.col (Result.cols - 1).setTo (Scalar (0));
}

int
main (int argc, char **argv)
{
  if (argc != 2)
    {
      printf ("Usage: SharpenImage image.jpg\n");
      return -1;
    }

  Mat image;
  image = imread (argv[1], 1);

  if (argc != 2 || !image.data)
    {
      printf ("No image data\n");
      return -1;
    }

  Mat result;
  Sharpen (image, result);

  namedWindow ("Sharpen Image", CV_WINDOW_AUTOSIZE);
  // imshow("Sharpen Image", image);
  imshow ("Sharpen Image", result);

  waitKey (0);

  return 0;
}
