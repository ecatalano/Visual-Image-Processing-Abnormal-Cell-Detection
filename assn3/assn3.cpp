#include <iostream>
using namespace std;

#include <stdio.h>
#include <math.h>
#include <fstream>

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/ml/ml.hpp"

using namespace cv;
using namespace cv::ml;


Mat getVerticalSobel(Mat image, string name){

	Mat verticalSobel = imread(name);
	//now to create the vertical sobel mask..

	//vertical mask is:
	//-1 0 1
	//-2 0 2
        //-1 0 1

	//for this mask, will arrange pixels as follows:
	//1 2 3
        //4 5 6
        //7 8 9
	//where 5 is the pixel to which this mask is being applied.

	//iteration-wise, the image will be 
	//0,0 0,1 0,2 0,3 0,4
        //1,0 1,1 1,2 1,3 1,4
        //2,0 2,1 2,2 2,3 2,4
	
	for (int i = 0; i < image.rows; i++){
        	for (int j = 0; j < image.cols; j++){
			int pixel1;
			int pixel2;
			int pixel3;
			int pixel4;
			int pixel5;
			int pixel6;
			int pixel7;
			int pixel8;
			int pixel9;

			if(i == 0){
				pixel1 = 0;
				pixel2 = 0;
				pixel3 = 0;
			}
			else{
				pixel1 = image.at<cv::Vec3b>(i-1,j-1)[0] * -1;
				pixel2 = image.at<cv::Vec3b>(i-1,j)[0] * 0;
				pixel3 = image.at<cv::Vec3b>(i-1,j+1)[0] * 1;
			}

			if(j == 0){
				pixel4 = 0;
			}
			else{
				pixel4 = image.at<cv::Vec3b>(i,j-1)[0] * -1;
			}
			pixel5 = image.at<cv::Vec3b>(i,j)[0] * 0;
			pixel6 = image.at<cv::Vec3b>(i,j+1)[0] * 1;
			if(j == 0){
				pixel7 = 0;
			}
			else{
				pixel7 = image.at<cv::Vec3b>(i+1,j-1)[0] * -1;
			}

			pixel8 = image.at<cv::Vec3b>(i+1,j)[0] * 0;
			pixel9 = image.at<cv::Vec3b>(i+1,j+1)[0] * 1;


			int sum = (pixel1 + pixel2 + pixel3 + pixel4 + pixel5 + pixel6 + pixel7 + pixel8 + pixel9);
			if(sum > 255) sum = 255;
			if(sum < 0) sum = 0;


			verticalSobel.at<Vec3b>(i,j)[0] = abs(sum);
			verticalSobel.at<Vec3b>(i,j)[1] = abs(sum);
			verticalSobel.at<Vec3b>(i,j)[2] = abs(sum);
		}
	}
	return verticalSobel;
}

Mat getHorizontalSobel(Mat image, string name){

	Mat horizontalSobel = imread(name);
	//now for the horizontal mask..
	//the horizontal mask should be:
	//-1 -2 -1
	// 0  0  0
        // 1  2  1
	
	for (int i = 1; i < image.rows; i++){
        	for (int j = 1; j < image.cols; j++){
			int pixel1 = image.at<cv::Vec3b>(i-1,j-1)[0] * -1;
			int pixel2 = image.at<cv::Vec3b>(i-1,j)[0] * -2;
			int pixel3 = image.at<cv::Vec3b>(i-1,j+1)[0] * -1;

			int pixel4 = image.at<cv::Vec3b>(i,j-1)[0] * 0;
			int pixel5 = image.at<cv::Vec3b>(i,j)[0] * 0;
			int pixel6 = image.at<cv::Vec3b>(i,j+1)[0] * 0;

			int pixel7 = image.at<cv::Vec3b>(i+1,j-1)[0] * 1;
			int pixel8 = image.at<cv::Vec3b>(i+1,j)[0] * 2;
			int pixel9 = image.at<cv::Vec3b>(i+1,j+1)[0] * 1;

			int sum = pixel1 + pixel2 + pixel3 + pixel4 + pixel5 + pixel6 + pixel7 + pixel8 + pixel9;
			if(sum > 255) sum = 255;

			horizontalSobel.at<Vec3b>(i,j)[0] = abs(sum);
			horizontalSobel.at<Vec3b>(i,j)[1] = abs(sum);
			horizontalSobel.at<Vec3b>(i,j)[2] = abs(sum);
		}
	}
	return horizontalSobel;
}

Mat addSobel(Mat verticalSobel, Mat horizontalSobel, string name){
	Mat sobel = imread(name);
	//now to get G from gy and gx..
	for (int i = 0; i < sobel.rows; i++){
        	for (int j = 0; j < sobel.cols; j++){
			
			int gy = verticalSobel.at<Vec3b>(i,j)[0];
			int gx = horizontalSobel.at<Vec3b>(i,j)[0];

			int sum = sqrt((gy * gy) + (gx * gx));
			//int sum = gy + gx;

			sobel.at<Vec3b>(i,j)[0] = abs(sum);
			sobel.at<Vec3b>(i,j)[1] = abs(sum);
			sobel.at<Vec3b>(i,j)[2] = abs(sum);
		}
	}
	return sobel;
}

Mat getGradient(Mat verticalSobel, Mat horizontalSobel, string name){

	Mat gradient = imread(name);

	float max = 0;
	float min = 255;
	for (int i = 0; i < gradient.rows; i++){
        	for (int j = 0; j < gradient.cols; j++){
			int gy = verticalSobel.at<Vec3b>(i,j)[0];
			int gx = horizontalSobel.at<Vec3b>(i,j)[0];

			float direction = atan2(gy, gx);

			if(direction > max) max = direction;
			if(direction < min) min = direction;
	
			/*gradient.at<Vec3b>(i,j)[0] = (int)abs(direction);
			gradient.at<Vec3b>(i,j)[1] = (int)abs(direction);
			gradient.at<Vec3b>(i,j)[2] = (int)abs(direction);*/

		}
	}

	for (int i = 0; i < gradient.rows; i++){
        	for (int j = 0; j < gradient.cols; j++){
			int gy = verticalSobel.at<Vec3b>(i,j)[0];
			int gx = horizontalSobel.at<Vec3b>(i,j)[0];

			float direction = atan2(gy, gx);

			
			float normalized = 360 * (direction) / (max);
			if(normalized > 180) normalized -= 180;
			//printf("Direction: %f Normalized: %f \n", direction, normalized);

			gradient.at<Vec3b>(i,j)[0] = (int)abs(normalized);
			gradient.at<Vec3b>(i,j)[1] = (int)abs(normalized);
			gradient.at<Vec3b>(i,j)[2] = (int)abs(normalized);

	

		}
	}

	//printf("Max: %f Min: %f\n", max, min); 

	
	return gradient;
}

Mat smoothImage(Mat image, string name){
	//smooth image via gaussian filter

	Mat smoothImage = imread(name);
	double kernel[5][5];
	double stdv = .5;
    	double r = 2.0 * stdv * stdv;
	double s = 2.0 * stdv * stdv;  // Assigning standard deviation to 1.0
    	double sum = 0.0;   // Initialization of sun for normalization
    	for (int x = -2; x <= 2; x++) // Loop to generate 5x5 kernel
    	{
     		for(int y = -2; y <= 2; y++)
     	   	{
     	    		r = sqrt(x*x + y*y);
     	         	kernel[x + 2][y + 2] = (exp(-(r*r)/s))/(M_PI * s);
     	         	sum += kernel[x + 2][y + 2];
    	    	}
  	 }
 
   	for(int i = 0; i < 5; ++i){ // Loop to normalize the kernel
      		for(int j = 0; j < 5; ++j){
            		kernel[i][j] /= sum;
		}
	}

	//Iterate through image in 5x5 blocks
	for(int i = 0; i < image.rows; i+=5){
        	for (int j = 0; j < image.cols; j+=5){
			//iterate through the block
			//iterate through every pixel in the block
            		for(int x = 0; x < 5; x++){
				for(int z = 0; z < 5; z++){
					double sum = 0;
					int r = i+x-2;
					int c = j+z-2;
					//iterate through gaussian kernel
					for(int q = 0; q < 5; q++){
						for(int w = 0; w < 5; w++){
							double pixel = 0;
							if(r < 0 || c < 0 || r > image.rows || c > image.cols){
								sum +=0;
							}
							else{	
								int pixel = image.at<Vec3b>(r,c)[0];
								sum+= pixel * kernel[q][w];								//printf("Sum is: %lf\n", sum);
							}
							c++;
						}
						r++;
					}

					//printf("SUM IS: %lf\n original value: %d\n", sum, image.at<Vec3b>(x,z)[0]);
					if(i+x < image.rows && j+z < image.cols){
						smoothImage.at<Vec3b>(i+x,j+z)[0] = sum;
						smoothImage.at<Vec3b>(i+x,j+z)[1] = sum;
						smoothImage.at<Vec3b>(i+x,j+z)[2] = sum;
					}

				}
			}
		}
		//printf("Still iterating...\n");
	}
	return smoothImage;
}
Mat makeGrey(Mat image, string name){
	Mat greyImage = imread(name);
	for(int i = 0; i < image.rows; i++){
		for(int j = 0; j < image.cols; j++){
			greyImage.at<Vec3b>(i,j)[0] = image.at<Vec3b>(i,j)[0] * .299 + image.at<Vec3b>(i,j)[1] * .587 + image.at<Vec3b>(i,j)[2] * .114;
			greyImage.at<Vec3b>(i,j)[1] = image.at<Vec3b>(i,j)[0] * .299 + image.at<Vec3b>(i,j)[1] * .587 + image.at<Vec3b>(i,j)[2] * .114;
			greyImage.at<Vec3b>(i,j)[2] = image.at<Vec3b>(i,j)[0] * .299 + image.at<Vec3b>(i,j)[1] * .587 + image.at<Vec3b>(i,j)[2] * .114;
		}
	}
	return greyImage;
}

Mat combineForHistogram(Mat sobel, Mat gradient, string name)
{
	//In this function, I will encode the image to reflect both the gradient and the edges...
	Mat combinedImage = imread(name);
	for(int i = 0; i < combinedImage.rows;i++){
		for(int j = 0; j < combinedImage.cols;j++){
			combinedImage.at<Vec3b>(i,j)[1] = sobel.at<Vec3b>(i,j)[0];
			combinedImage.at<Vec3b>(i,j)[0] = gradient.at<Vec3b>(i,j)[0];
			combinedImage.at<Vec3b>(i,j)[2] = 0;
		}
	}
	return combinedImage;
}

void collectHistogramData1(Mat combinedImage, string name, int constBlock){
	ofstream abnormalFile;
	ofstream otherFile;
	
	int blockSize = constBlock;

	int abnormalCount = 0;
	int otherCount = 0;	

	int run = 1;

	//abnormalFile.open("abnormalData.txt", std::ofstream::app);
	//otherFile.open("otherData.txt", std::ofstream::app);

	Mat displayImage = imread(name);
	Mat selectedImage(blockSize, blockSize, CV_8UC3, Scalar(0,0,0));

	int histBin[9];

	for(int i = 0; i < 9;i++){
		histBin[i] = 0;
	}

	Mat blockImage;
	string input;

	int i = 0;
	int j = 0;

	displayImage = imread(name);
	//abnormalFile.open("abnormalData.txt", std::ofstream::app);
	//otherFile.open("otherData.txt", std::ofstream::app);
	//Need to surround each block in a color
	//need to iterate through the outsize of block
	for(int r = i; r < i+blockSize; r++){
		for(int c = j; c < j+blockSize; c++){	
			if(r == i || r == i+blockSize-1 || c == j || c == j+blockSize-1){
				displayImage.at<Vec3b>(r,c)[0] = 255;
				displayImage.at<Vec3b>(r,c)[1] = 0;
				displayImage.at<Vec3b>(r,c)[2] = 0;
			}
		}
	}
	imshow("Display Image", displayImage);
	waitKey(1);
	while(run == 1){
		printf("Enter a command (l, r, u, d, s, g, collect, quit)\n");
		cin >> input;
		if(input == "l"){
			printf("How many spaces?\n");
			cin >> input;
			j-=stoi(input);
		}
		else if(input == "r"){
			displayImage = imread(name);
			printf("How many spaces?\n");
			cin >> input;
			j+=stoi(input);
			
		}
		else if(input == "u"){
			printf("How many spaces?\n");
			cin >> input;
			i-=stoi(input);
			
		}
		else if(input == "d"){
			printf("How many spaces?\n");
			cin >> input;
			i+=stoi(input);
		}
		else if(input == "s"){
			if(blockSize > constBlock) blockSize--;
		}
		else if(input == "g"){
			blockSize++;
		}
		else if(input == "collect"){

			printf("Type of image: 1. Abnormal, 2. Normal\n");
			cin >> input;

			int tempBlock;
			if(blockSize == constBlock) tempBlock = 1;
			else tempBlock = blockSize - constBlock;
		
			for(int r = i; r < i+tempBlock; r++){
				for(int c = j; c < j+tempBlock; c++){
					abnormalFile.open("abnormalData.txt", std::ofstream::app);
					otherFile.open("otherData.txt", std::ofstream::app);
					if(r == i || r == i+blockSize-1 || c == j || c == j+blockSize-1){
						displayImage.at<Vec3b>(r,c)[0] = 255;
						displayImage.at<Vec3b>(r,c)[1] = 0;
						displayImage.at<Vec3b>(r,c)[2] = 0;
					}

					for(int x = r; x < r + constBlock; x++){
						for(int z = c; z < c + constBlock; z++){
						
					
					
					//assign each gradient to a bin for use in histogram
					int direction = combinedImage.at<Vec3b>(x,z)[0];
					int intensity = combinedImage.at<Vec3b>(x,z)[1];

					for(int n = 0; n < 9; n++){
						histBin[n] = 0;
					}
					
					if(direction == 0){
						histBin[0] +=intensity;
					}
					if(direction > 0 && direction < 20){
						histBin[0] += intensity / 2;
						histBin[1] += intensity / 2;
					}
					else if(direction == 20){
						histBin[1] += intensity;
					}
					else if(direction > 20 && direction < 40){
						histBin[1] += intensity / 2;
						histBin[2] += intensity / 2;
					}
					else if(direction == 40){
						histBin[2] += intensity;
					}
					else if(direction > 40 && direction < 60){							histBin[2] += intensity / 2;
						histBin[3] += intensity / 2;
					}
					else if(direction == 60){
						histBin[3] += intensity;
					}
					else if(direction > 60 && direction < 80){
						histBin[3] += intensity / 2;
						histBin[4] += intensity / 2;
					}
					else if(direction == 80){
						histBin[4] += intensity;
					}
					else if(direction > 80 && direction < 100){
						histBin[4] += intensity / 2;
						histBin[5] += intensity / 2;
					}
					else if(direction == 100){
						histBin[5] += intensity;
					}
					else if(direction > 100 && direction < 120){
						histBin[5] += intensity / 2;
						histBin[6] += intensity / 2;
					}
					else if(direction == 120){
						histBin[6] += intensity;
					}
					else if(direction > 120 && direction < 140){
						histBin[6] += intensity / 2;
						histBin[7] += intensity / 2;
					}
					else if(direction == 140){
						histBin[7] += intensity;
					}
					else if(direction > 140 && direction < 160){
						histBin[7] += intensity / 2;
						histBin[8] += intensity / 2;
					}
					else if(direction >= 160){
						histBin[8] += intensity;
					}
				}
			}
			if(input == "1"){
				for(int n = 0; n < 9; n++){
					//write data to file...
					abnormalFile << histBin[n] << " ";
				}
				abnormalFile << "\n";
				abnormalFile.close();
				abnormalCount++;
				printf("Should write to abnormalFile... count:%d\n", abnormalCount);
			}
			else if(input == "2"){
				for(int n = 0; n < 9; n++){
					//write data to file...
					otherFile << histBin[n] << " ";
				}
				otherFile << "\n";
				otherFile.close();
				otherCount++;
				printf("Should write to otherFile... count:%d\n", otherCount);
			}
		}}

		}
		else if(input == "quit"){
			run = 0;
			abnormalFile.close();
			otherFile.close();
			exit(0);
		}
		displayImage = imread(name);
		if(j > displayImage.cols-blockSize){
			j = displayImage.cols-blockSize;
		}
		else if(j < 0){
			j = 0;
		}
		if(i > displayImage.rows-blockSize){
			i = displayImage.rows - blockSize;
		}
		else if(i < 0){
			i = 0;
		}
		for(int r = i; r < i+blockSize; r++){
			for(int c = j; c < j+blockSize; c++){	
				if(r == i || r == i+blockSize-1 || c == j || c == j+blockSize-1){
					displayImage.at<Vec3b>(r,c)[0] = 255;
					displayImage.at<Vec3b>(r,c)[1] = 0;
					displayImage.at<Vec3b>(r,c)[2] = 0;
				}
			}
		}
		imshow("Display Image", displayImage);
		waitKey(1);
	}
}
void collectHistogramData2(Mat combinedImage, string name, int constBlock){
	ofstream class1File;
	ofstream class2File;
	ofstream class3File;
	ofstream boundaryFile;
	
	int blockSize = constBlock;

	int class1Count = 0;
	int class2Count = 0;
	int class3Count = 0;
	int boundaryCount = 0;

	int run = 1;

	Mat displayImage = imread(name);
	Mat selectedImage(blockSize, blockSize, CV_8UC3, Scalar(0,0,0));

	int histBin[9];

	for(int i = 0; i < 9;i++){
		histBin[i] = 0;
	}

	Mat blockImage;
	string input;

	int i = 0;
	int j = 0;

	displayImage = imread(name);

	//Need to surround each block in a color
	//need to iterate through the outsize of block
	for(int r = i; r < i+blockSize; r++){
		for(int c = j; c < j+blockSize; c++){	
			if(r == i || r == i+blockSize-1 || c == j || c == j+blockSize-1){
				displayImage.at<Vec3b>(r,c)[0] = 255;
				displayImage.at<Vec3b>(r,c)[1] = 0;
				displayImage.at<Vec3b>(r,c)[2] = 0;
			}
		}
	}
	imshow("Display Image", displayImage);
	waitKey(1);
	while(run == 1){
		printf("Enter a command (l, r, u, d, s, g, collect, quit)\n");
		cin >> input;
		if(input == "l"){
			printf("How many spaces?\n");
			cin >> input;
			j-=stoi(input);
		}
		else if(input == "r"){
			displayImage = imread(name);
			printf("How many spaces?\n");
			cin >> input;
			j+=stoi(input);
			
		}
		else if(input == "u"){
			printf("How many spaces?\n");
			cin >> input;
			i-=stoi(input);
			
		}
		else if(input == "d"){
			printf("How many spaces?\n");
			cin >> input;
			i+=stoi(input);
		}
		else if(input == "s"){
			if(blockSize > constBlock) blockSize--;
		}
		else if(input == "g"){
			blockSize++;
		}
		else if(input == "collect"){

			printf("Type of image: 1. Class One, 2. Class Two, 3. Class Three or boundary\n");
			cin >> input;

			int tempBlock;
			if(blockSize == constBlock) tempBlock = 1;
			else tempBlock = blockSize - constBlock;
		
			for(int r = i; r < i+tempBlock; r++){
				for(int c = j; c < j+tempBlock; c++){
					class1File.open("class1Data.txt", std::ofstream::app);
					class2File.open("class2Data.txt", std::ofstream::app);
					class3File.open("class3Data.txt", std::ofstream::app);
					boundaryFile.open("boundaryData.txt", std::ofstream::app);

					if(r == i || r == i+blockSize-1 || c == j || c == j+blockSize-1){
						displayImage.at<Vec3b>(r,c)[0] = 255;
						displayImage.at<Vec3b>(r,c)[1] = 0;
						displayImage.at<Vec3b>(r,c)[2] = 0;
					}

					for(int x = r; x < r + constBlock; x++){
						for(int z = c; z < c + constBlock; z++){
						
					
					
					//assign each gradient to a bin for use in histogram
					int direction = combinedImage.at<Vec3b>(x,z)[0];
					int intensity = combinedImage.at<Vec3b>(x,z)[1];

					for(int n = 0; n < 9; n++){
						histBin[n] = 0;
					}
					
					if(direction == 0){
						histBin[0] +=intensity;
					}
					if(direction > 0 && direction < 20){
						histBin[0] += intensity / 2;
						histBin[1] += intensity / 2;
					}
					else if(direction == 20){
						histBin[1] += intensity;
					}
					else if(direction > 20 && direction < 40){
						histBin[1] += intensity / 2;
						histBin[2] += intensity / 2;
					}
					else if(direction == 40){
						histBin[2] += intensity;
					}
					else if(direction > 40 && direction < 60){							histBin[2] += intensity / 2;
						histBin[3] += intensity / 2;
					}
					else if(direction == 60){
						histBin[3] += intensity;
					}
					else if(direction > 60 && direction < 80){
						histBin[3] += intensity / 2;
						histBin[4] += intensity / 2;
					}
					else if(direction == 80){
						histBin[4] += intensity;
					}
					else if(direction > 80 && direction < 100){
						histBin[4] += intensity / 2;
						histBin[5] += intensity / 2;
					}
					else if(direction == 100){
						histBin[5] += intensity;
					}
					else if(direction > 100 && direction < 120){
						histBin[5] += intensity / 2;
						histBin[6] += intensity / 2;
					}
					else if(direction == 120){
						histBin[6] += intensity;
					}
					else if(direction > 120 && direction < 140){
						histBin[6] += intensity / 2;
						histBin[7] += intensity / 2;
					}
					else if(direction == 140){
						histBin[7] += intensity;
					}
					else if(direction > 140 && direction < 160){
						histBin[7] += intensity / 2;
						histBin[8] += intensity / 2;
					}
					else if(direction >= 160){
						histBin[8] += intensity;
					}
				}
			}
			if(input == "1"){
				for(int n = 0; n < 9; n++){
					//write data to file...
					class1File << histBin[n] << " ";
				}
				class1File << "\n";
				class1File.close();
				class1Count++;
				printf("Should write to class1File... count:%d\n", class1Count);
			}
			else if(input == "2"){
				for(int n = 0; n < 9; n++){
					//write data to file...
					class2File << histBin[n] << " ";
				}
				class2File << "\n";
				class2File.close();
				class2Count++;
				printf("Should write to class2File... count:%d\n", class2Count);
			}
			else if(input == "3"){
				for(int n = 0; n < 9; n++){
					//write data to file...
					class3File << histBin[n] << " ";
				}
				class3File << "\n";
				class3File.close();
				class3Count++;
				printf("Should write to class3File... count:%d\n", class3Count);
			}
			else if(input == "boundary"){
				for(int n = 0; n < 9; n++){
					//write data to file...
					boundaryFile << histBin[n] << " ";
				}
				boundaryFile << "\n";
				boundaryFile.close();
				boundaryCount++;
				printf("Should write to boundaryFile... count:%d\n", boundaryCount);

			}
		}}

		}
		else if(input == "quit"){
			run = 0;
			class1File.close();
			class2File.close();
			class3File.close();
			boundaryFile.close();
			exit(0);
		}
		displayImage = imread(name);
		if(j > displayImage.cols-blockSize){
			j = displayImage.cols-blockSize;
		}
		else if(j < 0){
			j = 0;
		}
		if(i > displayImage.rows-blockSize){
			i = displayImage.rows - blockSize;
		}
		else if(i < 0){
			i = 0;
		}
		for(int r = i; r < i+blockSize; r++){
			for(int c = j; c < j+blockSize; c++){	
				if(r == i || r == i+blockSize-1 || c == j || c == j+blockSize-1){
					displayImage.at<Vec3b>(r,c)[0] = 255;
					displayImage.at<Vec3b>(r,c)[1] = 0;
					displayImage.at<Vec3b>(r,c)[2] = 0;
				}
			}
		}
		imshow("Display Image", displayImage);
		waitKey(1);
	}
}

Mat runSVM1(Mat combinedImage, int blockSize, string name){
	
	Mat image = imread(name);

	int abnormalLines = 0;
	int otherLines = 0;
	string line;
	int abnormalCount = 0;
	
	int histBin[9];

	for(int i = 0; i < 9;i++){
		histBin[i] = 0;
	}

	int p1, p2, p3, p4, p5, p6, p7, p8, p9 = 0;	

	ifstream abnormalFile("abnormalData.txt");
	ifstream otherFile("otherData.txt");
	while(getline(abnormalFile, line)){
		abnormalLines++;
	}
	while(getline(otherFile, line)){
		otherLines++;
	}

	int totalLines = abnormalLines + otherLines;
	int dataLabels[abnormalLines + otherLines];

	for(int i = 0; i < totalLines; i++){
		if( i < abnormalLines) dataLabels[i] = 1;
		else dataLabels[i] = -1;
	}


	Mat labelsMat(totalLines, 1, CV_32SC1, dataLabels);

	int trainData[totalLines][9];

	//initialize values within data array
	for(int i = 0; i < totalLines; i++){
		for(int j = 0; j < 9; j++){
			trainData[i][j] = 0;
		}
	}

	//reset files to beginning to start reading lines
	abnormalFile.clear();
	abnormalFile.seekg(0, ios::beg);
	otherFile.clear();
	otherFile.seekg(0, ios::beg);

	int count = 0;
	//parse the file data into the data arrays...
	while(getline(abnormalFile, line)){
		std::istringstream iss(line);
		iss >> p1 >> p2 >> p3 >> p4 >> p5 >> p6 >> p7 >> p8 >> p9;
		
		trainData[count][0] = p1;
		trainData[count][1] = p2;
		trainData[count][2] = p3;
		trainData[count][3] = p4;
		trainData[count][4] = p5;
		trainData[count][5] = p6;
		trainData[count][6] = p7;
		trainData[count][7] = p8;
		trainData[count][8] = p9;

		count++;	
		//printf("abnormalFile...\n");	
	}
	while(getline(otherFile, line)){
		std::istringstream iss(line);
		iss >> p1 >> p2 >> p3 >> p4 >> p5 >> p6 >> p7 >> p8 >> p9;
		
		trainData[count][0] = p1;
		trainData[count][1] = p2;
		trainData[count][2] = p3;
		trainData[count][3] = p4;
		trainData[count][4] = p5;
		trainData[count][5] = p6;
		trainData[count][6] = p7;
		trainData[count][7] = p8;
		trainData[count][8] = p9;

		count++;
		//printf("otherFile...\n");		
	}

	/*for(int i = 0; i < count; i++){
		string tempString;
		if(dataLabels[i] == 1) tempString = "Abnormal: ";
		else tempString = "Normal: ";
	
		printf("%s", tempString.c_str());
		for(int j = 0; j < 9; j++){
			printf("%d ", trainData[i][j]);
		}
		printf("\n");
	}*/

    	Mat dataMat(totalLines, 9, CV_32FC1, trainData);

	Ptr<ml::SVM> svm = ml::SVM::create();

	svm->setType(ml::SVM::C_SVC);
	svm->setKernel(ml::SVM::LINEAR);
	
	printf("Training SVM...\n");
	svm->train(dataMat, ml::ROW_SAMPLE, labelsMat);
	printf("Done Training SVM...\n");
	
	//now need to iterate through the image, create sample mats, and run predictions
	printf("Running Predicions...\n");
	for(int i = 0; i < combinedImage.rows-blockSize; i+=blockSize){
		for(int j = 0; j < combinedImage.cols-blockSize; j+=blockSize){
			for(int r = i; r < i+blockSize; r++){
				for(int c = j; c < j+blockSize; c++){
					int direction = combinedImage.at<Vec3b>(r,c)[0];
					int intensity = combinedImage.at<Vec3b>(r,c)[1];
					
					for(int n = 0; n < 9; n++){
						histBin[n] = 0;
					}
					
					if(direction == 0){
						histBin[0] +=intensity;
					}
					if(direction > 0 && direction < 20){
						histBin[0] += intensity / 2;
						histBin[1] += intensity / 2;
					}
					else if(direction == 20){
						histBin[1] += intensity;
					}
					else if(direction > 20 && direction < 40){
						histBin[1] += intensity / 2;
						histBin[2] += intensity / 2;
					}
					else if(direction == 40){
						histBin[2] += intensity;
					}
					else if(direction > 40 && direction < 60){							histBin[2] += intensity / 2;
						histBin[3] += intensity / 2;
					}
					else if(direction == 60){
						histBin[3] += intensity;
					}
					else if(direction > 60 && direction < 80){
						histBin[3] += intensity / 2;
						histBin[4] += intensity / 2;
					}
					else if(direction == 80){
						histBin[4] += intensity;
					}
					else if(direction > 80 && direction < 100){
						histBin[4] += intensity / 2;
						histBin[5] += intensity / 2;
					}
					else if(direction == 100){
						histBin[5] += intensity;
					}
					else if(direction > 100 && direction < 120){
						histBin[5] += intensity / 2;
						histBin[6] += intensity / 2;
					}
					else if(direction == 120){
						histBin[6] += intensity;
					}
					else if(direction > 120 && direction < 140){
						histBin[6] += intensity / 2;
						histBin[7] += intensity / 2;
					}
					else if(direction == 140){
						histBin[7] += intensity;
					}
					else if(direction > 140 && direction < 160){
						histBin[7] += intensity / 2;
						histBin[8] += intensity / 2;
					}
					else if(direction >= 160){
						histBin[8] += intensity;
					}
				}
			}
			//now we have the hist bin..

			Mat predictMat(1, 9, CV_32FC1, histBin);
			
			float response = svm->predict(predictMat);
			printf("Response: %f\n", response);
			//now highlight the block if it is predicted as an abnormal cell..
			if(response == 1){
				count++;
				for(int r = i; r < i+blockSize; r++){
					for(int c = j; c < j+blockSize; c++){
						if(r == i || r == i+blockSize-1 || c == j || c == j+blockSize-1){
							image.at<Vec3b>(r,c)[2] = 255;
							image.at<Vec3b>(r,c)[1] = 0;
							image.at<Vec3b>(r,c)[0] = 0;
						}
						
					}
				}
			}
			else if(response == -1){
				for(int r = i; r < i+blockSize; r++){
					for(int c = j; c < j+blockSize; c++){
						if(r == i || r == i+blockSize-1 || c == j || c == j+blockSize-1){
							image.at<Vec3b>(r,c)[2] = 0;
							image.at<Vec3b>(r,c)[1] = 0;
							image.at<Vec3b>(r,c)[0] = 255;
						}
						
					}
				}
			}		
		}
	}
	printf("Count of Blocks with Abnormal Cells: %d\n", abnormalCount);
	return image;
}

Mat runSVM2(Mat combinedImage, int blockSize, string name){
	
	Mat image = imread(name);

	int class1Lines = 0;
	int class2Lines = 0;
	int class3Lines = 0;
	int boundaryLines = 0;

	string line;
	
	int histBin[9];

	for(int i = 0; i < 9;i++){
		histBin[i] = 0;
	}

	int p1, p2, p3, p4, p5, p6, p7, p8, p9 = 0;	

	ifstream class1File("class1Data.txt");
	ifstream class2File("class2Data.txt");
	ifstream class3File("class3Data.txt");
	ifstream boundaryFile("boundaryData.txt");

	while(getline(class1File, line)){
		class1Lines++;
	}
	while(getline(class2File, line)){
		class2Lines++;
	}
	while(getline(class3File, line)){
		class3Lines++;
	}
	while(getline(boundaryFile, line)){
		boundaryLines++;
	}

	int totalLines = class1Lines + class2Lines + class3Lines + boundaryLines;
	int data1Labels[totalLines];
	int data2Labels[totalLines];
	int data3Labels[totalLines];
	int boundaryLabels[totalLines];

	for(int i = 0; i < totalLines; i++){
		if( i < class1Lines) data1Labels[i] = 1;
		else data1Labels[i] = -1;

		if( i < class2Lines) data2Labels[i] = 1;
		else data2Labels[i] = -1;

		if( i < class3Lines) data3Labels[i] = 1;
		else data3Labels[i] = -1;
		
		if(i < boundaryLines) boundaryLabels[i] = 1;
		else boundaryLabels[i] = -1;
	}


	Mat class1LabelsMat(totalLines, 1, CV_32SC1, data1Labels);
	Mat class2LabelsMat(totalLines, 1, CV_32SC1, data2Labels);
	Mat class3LabelsMat(totalLines, 1, CV_32SC1, data3Labels);
	Mat boundaryLabelsMat(totalLines, 1, CV_32SC1, boundaryLabels);

	int train1Data[totalLines][9];
	int train2Data[totalLines][9];
	int train3Data[totalLines][9];
	int boundaryData[totalLines][9];

	

	//initialize values within data array
	for(int i = 0; i < totalLines; i++){
		for(int j = 0; j < 9; j++){
			train1Data[i][j] = 0;
			train2Data[i][j] = 0;
			train3Data[i][j] = 0;
			boundaryData[i][j] = 0;
		}
	}

	//reset files to beginning to start reading lines
	class1File.clear();
	class1File.seekg(0, ios::beg);

	class2File.clear();
	class2File.seekg(0, ios::beg);

	class3File.clear();
	class3File.seekg(0, ios::beg);

	boundaryFile.clear();
	boundaryFile.seekg(0, ios::beg);

	int count = 0;
	//parse the file data into the data arrays...
	//train1Data
	while(getline(class1File, line)){
		std::istringstream iss(line);
		iss >> p1 >> p2 >> p3 >> p4 >> p5 >> p6 >> p7 >> p8 >> p9;
		
		train1Data[count][0] = p1;
		train1Data[count][1] = p2;
		train1Data[count][2] = p3;
		train1Data[count][3] = p4;
		train1Data[count][4] = p5;
		train1Data[count][5] = p6;
		train1Data[count][6] = p7;
		train1Data[count][7] = p8;
		train1Data[count][8] = p9;

		count++;	
	}
	while(getline(class2File, line)){
		std::istringstream iss(line);
		iss >> p1 >> p2 >> p3 >> p4 >> p5 >> p6 >> p7 >> p8 >> p9;
		
		train1Data[count][0] = p1;
		train1Data[count][1] = p2;
		train1Data[count][2] = p3;
		train1Data[count][3] = p4;
		train1Data[count][4] = p5;
		train1Data[count][5] = p6;
		train1Data[count][6] = p7;
		train1Data[count][7] = p8;
		train1Data[count][8] = p9;

		count++;
	}
	while(getline(class3File, line)){
		std::istringstream iss(line);
		iss >> p1 >> p2 >> p3 >> p4 >> p5 >> p6 >> p7 >> p8 >> p9;
		
		train1Data[count][0] = p1;
		train1Data[count][1] = p2;
		train1Data[count][2] = p3;
		train1Data[count][3] = p4;
		train1Data[count][4] = p5;
		train1Data[count][5] = p6;
		train1Data[count][6] = p7;
		train1Data[count][7] = p8;
		train1Data[count][8] = p9;

		count++;
	}
	while(getline(boundaryFile, line)){
		std::istringstream iss(line);
		iss >> p1 >> p2 >> p3 >> p4 >> p5 >> p6 >> p7 >> p8 >> p9;
		
		train1Data[count][0] = p1;
		train1Data[count][1] = p2;
		train1Data[count][2] = p3;
		train1Data[count][3] = p4;
		train1Data[count][4] = p5;
		train1Data[count][5] = p6;
		train1Data[count][6] = p7;
		train1Data[count][7] = p8;
		train1Data[count][8] = p9;

		count++;
	}

	//reset files to beginning to start reading lines
	class1File.clear();
	class1File.seekg(0, ios::beg);

	class2File.clear();
	class2File.seekg(0, ios::beg);

	class3File.clear();
	class3File.seekg(0, ios::beg);

	boundaryFile.clear();
	boundaryFile.seekg(0, ios::beg);

	count = 0;
	//parse the file data into the data arrays...
	//train1Data
	while(getline(class2File, line)){
		std::istringstream iss(line);
		iss >> p1 >> p2 >> p3 >> p4 >> p5 >> p6 >> p7 >> p8 >> p9;
		
		train2Data[count][0] = p1;
		train2Data[count][1] = p2;
		train2Data[count][2] = p3;
		train2Data[count][3] = p4;
		train2Data[count][4] = p5;
		train2Data[count][5] = p6;
		train2Data[count][6] = p7;
		train2Data[count][7] = p8;
		train2Data[count][8] = p9;

		count++;	
	}
	while(getline(class1File, line)){
		std::istringstream iss(line);
		iss >> p1 >> p2 >> p3 >> p4 >> p5 >> p6 >> p7 >> p8 >> p9;
		
		train2Data[count][0] = p1;
		train2Data[count][1] = p2;
		train2Data[count][2] = p3;
		train2Data[count][3] = p4;
		train2Data[count][4] = p5;
		train2Data[count][5] = p6;
		train2Data[count][6] = p7;
		train2Data[count][7] = p8;
		train2Data[count][8] = p9;

		count++;
	}
	while(getline(class3File, line)){
		std::istringstream iss(line);
		iss >> p1 >> p2 >> p3 >> p4 >> p5 >> p6 >> p7 >> p8 >> p9;
		
		train2Data[count][0] = p1;
		train2Data[count][1] = p2;
		train2Data[count][2] = p3;
		train2Data[count][3] = p4;
		train2Data[count][4] = p5;
		train2Data[count][5] = p6;
		train2Data[count][6] = p7;
		train2Data[count][7] = p8;
		train2Data[count][8] = p9;

		count++;
	}
	while(getline(boundaryFile, line)){
		std::istringstream iss(line);
		iss >> p1 >> p2 >> p3 >> p4 >> p5 >> p6 >> p7 >> p8 >> p9;
		
		train2Data[count][0] = p1;
		train2Data[count][1] = p2;
		train2Data[count][2] = p3;
		train2Data[count][3] = p4;
		train2Data[count][4] = p5;
		train2Data[count][5] = p6;
		train2Data[count][6] = p7;
		train2Data[count][7] = p8;
		train2Data[count][8] = p9;

		count++;
	}
	//reset files to beginning to start reading lines
	class1File.clear();
	class1File.seekg(0, ios::beg);

	class2File.clear();
	class2File.seekg(0, ios::beg);

	class3File.clear();
	class3File.seekg(0, ios::beg);

	boundaryFile.clear();
	boundaryFile.seekg(0, ios::beg);

	count = 0;
	//parse the file data into the data arrays...
	//train1Data
	while(getline(class3File, line)){
		std::istringstream iss(line);
		iss >> p1 >> p2 >> p3 >> p4 >> p5 >> p6 >> p7 >> p8 >> p9;
		
		train3Data[count][0] = p1;
		train3Data[count][1] = p2;
		train3Data[count][2] = p3;
		train3Data[count][3] = p4;
		train3Data[count][4] = p5;
		train3Data[count][5] = p6;
		train3Data[count][6] = p7;
		train3Data[count][7] = p8;
		train3Data[count][8] = p9;

		count++;	
	}
	while(getline(class2File, line)){
		std::istringstream iss(line);
		iss >> p1 >> p2 >> p3 >> p4 >> p5 >> p6 >> p7 >> p8 >> p9;
		
		train3Data[count][0] = p1;
		train3Data[count][1] = p2;
		train3Data[count][2] = p3;
		train3Data[count][3] = p4;
		train3Data[count][4] = p5;
		train3Data[count][5] = p6;
		train3Data[count][6] = p7;
		train3Data[count][7] = p8;
		train3Data[count][8] = p9;

		count++;
	}
	while(getline(class1File, line)){
		std::istringstream iss(line);
		iss >> p1 >> p2 >> p3 >> p4 >> p5 >> p6 >> p7 >> p8 >> p9;
		
		train3Data[count][0] = p1;
		train3Data[count][1] = p2;
		train3Data[count][2] = p3;
		train3Data[count][3] = p4;
		train3Data[count][4] = p5;
		train3Data[count][5] = p6;
		train3Data[count][6] = p7;
		train3Data[count][7] = p8;
		train3Data[count][8] = p9;

		count++;
	}
	while(getline(boundaryFile, line)){
		std::istringstream iss(line);
		iss >> p1 >> p2 >> p3 >> p4 >> p5 >> p6 >> p7 >> p8 >> p9;
		
		train3Data[count][0] = p1;
		train3Data[count][1] = p2;
		train3Data[count][2] = p3;
		train3Data[count][3] = p4;
		train3Data[count][4] = p5;
		train3Data[count][5] = p6;
		train3Data[count][6] = p7;
		train3Data[count][7] = p8;
		train3Data[count][8] = p9;

		count++;
	}

		//reset files to beginning to start reading lines
	class1File.clear();
	class1File.seekg(0, ios::beg);

	class2File.clear();
	class2File.seekg(0, ios::beg);

	class3File.clear();
	class3File.seekg(0, ios::beg);

	boundaryFile.clear();
	boundaryFile.seekg(0, ios::beg);

	count = 0;
	//parse the file data into the data arrays...
	//train1Data
	while(getline(boundaryFile, line)){
		std::istringstream iss(line);
		iss >> p1 >> p2 >> p3 >> p4 >> p5 >> p6 >> p7 >> p8 >> p9;
		
		boundaryData[count][0] = p1;
		boundaryData[count][1] = p2;
		boundaryData[count][2] = p3;
		boundaryData[count][3] = p4;
		boundaryData[count][4] = p5;
		boundaryData[count][5] = p6;
		boundaryData[count][6] = p7;
		boundaryData[count][7] = p8;
		boundaryData[count][8] = p9;

		count++;	
	}
	while(getline(class2File, line)){
		std::istringstream iss(line);
		iss >> p1 >> p2 >> p3 >> p4 >> p5 >> p6 >> p7 >> p8 >> p9;
		
		boundaryData[count][0] = p1;
		boundaryData[count][1] = p2;
		boundaryData[count][2] = p3;
		boundaryData[count][3] = p4;
		boundaryData[count][4] = p5;
		boundaryData[count][5] = p6;
		boundaryData[count][6] = p7;
		boundaryData[count][7] = p8;
		boundaryData[count][8] = p9;

		count++;
	}
	while(getline(class1File, line)){
		std::istringstream iss(line);
		iss >> p1 >> p2 >> p3 >> p4 >> p5 >> p6 >> p7 >> p8 >> p9;
		
		boundaryData[count][0] = p1;
		boundaryData[count][1] = p2;
		boundaryData[count][2] = p3;
		boundaryData[count][3] = p4;
		boundaryData[count][4] = p5;
		boundaryData[count][5] = p6;
		boundaryData[count][6] = p7;
		boundaryData[count][7] = p8;
		boundaryData[count][8] = p9;

		count++;
	}
	while(getline(boundaryFile, line)){
		std::istringstream iss(line);
		iss >> p1 >> p2 >> p3 >> p4 >> p5 >> p6 >> p7 >> p8 >> p9;
		
		boundaryData[count][0] = p1;
		boundaryData[count][1] = p2;
		boundaryData[count][2] = p3;
		boundaryData[count][3] = p4;
		boundaryData[count][4] = p5;
		boundaryData[count][5] = p6;
		boundaryData[count][6] = p7;
		boundaryData[count][7] = p8;
		boundaryData[count][8] = p9;

		count++;
	}

    	Mat class1DataMat(totalLines, 9, CV_32FC1, train1Data);
    	Mat class2DataMat(totalLines, 9, CV_32FC1, train2Data);
    	Mat class3DataMat(totalLines, 9, CV_32FC1, train3Data);
	Mat boundaryDataMat(totalLines, 9, CV_32FC1, boundaryData);


	Ptr<ml::SVM> class1svm = ml::SVM::create();
	Ptr<ml::SVM> class2svm = ml::SVM::create();
	Ptr<ml::SVM> class3svm = ml::SVM::create();
	Ptr<ml::SVM> boundarysvm = ml::SVM::create();



	class1svm->setType(ml::SVM::C_SVC);
	class1svm->setKernel(ml::SVM::LINEAR);
	
	printf("Training SVM 1...\n");
	class1svm->train(class1DataMat, ml::ROW_SAMPLE, class1LabelsMat);
	printf("Done Training SVM 1...\n");

	class2svm->setType(ml::SVM::C_SVC);
	class2svm->setKernel(ml::SVM::LINEAR);
	
	printf("Training SVM 2...\n");
	class2svm->train(class2DataMat, ml::ROW_SAMPLE, class2LabelsMat);
	printf("Done Training SVM 2...\n");

	class3svm->setType(ml::SVM::C_SVC);
	class3svm->setKernel(ml::SVM::LINEAR);
	
	printf("Training SVM 3...\n");
	class3svm->train(class3DataMat, ml::ROW_SAMPLE, class3LabelsMat);
	printf("Done Training SVM 3...\n");

	printf("Training Boundary SVM...\n");
	boundarysvm->train(boundaryDataMat, ml::ROW_SAMPLE, boundaryLabelsMat);
	printf("Done Training Boundary SVM...\n");	
	
	//now need to iterate through the image, create sample mats, and run predictions
	printf("Running Predicions...\n");
	for(int i = 0; i < combinedImage.rows-blockSize; i+=blockSize){
		for(int j = 0; j < combinedImage.cols-blockSize; j+=blockSize){
			for(int r = i; r < i+blockSize; r++){
				for(int c = j; c < j+blockSize; c++){
					int direction = combinedImage.at<Vec3b>(r,c)[0];
					int intensity = combinedImage.at<Vec3b>(r,c)[1];
					
					for(int n = 0; n < 9; n++){
						histBin[n] = 0;
					}
					
					if(direction == 0){
						histBin[0] +=intensity;
					}
					if(direction > 0 && direction < 20){
						histBin[0] += intensity / 2;
						histBin[1] += intensity / 2;
					}
					else if(direction == 20){
						histBin[1] += intensity;
					}
					else if(direction > 20 && direction < 40){
						histBin[1] += intensity / 2;
						histBin[2] += intensity / 2;
					}
					else if(direction == 40){
						histBin[2] += intensity;
					}
					else if(direction > 40 && direction < 60){							histBin[2] += intensity / 2;
						histBin[3] += intensity / 2;
					}
					else if(direction == 60){
						histBin[3] += intensity;
					}
					else if(direction > 60 && direction < 80){
						histBin[3] += intensity / 2;
						histBin[4] += intensity / 2;
					}
					else if(direction == 80){
						histBin[4] += intensity;
					}
					else if(direction > 80 && direction < 100){
						histBin[4] += intensity / 2;
						histBin[5] += intensity / 2;
					}
					else if(direction == 100){
						histBin[5] += intensity;
					}
					else if(direction > 100 && direction < 120){
						histBin[5] += intensity / 2;
						histBin[6] += intensity / 2;
					}
					else if(direction == 120){
						histBin[6] += intensity;
					}
					else if(direction > 120 && direction < 140){
						histBin[6] += intensity / 2;
						histBin[7] += intensity / 2;
					}
					else if(direction == 140){
						histBin[7] += intensity;
					}
					else if(direction > 140 && direction < 160){
						histBin[7] += intensity / 2;
						histBin[8] += intensity / 2;
					}
					else if(direction >= 160){
						histBin[8] += intensity;
					}
				}
			}
			//now we have the hist bin..

			Mat predictMat(1, 9, CV_32FC1, histBin);
			int response1 = class1svm->predict(predictMat);
			int response2 = class2svm->predict(predictMat);
			int response3 = class3svm->predict(predictMat);
			int boundaryResponse = boundarysvm -> predict(predictMat);
			printf("Response 1: %d, Response 2: %d, Response 3: %d, Boundary Response: %d\n", response1, response2, response3, boundaryResponse);

			//now highlight the block if it is predicted as an abnormal cell..
			if(response1 == 1){
				for(int r = i; r < i+blockSize; r++){
					for(int c = j; c < j+blockSize; c++){
						if(r == i || r == i+blockSize-1 || c == j || c == j+blockSize-1){
							image.at<Vec3b>(r,c)[2] = 255;
							image.at<Vec3b>(r,c)[1] = 0;
							image.at<Vec3b>(r,c)[0] = 0;
						}
						
					}
				}
			}
			if(response2 == 2){
				for(int r = i; r < i+blockSize; r++){
					for(int c = j; c < j+blockSize; c++){
						if(r == i || r == i+blockSize-1 || c == j || c == j+blockSize-1){
							image.at<Vec3b>(r,c)[2] = 0;
							image.at<Vec3b>(r,c)[1] = 255;
							image.at<Vec3b>(r,c)[0] = 0;
						}
						
					}
				}
			}
			if(response3 == 1){
				for(int r = i; r < i+blockSize; r++){
					for(int c = j; c < j+blockSize; c++){
						if(r == i || r == i+blockSize-1 || c == j || c == j+blockSize-1){
							image.at<Vec3b>(r,c)[2] = 0;
							image.at<Vec3b>(r,c)[1] = 0;
							image.at<Vec3b>(r,c)[0] = 255;
						}
						
					}
				}
			}
			if(boundaryResponse == 1){
				for(int r = i; r < i+blockSize; r++){
					for(int c = j; c < j+blockSize; c++){
						if(r == i || r == i+blockSize-1 || c == j || c == j+blockSize-1){
							image.at<Vec3b>(r,c)[2] = 255;
							image.at<Vec3b>(r,c)[1] = 0;
							image.at<Vec3b>(r,c)[0] = 255;
						}
						
					}
				}
			}	
		}
	}	
	return image;
}


int main(int argc, char** argv){

	string name1 = "sample1.bmp";
	string name2 = "sample2.bmp";

	string input;
	string mode;

	printf("Part 1 or Part 2? (Enter 1 or 2)\n");
	cin >> input;
	printf("Collect Data or Run SVM? (Enter collect or svm)\n");
	cin >> mode;

	int blockSize1 = 8;
	int blockSize2 = 20;



	Mat orig1 = imread(name1);
	Mat grey1 = makeGrey(orig1, name1);
	Mat image1 = smoothImage(grey1, name1);
	Mat verticalSobel1 =  getVerticalSobel(image1, name1);
	Mat horizontalSobel1 = getHorizontalSobel(image1, name1);
	Mat sobel1 = addSobel(verticalSobel1, horizontalSobel1, name1);
	Mat gradient1 = getGradient(verticalSobel1, horizontalSobel1, name1);
	Mat combinedImage1 = combineForHistogram(sobel1, gradient1, name1);


	Mat orig2 = imread(name2);
	Mat grey2 = makeGrey(orig2, name2);
	Mat image2 = smoothImage(grey2, name2);
	Mat verticalSobel2 = getVerticalSobel(image2, name2);
	Mat horizontalSobel2 = getHorizontalSobel(image2, name2);
	Mat sobel2 = addSobel(verticalSobel2, horizontalSobel2, name2);
	Mat gradient2 = getGradient(verticalSobel2, horizontalSobel2, name2);
	Mat combinedImage2 = combineForHistogram(sobel2, gradient2, name2);
	
	if(input == "1"){

		
		//imshow("Original Image 1", orig1);
		//imshow("Grey Image 1", image1);
		//imshow("Vertical Sobel 1", verticalSobel1);
		//imshow("Horizontal Sobel 1", horizontalSobel1);
		//imshow("Sobel 1", sobel1);
		//imshow("Gradient Image 1", gradient1);
		//imshow("Combined Sobel / Gradient 1", combinedImage1);
		
		if(mode == "collect"){
			imshow("Sobel 1", sobel1);
			collectHistogramData1(combinedImage1, name1, blockSize1);
		}
		else if(mode == "svm"){
			Mat responseImage1 = runSVM1(combinedImage1, blockSize1, name1);
			imshow("Response Image 1", responseImage1);
		}
	}
	else if(input == "2"){
	
		//imshow("Original Image 2", orig2);
		//imshow("Grey Image 2", image2);
		//imshow("Vertical Sobel 2", verticalSobel2);
		//imshow("Horizontal Sobel 2", horizontalSobel2);
		//imshow("Sobel 2", sobel2);
		//imshow("Gradient Image 2", gradient2);
		//imshow("Combined Sobel / Gradient 2", combinedImage2);
		
		if(mode == "collect"){
			imshow("Sobel 2", sobel2);
			collectHistogramData2(combinedImage2, name2, blockSize2);
		}
		else if(mode == "svm"){
			Mat responseImage2 = runSVM2(combinedImage1, blockSize2, name2);
			imshow("Response Image 2", responseImage2);
		}
	}
	


	waitKey();
	return 0;
}