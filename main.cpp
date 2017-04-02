#include <opencv2/opencv.hpp>
#include <iostream>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>

#include <time.h>
#include <cmath>

#include <stdio.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>

//for CUDA GpuMat
#include <opencv2/core/cuda.hpp>

//for CUDA corner detection
#include <opencv2/cudaimgproc.hpp>


#include "PracticalSocket.h"
#include <cstdlib>
#include "config.h"

using namespace std;
using namespace cv;

const int HORIZONTAL_BORDER_CROP = 80; // In pixels. Crops the border to reduce the black borders from stabilisation being too noticeable.


//Graph Data
struct GraphData{
    double dx, dy, da, dx_corrected, dy_corrected, da_corrected;
    GraphData(double d2, double d3, double d4, double d5, double d6, double d7) {
        dx = d2;
        dy = d3;
        da = d4;
        dx_corrected = d5;
        dy_corrected = d6;
        da_corrected = d7;
    }
    GraphData(){}
};

//END Graph Data

struct TransformParam
{
    TransformParam() {}
    TransformParam(double _dx, double _dy, double _da) {
        dx = _dx;
        dy = _dy;
        da = _da;
    }
    double dx;
    double dy;
    double da; // angle
};

struct Trajectory
{
    Trajectory() {}
    Trajectory(double _x, double _y, double _a) {
        x = _x;
        y = _y;
        a = _a;
    }
	// "+"
	friend Trajectory operator+(const Trajectory &c1,const Trajectory  &c2){
		return Trajectory(c1.x+c2.x,c1.y+c2.y,c1.a+c2.a);
	}
	//"-"
	friend Trajectory operator-(const Trajectory &c1,const Trajectory  &c2){
		return Trajectory(c1.x-c2.x,c1.y-c2.y,c1.a-c2.a);
	}
	//"*"
	friend Trajectory operator*(const Trajectory &c1,const Trajectory  &c2){
		return Trajectory(c1.x*c2.x,c1.y*c2.y,c1.a*c2.a);
	}
	//"/"
	friend Trajectory operator/(const Trajectory &c1,const Trajectory  &c2){
		return Trajectory(c1.x/c2.x,c1.y/c2.y,c1.a/c2.a);
	}
	//"="
	Trajectory operator =(const Trajectory &rx){
		x = rx.x;
		y = rx.y;
		a = rx.a;
		return Trajectory(x,y,a);
	}
    double x;
    double y;
    double a; // angle
};
//
int main(int argc, char **argv)
{
	//FPS declarations
    int64 now, then;
    double elapsed_seconds;
    double tickspersecond = cvGetTickFrequency()*1.0e6;
	string tempFPS, tempPAN;

	//Graph data socket
	cout << "Waiting for the client to connect -> " << endl;
	int server_fd, new_socket;
	struct sockaddr_in address;
	int opt = 1;
	int addrlen = sizeof(address);

	server_fd = socket(AF_INET, SOCK_STREAM, 0);
	setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt));
	address.sin_family = AF_INET;
	address.sin_addr.s_addr = INADDR_ANY;
	address.sin_port = htons(1234);

	bind(server_fd, (struct sockaddr *)&address, sizeof(address));
	listen(server_fd, 3);
	new_socket = accept(server_fd, (struct sockaddr *)&address, (socklen_t*)&addrlen);
	cout <<"Connection from client accepted ->" << endl;
	//end Graph data socket

	// Graph parameters
	double graphDx, graphDy, graphDa, newGraphDx, newGraphDy, newGraphDa;
	// end Graph parameters

    // Panning parameter : Max tracked points per frame
    int panningParameter;
    // end Panning parameter

	// For further analysis
	ofstream out_transform("prev_to_cur_transformation.txt");
	ofstream out_trajectory("trajectory.txt");
	ofstream out_smoothed_trajectory("smoothed_trajectory.txt");
	ofstream out_new_transform("new_prev_to_cur_transformation.txt");

	string srcIP ;
	map<int, string> testVideos;
	testVideos[1] = "/home/srp3003/Desktop/Samples/s1.avi";
	testVideos[2] = "/home/srp3003/Desktop/Samples/s2.avi";
	testVideos[3] = "/home/srp3003/Desktop/Samples/s3.avi";
	testVideos[4] = "/home/srp3003/Desktop/Samples/s4.avi";
	testVideos[5] = "/home/srp3003/Desktop/Samples/s5.avi";
	testVideos[6] = "/home/srp3003/Desktop/Samples/s6.avi";
	testVideos[7] = "/home/srp3003/Desktop/Samples/s7.avi";

	string testVideo;
	//GUI goes here

	VideoCapture cap;
	int option;
	cout << "1. Close objects" << endl;
	cout << "2. Drone footage" << endl;
	cout << "3. Bike Footage" << endl;
	cout << "4. Bike footage with close objects and rotation" << endl;
	cout << "5. Footage with varied focus" << endl;
	cout << "6. Hackathon footage : Jitter" << endl;
	cout << "7. Hackathon footage : Rotation7" << endl;
	cout << "Enter video # (0 for IP cam) : " ;
	cin >> option;
	if(option == 10)
		cap.open(0);
	else if (option == 0) {
		cout << "Enter the IP of the camera : ";
		cin >> srcIP;
		string srcURL = "http://" + srcIP + ":8080/videofeed?dummy=param.mpjg";
		cap.open(srcURL);
	}
	else
		cap.open(testVideos[option]);

//	//test with webcam
//	VideoCapture cap(0);

	// Setting the panning parameter
	if(option == 0)
		panningParameter = 10;
	else if(option == 1)
		panningParameter = 10;
	else if(option == 2)
		panningParameter = 10;
	else if(option == 3)
		panningParameter = 10;
	else if(option == 4)
		panningParameter = 10;
	else if(option == 5)
		panningParameter = 10;
	else if(option == 6)
		panningParameter == 10;
	else if(option == 10)
		panningParameter == 10;
	// end setting panning paramter

	assert(cap.isOpened());

	Mat cur, cur_grey;
	Mat prev, prev_grey;

	cap >> prev;//get the first frame.ch
	cvtColor(prev, prev_grey, COLOR_BGR2GRAY);

	// Storing the footage
//	Size s = Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH), (int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));
//	VideoWriter footage("Footage.avi", CV_FOURCC('M','J','P','G'),24,s,true);
	// End storing the footage

	// Step 1 - Get previous to current frame transformation (dx, dy, da) for all frames
	vector <TransformParam> prev_to_cur_transform; // previous to current
	// Accumulated frame to frame transform
	double a = 0;
	double x = 0;
	double y = 0;
	// Step 2 - Accumulate the transformations to get the image trajectory
	vector <Trajectory> trajectory; // trajectory at all frames
	//
	// Step 3 - Smooth out the trajectory using an averaging window
	vector <Trajectory> smoothed_trajectory; // trajectory at all frames
	Trajectory X;//posteriori state estimate
	Trajectory	X_;//priori estimate
	Trajectory P;// posteriori estimate error covariance
	Trajectory P_;// priori estimate error covariance
	Trajectory K;//gain
	Trajectory	z;//actual measurement
	double pstd = 4e-3;//can be changed
	double cstd = 0.25;//can be changed
	Trajectory Q(pstd,pstd,pstd);// process noise covariance
	Trajectory R(cstd,cstd,cstd);// measurement noise covariance
	// Step 4 - Generate new set of previous to current transform, such that the trajectory ends up being the same as the smoothed trajectory
	vector <TransformParam> new_prev_to_cur_transform;
	//
	// Step 5 - Apply the new transformation to the video
	//cap.set(CV_CAP_PROP_POS_FRAMES, 0);
	Mat T(2,3,CV_64F);

	int vert_border = HORIZONTAL_BORDER_CROP * prev.rows / prev.cols; // get the aspect ratio correct
//	VideoWriter outputVideo;
//	outputVideo.open("compare.avi" , CV_FOURCC('X','V','I','D'), 24,cvSize(cur.rows, cur.cols*2+10), true);
	//
	int k=1;
	int max_frames = cap.get(CV_CAP_PROP_FRAME_COUNT);
	Mat last_T;
	Mat prev_grey_,cur_grey_;
	Mat output;

	// UDP Socket configuration
	string servAddress = "192.168.1.255"; // First arg: server address
	unsigned short servPort = Socket::resolveService("12345", "udp");
    UDPSocket sock;
	int jpegqual =  ENCODE_QUALITY; // Compression Parameter
    Mat frame, send;
    vector < uchar > encoded;
    //end UDP Socket configuration

    // UDP Socket config for jittery video
    string servAddressIn = "192.168.1.255";
    unsigned short servPortIn = Socket::resolveService("12346", "udp");
    UDPSocket sockIn;
    int jpegqualIn = ENCODE_QUALITY;
    Mat frameIn, sendIn;
    vector <uchar> encodedIn;
    // end socket config for stabilised video

	while(true) {
		cap >> cur;
		then = cvGetTickCount();
		if(cur.data == NULL) {
			break;
		}
		cvtColor(cur, cur_grey, COLOR_BGR2GRAY);

		vector <Point2f> prev_corner2, cur_corner2;

//		goodFeaturesToTrack(prev_grey, prev_corner, 200, 0.01, 30);

		cuda::GpuMat d_prev_corner;
		cuda::GpuMat d_prev_grey(prev_grey);
		Ptr<cuda::CornersDetector> cudaCornerDetector = cuda::createGoodFeaturesToTrackDetector(CV_8UC1, 500, 0.01, 30);
		cudaCornerDetector->detect(d_prev_grey, d_prev_corner);
		vector <Point2f> prev_corner(d_prev_corner.cols);
		Mat prev_corner_mat(1,d_prev_corner.cols, CV_32FC2, (void*) &prev_corner[0]);
		d_prev_corner.download(prev_corner_mat);

		//we already have d_prev_corner
		//we need to create a d_cur_corner

		//CUDA FLOW
		Ptr<cuda::SparsePyrLKOpticalFlow> pyrLK = cuda::SparsePyrLKOpticalFlow::create();

		cuda::GpuMat d_cur_grey(cur_grey);
		cuda::GpuMat d_cur_corner;
		cuda::GpuMat d_status;
		pyrLK->calc(d_prev_grey, d_cur_grey, d_prev_corner, d_cur_corner, d_status, cv::noArray());

		vector <Point2f> cur_corner(d_cur_corner.cols);
		vector <uchar> status(d_status.cols);
		Mat cur_corner_mat(1,d_cur_corner.cols, CV_32FC2, (void*) &cur_corner[0]);
		d_cur_corner.download(cur_corner_mat);
		Mat status_mat(1,d_status.cols, CV_8UC1,(void*) &status[0]);
		d_status.download(status_mat);
		//END CUDA FLOW

//		//CPU FLOW
//		vector <Point2f> cur_corner;
//		vector <uchar> status;
//		vector <float> err;
//		calcOpticalFlowPyrLK(prev_grey, cur_grey, prev_corner, cur_corner, status, err);
//		//END CPU FLOW

		// weed out bad matches
		for(size_t i=0; i < status.size(); i++) {
			if(status[i]) {
				prev_corner2.push_back(prev_corner[i]);
				cur_corner2.push_back(cur_corner[i]);
			}
		}


		if(prev_corner2.size()>0) {


		// translation + rotation only
		Mat T = estimateRigidTransform(prev_corner2, cur_corner2, false); // false = rigid transform, no scaling/shearing

		// in rare cases no transform is found. We'll just use the last known good transform.
		if(T.data == NULL) {
			last_T.copyTo(T);
		}

		T.copyTo(last_T);

		// decompose T
		double dx = T.at<double>(0,2);
		double dy = T.at<double>(1,2);
		double da = atan2(T.at<double>(1,0), T.at<double>(0,0));
		//
		//prev_to_cur_transform.push_back(TransformParam(dx, dy, da));

		// Graph dx, dy, da values
		graphDx = dx;
		graphDy = dy;
		graphDa = da;
		// end graph values

		out_transform << k << " " << dx << " " << dy << " " << da << endl;
		//
		// Accumulated frame to frame transform
		x += dx;
		y += dy;
		a += da;
		//trajectory.push_back(Trajectory(x,y,a));
		//
		out_trajectory << k << " " << x << " " << y << " " << a << endl;
		//
		z = Trajectory(x,y,a);
		//
		if(k==1){
			// intial guesses
			X = Trajectory(0,0,0); //Initial estimate,  set 0
			P =Trajectory(1,1,1); //set error variance,set 1
		}
		else
		{
			//time update（prediction）
			X_ = X; //X_(k) = X(k-1);
			P_ = P+Q; //P_(k) = P(k-1)+Q;
			// measurement update（correction）
			K = P_/( P_+R ); //gain;K(k) = P_(k)/( P_(k)+R );
			X = X_+K*(z-X_); //z-X_ is residual,X(k) = X_(k)+K(k)*(z(k)-X_(k));
			P = (Trajectory(1,1,1)-K)*P_; //P(k) = (1-K(k))*P_(k);
		}
		//smoothed_trajectory.push_back(X);
		out_smoothed_trajectory << k << " " << X.x << " " << X.y << " " << X.a << endl;
		//-
		// target - current
		double diff_x = X.x - x;//
		double diff_y = X.y - y;
		double diff_a = X.a - a;

		dx = dx + diff_x;
		dy = dy + diff_y;
		da = da + diff_a;

		// new Graph values
		newGraphDx = dx;
		newGraphDy = dy;
		newGraphDa = da;

		// end new Graph values
		//new_prev_to_cur_transform.push_back(TransformParam(dx, dy, da));
		//
		out_new_transform << k << " " << dx << " " << dy << " " << da << endl;
		//
		T.at<double>(0,0) = cos(da);
		T.at<double>(0,1) = -sin(da);
		T.at<double>(1,0) = sin(da);
		T.at<double>(1,1) = cos(da);

		T.at<double>(0,2) = dx;
		T.at<double>(1,2) = dy;

		//CUDA WarpAffine
		cuda::GpuMat d_prev(prev);
		cuda::GpuMat d_output;
		cuda::warpAffine(d_prev, d_output, T, d_prev.size());
		d_output.download(output);
		//END CUDA WarpAffine

		//warpAffine(prev, output, T, cur.size());
		tempPAN = "";
		}
		else {
			output=cur;
			tempPAN = "PANNING";
		}
		output = output(Range(vert_border, output.rows-vert_border), Range(HORIZONTAL_BORDER_CROP, output.cols-HORIZONTAL_BORDER_CROP));

		// Resize output back to cur size, for better side by side comparison
		resize(output, output, cur.size());

		// Now draw the original and stablised side by side for coolness
		Mat canvas = Mat::zeros(cur.rows, cur.cols*2+10, cur.type());

		//Displaying the corners in input and ouptut
		for(int i = 0;i<prev_corner2.size();i++) {
			circle(prev, prev_corner[i], 3, Scalar(0,0,255),CV_FILLED);
//			circle(prev, prev_corner2[i], 3, Scalar(255,255,255), CV_FILLED);
		}
		//end display of corners

		prev.copyTo(canvas(Range::all(), Range(0, output.cols)));
		output.copyTo(canvas(Range::all(), Range(output.cols+10, output.cols*2+10)));

		// If too big to fit on the screen, then scale it down by 2, hopefully it'll fit :)
		if(canvas.cols > 1920) {
			resize(canvas, canvas, Size(canvas.cols/2, canvas.rows/2));
		}
		//outputVideo<<canvas;

		//display fps
		now = cvGetTickCount();
		elapsed_seconds = (double)(now - then) /tickspersecond;

		stringstream s;
		s << 1/elapsed_seconds;

		tempFPS = s.str();
		putText(canvas, tempFPS, Point2f (30,30), FONT_HERSHEY_PLAIN, 2, Scalar(0,0,255,255));
		//end display fps
		//display panning
		putText(canvas, tempPAN, Point2f(300,30), FONT_HERSHEY_PLAIN, 3, Scalar(0,0,255,255));
		//end display panning


		imshow("stabilzed", canvas);
		// Storing Footage
//		footage.write(output);
		//End storing footage

		//SOCKET STREAMING
		frame=output;
		resize(frame,send,Size(FRAME_WIDTH, FRAME_HEIGHT), 0, 0, INTER_LINEAR);
		vector < int > compression_params;
		compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
		compression_params.push_back(jpegqual);

		imencode(".jpg", send, encoded, compression_params);
//		imshow("ServerOUPUT", send);
		int total_pack = 1 + (encoded.size() - 1) / PACK_SIZE;

		int ibuf[1];
		ibuf[0] = total_pack;
		sock.sendTo(ibuf, sizeof(int), servAddress, servPort);
		for (int i = 0; i < total_pack; i++)
			sock.sendTo( & encoded[i * PACK_SIZE], PACK_SIZE, servAddress, servPort);
		//END SOCKET STREAMING

		//SOCKET STREAMING For input
		frameIn=prev;
		resize(frameIn,sendIn,Size(FRAME_WIDTH, FRAME_HEIGHT), 0, 0, INTER_LINEAR);
		vector < int > compression_paramsIn;
		compression_paramsIn.push_back(CV_IMWRITE_JPEG_QUALITY);
		compression_paramsIn.push_back(jpegqualIn);

		imencode(".jpg", sendIn, encodedIn, compression_paramsIn);
//		imshow("ServerOUPUT", send);
		int total_packIn = 1 + (encodedIn.size() - 1) / PACK_SIZE;

		int ibufIn[1];
		ibufIn[0] = total_packIn;
		sockIn.sendTo(ibufIn, sizeof(int), servAddressIn, servPortIn);
		for (int i = 0; i < total_packIn; i++)
			sockIn.sendTo( & encodedIn[i * PACK_SIZE], PACK_SIZE, servAddressIn, servPortIn);
		//END SOCKET STREAMING for input


		// Graph Data streaming
		GraphData gd(graphDx, graphDy, (180*graphDa)/3.14, newGraphDx, newGraphDy, (180*newGraphDa)/3.14);
        ::send(new_socket, &gd, sizeof(GraphData), 0);
		// END graph data streaming

		prev = cur.clone();//cur.copyTo(prev);
		cur_grey.copyTo(prev_grey);

		cout << "Frame: " << k << "/" << max_frames << " - good optical flow: " << prev_corner2.size() << endl;

		k++;

		if(waitKey(10)>0) break;
	}
	return 0;
}


