# VideoStabilizationServer

Store the sample files in a folder and edit the paths accordingly for "testVideos".
The videos have to be encoded using mjpeg without audio in .avi containers. 

To convert videos using ffmpeg : 
ffmpeg -i inputfile.[mp4|mkv|mov|etc.] -c:v mjpeg -an outputfile.avi

Edit the the config file according to the sample videos to be streamed.