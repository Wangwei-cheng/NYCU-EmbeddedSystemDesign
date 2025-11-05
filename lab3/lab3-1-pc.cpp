#include <iostream>
#include <unistd.h>
#include <stdlib.h>
#include <algorithm>
#include <csignal>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>

std::map<int, std::string> label_names = {
    {0, "Shark"},
    {1, "Wilson"},
};

void sigint_handler(int)
{
    std::cerr << "\nReceived signal, exit!\n";
    exit(0);
}

int main ( int argc, const char *argv[] )
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model_path>" << std::endl;
        return 1;
    }
    std::string model_path = argv[1];
    std::string face_cascade_path = "./haarcascades/haarcascade_frontalface_default.xml";
    
    std::signal(SIGINT, sigint_handler);

    cv::VideoCapture camera(2);
    if( !camera.isOpened() )
    {
        std::cerr << "Could not open video device." << std::endl;
        return 1;
    }

    camera.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M','J','P','G'));
    camera.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    camera.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    camera.set(cv::CAP_PROP_FPS, 30);

    // 載入 Haar Cascade 模型
    cv::CascadeClassifier face_cascade;
    if (!face_cascade.load(face_cascade_path)) {
        std::cerr << "Error: Cannot load Haar cascade classifier." << std::endl;
        return 1;
    }

    // === 載入 LBPH 模型 ===
    cv::Ptr<cv::face::LBPHFaceRecognizer> recognizer = cv::face::LBPHFaceRecognizer::create();
    try {
        recognizer->read(model_path);
    } catch (...) {
        std::cerr << "Warning: Could not load LBPH model. Recognition will be skipped." << std::endl;
    }

    cv::Mat frame;      // variable to store the frame get from video stream

    while ( true )
    {
        camera >> frame;
         if (frame.empty()) {
            std::cerr << "Error:No Image , capture failed" << std::endl;
            break;
        }

        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::equalizeHist(gray, gray);

        std::vector<cv::Rect> faces;
        face_cascade.detectMultiScale(gray, faces, 1.1, 6);

        for (auto &face : faces) {
            cv::rectangle(frame, face, cv::Scalar(0, 255, 0), 2);

            cv::Mat faceROI = gray(face);
            cv::resize(faceROI, faceROI, cv::Size(100, 100));

            int label;
            double confidence = 0.0;
            if (!recognizer.empty()) {
                recognizer->predict(faceROI, label, confidence);
            }

            std::string text;

            if (label >= 0) {
                text = label_names[label];
            }

            cv::putText(frame, text + ", confidence: " + std::to_string(confidence), cv::Point(face.x, face.y - 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
        }

        cv::imshow("Face", frame);
        if(cv::waitKey(1) == 27) break;
    }
    
    camera.release();

    return 0;
}
