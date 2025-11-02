#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;
using namespace cv;
using namespace cv::face;
using namespace std;
map<int, string> label_names = {
    {0, "Shark"},
    {1, "Wilson"},
};

int main(){
    string test_path = "../data";
    string face_cascade_path = "/home/wilsonw/opencv/opencv-4.x/data/haarcascades/haarcascade_frontalface_default.xml";
    string lbph_model_path = "../../lbph_train/lbph_model.yml";
    string result_name = "../result";

    CascadeClassifier face_cascade;
    if (!face_cascade.load(face_cascade_path)) {
        cerr << "Error: Cannot load Haar cascade classifier." << endl;
    }

    Ptr<LBPHFaceRecognizer> lbph_model = Algorithm::load<LBPHFaceRecognizer>(lbph_model_path);
    if(lbph_model->empty()){
        cerr << "Error: Cannot load LBPH model." << endl;
    }

    if (fs::exists(result_name)){
        fs::remove_all(result_name);
    }
    else{
        fs::create_directory(result_name);
    }

    for (const auto& person_dir : fs::directory_iterator(test_path)) {
        if (!person_dir.is_directory()) continue;
        string person = person_dir.path().filename().string();
        cout << "Processing person: " << person << endl;

        int idx = 0;
        for (const auto& img_file : fs::directory_iterator(person_dir.path())) {
            Mat img = imread(img_file.path().string());
            if (img.empty()) continue;

            Mat img_gray;
            cvtColor(img, img_gray, COLOR_BGR2GRAY);

            vector<Rect> detected_faces;
            face_cascade.detectMultiScale(img_gray, detected_faces, 1.2, 5);

            if (detected_faces.empty()) continue;

            for (const auto& face : detected_faces) {
                Mat face_img = img_gray(face).clone();
                int predicted_label = -1;
                double confidence = 0.0;

                equalizeHist(face_img, face_img);
                lbph_model->predict(face_img, predicted_label, confidence);

                string text;
                if(predicted_label >= 0 && confidence < 80.0){
                    text = label_names[predicted_label] + " (" + cv::format("%.1f", confidence) + ")";
                }
                else{
                    text = "Unknown";
                }

                putText(img, text, Point(face.x, face.y - 10),
                        FONT_HERSHEY_SIMPLEX, 2.0, Scalar(0, 255, 0), 2);
            }

            string img_path = result_name + "/" + person + to_string(idx) + ".jpg";
            imwrite(img_path, img);
            idx++;
        }
    }

    // int correct_predictions = 0;

    // for (size_t i = 0; i < face_list.size(); i++) {
    //     int predicted_label = -1;
    //     double confidence = 0.0;
    //     lbph_model->predict(face_list[i], predicted_label, confidence);

    //     string text;
    //     if(predicted_label >= 0 && confidence < 80.0){
    //         text = label_names[predicted_label] + " (" + cv::format("%.1f", confidence) + ")";
    //     }
    //     else{
    //         text = "Unknown";
    //     }

    //     putText()
    //     // 對照 class_list 計算正確率
    //     if (predicted_label == class_list[i]) {
    //         correct_predictions++;
    //     }

    //     cout << "Image " << i << ": True label = " << class_list[i]
    //          << ", Predicted = " << predicted_label
    //          << ", Confidence = " << confidence << endl;
    // }

    // double accuracy = 0.0;
    // if (!face_list.empty()) {
    //     accuracy = static_cast<double>(correct_predictions) / face_list.size();
    // }

    // cout << "Total images: " << face_list.size() << endl;
    // cout << "Correct predictions: " << correct_predictions << endl;
    // cout << "Accuracy: " << accuracy * 100.0 << "%" << endl;

    return 0;
}