#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;
using namespace cv;
using namespace std;

// 標籤對應
std::map<std::string, int> label_names = {
    {"Shark", 0},
    {"Wilson", 1},
};

string train_path = "./data";
string face_cascade_path = "./haarcascades/haarcascade_frontalface_default.xml";
string face_img_path = "./face";
string model_name = "lbph_model_class.yml";

int main() {
    vector<Mat> face_list;
    vector<int> class_list;

    CascadeClassifier face_cascade;
    if (!face_cascade.load(face_cascade_path)) {
        cerr << "Error: Cannot load Haar cascade classifier." << std::endl;
    }

    if (fs::exists(face_img_path)){
        fs::remove_all(face_img_path);
    }
    fs::create_directory(face_img_path);

    for (const auto& person_dir : fs::directory_iterator(train_path)) {
        if (!person_dir.is_directory()) continue;
        string person = person_dir.path().filename().string();
        cout << "Processing person: " << person << endl;

        int idx = label_names[person];

        for (const auto& img_file : fs::directory_iterator(person_dir.path())) {
            Mat img = imread(img_file.path().string(), IMREAD_GRAYSCALE);
            if (img.empty()) continue;

            vector<Rect> detected_faces;
            face_cascade.detectMultiScale(img, detected_faces, 1.2, 5);

            if (detected_faces.empty()) continue;

            int num = 0;
            for (const auto& face_rect : detected_faces) {
                // if (face_rect.width < 100)  continue;

                Mat face_img = img(face_rect).clone();
                equalizeHist(face_img, face_img);
                resize(face_img, face_img, Size(100, 100));
                face_list.push_back(face_img);
                class_list.push_back(idx);
                imwrite(face_img_path + "/" + person + "_" + img_file.path().stem().string() + "_" + to_string(num) + ".jpg", face_img);
                num++;
            }
        }
    }

    // 建立 LBPH 臉部辨識器
    Ptr<face::LBPHFaceRecognizer> face_recognizer = face::LBPHFaceRecognizer::create();
    face_recognizer->train(face_list, class_list);

    // 儲存模型
    face_recognizer->save("./" + model_name);
    cout << "Training complete. Model saved to " << model_name << endl;

    return 0;
}