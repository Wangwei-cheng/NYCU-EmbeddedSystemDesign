#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <unistd.h>
#include <stdlib.h>
#include <algorithm>
#include <csignal>
#include <cstring>
#include <cstdint>
#include <string>

#include <linux/fb.h>
#include <sys/ioctl.h>
#include <sys/mman.h>

#include <opencv2/opencv.hpp>

struct framebuffer_info
{
    uint32_t bits_per_pixel;    // depth of framebuffer
    uint32_t xres_virtual;      // 記憶體中每一列實際保留的pixel數，包含為panning或其他所保留的空間
    uint32_t yres_virtual;      // (visible resolution就沒有包含這些，所以通常較小)
    uint32_t xres;
    uint32_t yres;
    uint32_t line_length;       // bytes per row in framebuffer (包含 padding)
    size_t smem_len;            // total framebuffer memory len
};

struct framebuffer_info get_framebuffer_info (int fd_fb);

int fd_fb_global = -1;
uint8_t *fb_ptr_global = nullptr;
size_t fb_size_global = 0;

void cleanup_and_exit(int code)
{
    if (fb_ptr_global && fb_ptr_global != (uint8_t*)MAP_FAILED) {
        munmap(fb_ptr_global, fb_size_global);
        fb_ptr_global = nullptr;
    }
    if (fd_fb_global >= 0) {
        close(fd_fb_global);
        fd_fb_global = -1;
    }
    std::exit(code);
}

void sigint_handler(int)
{
    std::cerr << "\nReceived signal, cleaning up...\n";
    cleanup_and_exit(0);
}

std::string face_cascade_path = "./haarcascades/haarcascade_frontalface_default.xml";
int cam_width = 640;
int cam_height = 480;
float cam_fps = 10;

int main ( int argc, const char *argv[] )
{
    if (argc >= 4) {
        cam_width = atoi(argv[1]);
        cam_height = atoi(argv[2]);
        cam_fps = atoi(argv[3]);
    }

    std::signal(SIGINT, sigint_handler);

    int fd_fb = open("/dev/fb0", O_RDWR);
    if (fd_fb < 0) {
        std::cerr << "Error: Could not open framebuffer device /dev/fb0" << std::endl;
        exit(1);
    }
    fd_fb_global = fd_fb;

    framebuffer_info fb_info = get_framebuffer_info(fd_fb);
    size_t fb_size = fb_info.smem_len;
    if (fb_size == 0) {
        fb_size = (size_t)fb_info.yres_virtual * fb_info.line_length;
    }
    fb_size_global = fb_size;

    uint8_t *fb_ptr = (uint8_t*)mmap(nullptr, fb_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_fb, 0);
    if (fb_ptr == MAP_FAILED){
        std::cerr << "Error: mmap failed"<< std::endl;
        close(fd_fb);
        exit(1); // Exit if mmap failed
    }
    fb_ptr_global = fb_ptr;

    cv::VideoCapture camera(2);
    if( !camera.isOpened() )
    {
        std::cerr << "Could not open video device." << std::endl;
        cleanup_and_exit(1);
    }
    camera.set(cv::CAP_PROP_FRAME_WIDTH, cam_width);
    camera.set(cv::CAP_PROP_FRAME_HEIGHT, cam_height);
    camera.set(cv::CAP_PROP_FPS, cam_fps);

    // 載入 Haar Cascade 模型
    cv::CascadeClassifier face_cascade;
    if (!face_cascade.load(face_cascade_path)) {
        std::cerr << "Error: Cannot load Haar cascade classifier." << std::endl;
        cleanup_and_exit(1);
    }

    // 目標 framebuffer 的顯示大小 (visible)
    const int fb_width = fb_info.xres;
    const int fb_height = fb_info.yres;
    const int fb_line_len = fb_info.line_length;
    const size_t max_row_bytes = (size_t)fb_line_len;

    cv::Mat frame;      // variable to store the frame get from video stream

    while ( true )
    {
        camera >> frame;
         if (frame.empty()) {
            std::cerr << "Error:No Image , capture failed" << std::endl;
            break;
        }

        // 人臉偵測
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::equalizeHist(gray, gray);

        cv::Mat small_gray;
        const double small_scale = 2.0;
        cv::resize(
            gray, small_gray, 
            cv::Size(), 1.0 / small_scale, 1.0 / small_scale
        );

        std::vector<cv::Rect> faces;
        cv::Size minSize(gray.cols / 20, gray.rows / 20);
        cv::Size maxSize(gray.cols / 2, gray.rows / 2);
        face_cascade.detectMultiScale(
            small_gray, faces,
            1.1, 6, 0, minSize, maxSize
        );

        for (auto &face : faces) {
            face.x = cvRound(face.x * small_scale);
            face.y = cvRound(face.y * small_scale);
            face.width = cvRound(face.width * small_scale);
            face.height = cvRound(face.height * small_scale);
            cv::rectangle(frame, face, cv::Scalar(0, 255, 0), 2);
        }

        double scale_x = (double)fb_width / (double)frame.cols;
        double scale_y = (double)fb_height / (double)frame.rows;
        double scale = std::min(scale_x, scale_y);
        if (scale <= 0) scale = 1.0;
        int new_w = std::max(1, (int)std::round(frame.cols * scale));
        int new_h = std::max(1, (int)std::round(frame.rows * scale));

        cv::Mat display_frame;
        if (new_w != frame.cols || new_h != frame.rows) {
            cv::resize(frame, display_frame, cv::Size(new_w, new_h), 0, 0, cv::INTER_AREA);
        } else {
            display_frame = frame;
        }

        cv::Mat background = cv::Mat::zeros(cv::Size(fb_width, fb_height), display_frame.type());
        int x_offset = (fb_width - display_frame.cols) / 2;
        int y_offset = (fb_height - display_frame.rows) / 2;
        if (x_offset < 0) x_offset = 0;
        if (y_offset < 0) y_offset = 0;
        cv::Rect roi(x_offset, y_offset, display_frame.cols, display_frame.rows);
        display_frame.copyTo(background(roi));

        cv::Mat converted_image;
        cv::cvtColor(background, converted_image, cv::COLOR_BGR2BGR565);
        
        size_t row_bytes = (size_t)fb_width * 2;
        if (row_bytes > max_row_bytes) row_bytes = max_row_bytes; // safety

        for (int y = 0; y < fb_height; ++y) {
            uint8_t *dst_row = fb_ptr + (size_t)y * fb_line_len;
            std::memset(dst_row, 0, fb_line_len);   // 清除整列 (包含 padding) 以避免殘留像素
            const uint8_t *src_row = converted_image.ptr<uint8_t>(y);
            std::memcpy(dst_row, src_row, row_bytes);
        }

        usleep(1000);
    }
    
    camera.release();
    cleanup_and_exit(0);

    return 0;
}

struct framebuffer_info get_framebuffer_info (int fd_fb)
{
    struct framebuffer_info fb_info;
    struct fb_var_screeninfo screen_var_info;
    struct fb_fix_screeninfo screen_fix_info;

    // 透過 ioctl() 系統呼叫取得螢幕的可變資訊 (variable screen info)
    // FBIOGET_VSCREENINFO 這個指令會讓核心把資訊填入 screen_var_info 結構中
    if (ioctl(fd_fb, FBIOGET_VSCREENINFO, &screen_var_info) < 0) {
        std::cerr << "Error: Could not get screen info" << std::endl;
        close(fd_fb);
        exit(1);
    }

    // 透過 ioctl() 系統呼叫取得螢幕的不可變資訊 (fix screen info)
    // FBIOGET_FSCREENINFO 這個指令會讓核心把資訊填入 screen_fix_info 結構中
    if (ioctl(fd_fb, FBIOGET_FSCREENINFO, &screen_fix_info) < 0) {
        std::cerr << "Error: Could not get screen info" << std::endl;
        close(fd_fb);
        exit(1);
    }

    // 從 screen_info 中取出我們需要的資訊，存入要回傳的 fb_info 中
    fb_info.xres_virtual = screen_var_info.xres_virtual;
    fb_info.yres_virtual = screen_var_info.yres_virtual;
    fb_info.bits_per_pixel = screen_var_info.bits_per_pixel;
    fb_info.xres = screen_var_info.xres;
    fb_info.yres = screen_var_info.yres;
    fb_info.line_length = screen_fix_info.line_length;
    fb_info.smem_len = screen_fix_info.smem_len;

    return fb_info;
};