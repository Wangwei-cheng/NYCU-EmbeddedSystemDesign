#include <iostream>
#include <string>
#include <vector>
#include <thread> // for sleep_for
#include <chrono> // for milliseconds

#include <fcntl.h>
#include <unistd.h>
#include <termios.h>
#include <stdlib.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <cstring>

#include <opencv2/opencv.hpp>

// --- STB Image 整合 ---
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <termios.h> // for termios
#include <unistd.h>  // for STDIN_FILENO
#include <fcntl.h>   // for fcntl
#include <stdlib.h>  // for atexit
#include <bits/stdc++.h>
#include <fstream>
#include <linux/fb.h>

struct termios orig_termios; // 宣告為全域變數，以便還原函式可以存取

// 還原終端機設定的函式
void restore_terminal() {
    tcsetattr(STDIN_FILENO, TCSANOW, &orig_termios);
}

// 您提供的設定函式
void setup_terminal_for_nonblocking_input() {
    tcgetattr(STDIN_FILENO, &orig_termios);
    // 註冊 restore_terminal 函式，讓它在程式正常結束時自動被呼叫
    atexit(restore_terminal);
    
    struct termios new_termios = orig_termios;
    new_termios.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &new_termios);
    
    // 將標準輸入設為非阻塞
    int old_flags = fcntl(STDIN_FILENO, F_GETFL, 0);
    fcntl(STDIN_FILENO, F_SETFL, old_flags | O_NONBLOCK);
}

cv::Mat imread_with_fallback(const std::string& path) {
    cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
    

    std::cout << "OpenCV failed to load image, trying stb_image fallback..." << std::endl;
    int w, h, c;
    unsigned char* data = stbi_load(path.c_str(), &w, &h, &c, 3); // 強制轉為 3 通道 (BGR)
    if (!data) {
        std::cerr << "Cannot decode image (OpenCV + stb_image both failed): " << path << std::endl;
        return cv::Mat();
    }
    std::cout << "✅ Loaded image using stb_image fallback (" << w << "x" << h << ")." << std::endl;

    // stb_image 讀取的是 RGB 格式，需要轉為 OpenCV 慣用的 BGR
    cv::Mat mat(h, w, CV_8UC3, data);
    cv::cvtColor(mat, mat, cv::COLOR_RGB2BGR);
    cv::Mat clone = mat.clone(); // 必須做一次深拷貝，因為 stb_image 的記憶體會被釋放
    stbi_image_free(data);
    return clone;
}
struct framebuffer_info {
    uint32_t bits_per_pixel;
    uint32_t xres;
    uint32_t yres;
    long line_length; // 每一行的實際位元組數 (stride)，比自己計算更精確
};

framebuffer_info get_framebuffer_info(int fd);

int main(int argc, const char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <advance.png_path>" << std::endl;
        return 1;
    }
    std::string img_path = "./advance.png"; 
    const char* fb_path = "/dev/fb0"; // 固定使用 HDMI

    int fb_fd = open(fb_path, O_RDWR);
    if (fb_fd == -1) { /* ... error handling ... */ return -1; }
    framebuffer_info fb_info = get_framebuffer_info(fb_fd);
    long screensize = fb_info.yres * fb_info.line_length;
    char* fbp = (char*)mmap(0, screensize, PROT_READ | PROT_WRITE, MAP_SHARED, fb_fd, 0);
    if (fbp == MAP_FAILED) { /* ... error handling ... */ return -1; }

    cv::Mat scroll_image = imread_with_fallback(img_path);
    if (scroll_image.empty()) { /* ... error handling ... */ return -1; }

    setup_terminal_for_nonblocking_input();
    atexit(restore_terminal);
    std::cout << "Electronic Scroll Board started. Press 'j' (left), 'l' (right), 'q' (quit)." << std::endl;
    
    int scroll_offset = 0;
    int scroll_dir = 0;
    const int step = 20;

    while (true) {
        int c = getchar();
        if (c == 'q' || c == 'Q') break;
        if (c == 'j' || c == 'J') scroll_dir = -1;
        if (c == 'l' || c == 'L') scroll_dir = 1;
        scroll_offset += step*scroll_dir;
        if (scroll_offset < 0) scroll_offset += scroll_image.cols;
        if (scroll_offset >= scroll_image.cols) scroll_offset -= scroll_image.cols;

        cv::Mat frame_to_display;
        if (scroll_offset + fb_info.xres <= scroll_image.cols) {
            frame_to_display = scroll_image(cv::Rect(scroll_offset, 0, fb_info.xres, fb_info.yres));
        } else {
            int right_part_width = scroll_image.cols - scroll_offset;
            int left_part_width = fb_info.xres - right_part_width;

            cv::Mat part1 = scroll_image(cv::Rect(scroll_offset, 0, right_part_width, fb_info.yres));
            cv::Mat part2 = scroll_image(cv::Rect(0, 0, left_part_width, fb_info.yres));

            cv::hconcat(part1, part2, frame_to_display);
        }

        cv::Mat converted_image;
        switch (fb_info.bits_per_pixel) {
            case 16: cv::cvtColor(frame_to_display, converted_image, cv::COLOR_BGR2BGR565); break;
            case 24: converted_image = frame_to_display; break;
            case 32: cv::cvtColor(frame_to_display, converted_image, cv::COLOR_BGR2BGRA); break;
            default: std::cerr << "Unsupported color depth\n"; return -1;
        }

        size_t bytes_to_copy = fb_info.xres * (fb_info.bits_per_pixel / 8);
        for (int y = 0; y < fb_info.yres; y++) {
            memcpy(fbp + y * fb_info.line_length, converted_image.ptr(y), bytes_to_copy);
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(16));
    }

    munmap(fbp, screensize);
    close(fb_fd);
    return 0;
}

framebuffer_info get_framebuffer_info(int fd) {
    framebuffer_info fb_info;
    fb_var_screeninfo vinfo;
    fb_fix_screeninfo finfo;

    if (ioctl(fd, FBIOGET_VSCREENINFO, &vinfo) == -1) {
        perror("Error reading variable information");
        exit(2);
    }
    // ** 優化: 額外讀取固定資訊以取得 line_length **
    if (ioctl(fd, FBIOGET_FSCREENINFO, &finfo) == -1) {
        perror("Error reading fixed information");
        exit(3);
    }

    fb_info.xres = vinfo.xres;
    fb_info.yres = vinfo.yres;
    fb_info.bits_per_pixel = vinfo.bits_per_pixel;
    fb_info.line_length = finfo.line_length;
    return fb_info;
}