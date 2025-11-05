#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <termios.h> // for termios
#include <unistd.h>  // for STDIN_FILENO
#include <fcntl.h>   // for fcntl
#include <stdlib.h>  // for atexit
#include <algorithm>
#include <csignal>
#include <cstring>
#include <cstdint>
#include <cmath>

#include <linux/fb.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/mman.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

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

int cam_width = 640;
int cam_height = 480;
float cam_fps = 10;

int main ( int argc, const char *argv[] )
{
    if (argc > 1) {
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

    // 目標 framebuffer 的顯示大小 (visible)
    const int fb_width = fb_info.xres;
    const int fb_height = fb_info.yres;
    const int fb_line_len = fb_info.line_length;
    const size_t max_row_bytes = (size_t)fb_line_len;

    int screenshot_id = 0;
    std::string base_path = "/run/media/mmcblk1p1/";
    int session_id = 0;
    std::string session_dir_path;

    cv::Mat frame;

    while (true) {
        session_dir_path = base_path + "screenshots_" + std::to_string(session_id);
        
        // 使用 stat 檢查資料夾是否存在
        struct stat info;
        if (stat(session_dir_path.c_str(), &info) != 0) {
            // stat 不等於 0，表示路徑不存在，我們找到了可用的 ID
            break;
        }
        // 如果路徑存在，則繼續尋找下一個 ID
        session_id++;
    }

    // 建立這個新的、唯一的資料夾
    if (mkdir(session_dir_path.c_str(), 0777) == 0) {
        std::cout << "Saving all screenshots for this session in: " << session_dir_path << std::endl;
    } else {
        perror("Error creating directory");
    }
    
    int screenshot_id_in_folder = 0;

    setup_terminal_for_nonblocking_input();
    while ( true )
    {
        camera >> frame;
         if (frame.empty()) {
            std::cerr << "Error:No Image , capture failed" << std::endl;
            break;
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

        cv::Mat background = cv::Mat::zeros(cv::Size(fb_info.xres_virtual, fb_info.yres_virtual), display_frame.type());
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

        int cvkey = getchar();
        if(cvkey == 'c' || cvkey == 'C'){
            std::string filename = "screenshot_" + std::to_string(screenshot_id_in_folder) + ".bmp";
            std::string full_path = session_dir_path + "/" + filename;
            cv::imwrite(full_path, frame);
            std::cout << "Saved: " << full_path << std::endl;
                
            screenshot_id_in_folder++;
        }
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