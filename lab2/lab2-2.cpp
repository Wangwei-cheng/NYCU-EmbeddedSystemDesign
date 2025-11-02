#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <linux/fb.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <sys/ioctl.h>
#include <sys/stat.h> 
#include <sstream>  
#include <string>
#include <termios.h> // for termios
#include <unistd.h>  // for STDIN_FILENO
#include <fcntl.h>   // for fcntl
#include <stdlib.h>  // for atexit
#include <bits/stdc++.h>

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
    uint32_t xres_virtual;      // how many pixel in a row in virtual screen
    uint32_t yres_virtual;
};

struct framebuffer_info get_framebuffer_info ( const char *framebuffer_device_path );

std::string to_string(int num) {
    std::ostringstream ss;
    ss << num;
    return ss.str();
}

int main ( int argc, const char *argv[] )
{
    // variable to store the frame get from video stream
    cv::Mat frame;
    const char *fb_path = "/dev/fb0";
    framebuffer_info fb_info = get_framebuffer_info(fb_path);
    uint32_t x = fb_info.xres_virtual;
    uint32_t bit = fb_info.bits_per_pixel;
    uint32_t y = fb_info.yres_virtual;
    uint32_t rx ,lx;
    lx = (x-y*1.33)/2;
    rx = x - (x-y*1.33)/2;

    std::ofstream ofs(fb_path, std::ios::binary);
    if (!ofs.is_open()) {
        std::cerr << "Error: Could not open framebuffer device " << fb_path << std::endl;
        return 1;
    }
    cv::VideoCapture camera ( 2 );
    if( !camera.isOpened() )
    {
        std::cerr << "Could not open video device." << std::endl;
        return 1;
    }

    int screenshot_id = 0;
    std::string base_path = "/run/media/mmcblk1p1/";
    int session_id = 0;
    std::string session_dir_path;

    while (true) {
        session_dir_path = base_path + "screenshots_" + to_string(session_id);
        
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
        cv::Mat display_frame;
        cv::Size frame_size;
       
        frame_size = frame.size();
        if (frame.cols > fb_info.xres_virtual || frame.rows > fb_info.yres_virtual)
        {
            double scale_x = (double)fb_info.xres_virtual / frame.cols;
            double scale_y = (double)fb_info.yres_virtual / frame.rows;
            double scale = std::min(scale_x, scale_y); 

            cv::resize(frame, display_frame, cv::Size(), scale, scale, cv::INTER_AREA);
        }
        else
        {
            display_frame = frame;
        }

        cv::Mat background = cv::Mat::zeros(cv::Size(fb_info.xres_virtual, fb_info.yres_virtual), display_frame.type());
        int x_offset = (fb_info.xres_virtual - display_frame.cols) / 2;
        int y_offset = (fb_info.yres_virtual - display_frame.rows) / 2;

        cv::Rect roi = cv::Rect(x_offset, y_offset, display_frame.cols, display_frame.rows);

        display_frame.copyTo(background(roi));
        cv::Mat converted_image;
        cv::cvtColor(background, converted_image, cv::COLOR_BGR2BGR565);
        for ( int y = 0; y < fb_info.yres_virtual; y++ )
        {
            long position = y * fb_info.xres_virtual * (fb_info.bits_per_pixel / 8);
            ofs.seekp(position);
            ofs.write(reinterpret_cast<const char*>(converted_image.ptr(y)), fb_info.xres_virtual * (fb_info.bits_per_pixel / 8));
            // move to the next written position of output device framebuffer by "std::ostream::seekp()"
            // http://www.cplusplus.com/reference/ostream/ostream/seekp/

            // write to the framebuffer by "std::ostream::write()"
            // you could use "cv::Mat::ptr()" to get the pointer of the corresponding row.
            // you also need to cacluate how many bytes required to write to the buffer
            // http://www.cplusplus.com/reference/ostream/ostream/write/
            // https://docs.opencv.org/3.4.7/d3/d63/classcv_1_1Mat.html#a13acd320291229615ef15f96ff1ff738
        }
        ofs.flush();

        int cvkey = getchar();
        if(cvkey == 'c' || cvkey == 'C'){
            std::string filename = "screenshot_" + to_string(screenshot_id_in_folder) + ".bmp";
            std::string full_path = session_dir_path + "/" + filename;
            cv::imwrite(full_path, frame);
            std::cout << "Saved: " << full_path << std::endl;
                
            screenshot_id_in_folder++;
            // mkdir(dir_path.c_str(), 0777);
            // std::string file_path = dir_path + "/screenshot.jpg";
            // cv::imwrite(file_path, frame);/.
            // screenshot_id++;
            //std::cout << "Saved: " << filename.str() << std::endl;
        }
    }
    ofs.close();
    camera.release ( );

    return 0;
}

struct framebuffer_info get_framebuffer_info ( const char *framebuffer_device_path )
{
    struct framebuffer_info fb_info;
    struct fb_var_screeninfo screen_info;

    // 開啟 Framebuffer 設備檔案
    int fd = open(framebuffer_device_path, O_RDWR);
    if (fd < 0) {
        std::cerr << "Error: Could not open framebuffer device " << framebuffer_device_path << std::endl;
        exit(1); // Exit if framebuffer cannot be opened
    }

    // 透過 ioctl() 系統呼叫取得螢幕的可變資訊 (variable screen info)
    // FBIOGET_VSCREENINFO 這個指令會讓核心把資訊填入 screen_info 結構中
    if (ioctl(fd, FBIOGET_VSCREENINFO, &screen_info) < 0) {
        std::cerr << "Error: Could not get screen info" << std::endl;
        close(fd);
        exit(1);
    }

    // 從 screen_info 中取出我們需要的資訊，存入要回傳的 fb_info 中
    fb_info.xres_virtual = screen_info.xres_virtual;
    fb_info.yres_virtual = screen_info.yres_virtual;
    fb_info.bits_per_pixel = screen_info.bits_per_pixel;

    // 關閉設備檔案
    close(fd);

    return fb_info;
};