#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <linux/fb.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <sys/ioctl.h>
#include <unistd.h>

// Struct to hold essential framebuffer information
struct framebuffer_info
{
    uint32_t bits_per_pixel; // Bits per pixel (e.g., 16, 24, 32)
    uint32_t xres_virtual;   // Virtual screen width in pixels
};

// Function to open the framebuffer device and get its information
struct framebuffer_info get_framebuffer_info(const char *framebuffer_device_path);

int main(int argc, char const *argv[])
{
    // --- Argument Check ---
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <framebuffer_device_path> <image_path>" << std::endl;
        std::cerr << "Example: " << argv[0] << " /dev/fb0 ./nycu_logo.bmp" << std::endl;
        return 1;
    }

    const char *fb_path = argv[1];
    const char *image_path = argv[2];

    // --- Step 1: Get Framebuffer Information ---
    // std::cout << "Step 1: Reading framebuffer info from " << fb_path << "..." << std::endl;
    framebuffer_info fb_info = get_framebuffer_info(fb_path);
    // std::cout << "  > Success! Virtual width: " << fb_info.xres_virtual << ", Color depth: " << fb_info.bits_per_pixel << " bpp" << std::endl;

    // --- Step 2: Open Framebuffer Device File ---
    // std::cout << "Step 2: Opening framebuffer device for writing..." << std::endl;
    std::ofstream ofs(fb_path);
    // Confirmation: Check if the file was opened successfully
    if (!ofs)
    {
        std::cerr << "Error: Cannot open framebuffer device file: " << fb_path << std::endl;
        return -1;
    }
    // std::cout << "  > Success!" << std::endl;

    // --- Step 3: Read Image File ---
    // std::cout << "Step 3: Reading image file from " << image_path << "..." << std::endl;
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    // Confirmation: Check if the image was read successfully and is not empty
    if (image.empty())
    {
        std::cerr << "Error: Could not read image file, or the file is empty: " << image_path << std::endl;
        return -1;
    }
    // std::cout << "  > Success! Image size: " << image.cols << "x" << image.rows << std::endl;
    
    // --- Step 4: Convert Image Color Format ---
    // std::cout << "Step 4: Converting image color format to BGR565 (16-bit)..." << std::endl;
    cv::Mat converted_image;
    cv::cvtColor(image, converted_image, cv::COLOR_BGR2BGR565);
    // Confirmation: Check if the converted image is empty
    if (converted_image.empty())
    {
        std::cerr << "Error: Color space conversion failed!" << std::endl;
        return -1;
    }
    // std::cout << "  > Success!" << std::endl;


    // --- Step 5: Write Pixel Data to Framebuffer ---
    // std::cout << "Step 5: Writing pixel data to framebuffer row by row..." << std::endl;
    cv::Size2f image_size = image.size();
    for (int y = 0; y < image_size.height; y++)
    {
        // Calculate the starting position of the current row
        long position = y * fb_info.xres_virtual * (fb_info.bits_per_pixel / 8);
        ofs.seekp(position);
        // Confirmation: Check if the seekp operation was successful
        if (ofs.fail())
        {
            std::cerr << "Error: Failed to seek to position for y=" << y << " in framebuffer!" << std::endl;
            ofs.close();
            return -1;
        }

        // Write the pixel data for the current row directly
        ofs.write(reinterpret_cast<const char*>(converted_image.ptr(y)), image_size.width * (fb_info.bits_per_pixel / 8));
        // Confirmation: Check if the write operation was successful
        if (ofs.fail())
        {
            std::cerr << "Error: Failed to write pixel data for row y=" << y << " to framebuffer!" << std::endl;
            ofs.close();
            return -1;
        }
    }
    // std::cout << "  > Success! All pixel data has been written." << std::endl;

    ofs.close();
    // std::cout << "Program finished." << std::endl;
    return 0;
}

/**
 * @brief Opens the specified framebuffer device and uses an ioctl system call to get its screen info.
 *
 * @param framebuffer_device_path A C-style string to the framebuffer device file path (e.g., "/dev/fb0").
 * @return A framebuffer_info struct containing bits_per_pixel and xres_virtual.
 */
struct framebuffer_info get_framebuffer_info(const char *framebuffer_device_path)
{
    struct framebuffer_info fb_info;
    struct fb_var_screeninfo screen_info;

    int fd = open(framebuffer_device_path, O_RDWR);
    // Confirmation: Check if the open system call was successful
    if (fd == -1)
    {
        perror("Error: Cannot open framebuffer device (open failed)");
        exit(1);
    }

    // Use ioctl to get the variable screen information
    // Confirmation: Check if the ioctl system call was successful
    if (ioctl(fd, FBIOGET_VSCREENINFO, &screen_info) == -1)
    {
        perror("Error: Failed to read variable screen info (ioctl failed)");
        close(fd);
        exit(2);
    }

    // Store the necessary information in our struct
    fb_info.xres_virtual = screen_info.xres_virtual;
    fb_info.bits_per_pixel = screen_info.bits_per_pixel;

    close(fd);
    return fb_info;
};
