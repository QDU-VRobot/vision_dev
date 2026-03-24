#include <memory>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/persistence.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <glob.h>

using namespace cv;
using namespace std;
cv::Point3d rotationMatrixToEulerAngles(const cv::Mat &R);

//pitch轴是否有误差
#define R_error 
const double R_err = 1.4; //cm

int main()
{
    // 棋盘参数设置
    const Size BOARD_SIZE(11, 8);  // 棋盘内角点数量 (列数, 行数)
    const double SQUARE_SIZE = 2.0;  // 每个方格的实际尺寸 (厘米)
    
    string config_path = "../Data/Calibration_R_T.yaml";
    string image_path = "../Data/images/*.png";
    
    //
    //加载存储数据的YAML文件
    FileStorage fs;
    if (!fs.open(config_path, FileStorage::READ)) {
        cerr << "Error: Failed to open YAML file: " << config_path << endl;
        return -1;  // 失败时返回
    } else {
        cout << "open YAML yes" << endl;
    }


    // 创建棋盘的世界坐标系坐标点
    vector<Point3f> objectPoints;
    for (int i = 0; i < BOARD_SIZE.height; i++) {
        for (int j = 0; j < BOARD_SIZE.width; j++) {
            objectPoints.push_back(Point3f(j * SQUARE_SIZE, i * SQUARE_SIZE, 0));
        }
    }
    
    // 存储所有图像的角点坐标和对应的世界坐标
    vector<vector<Point2f>> imagePointsAll;
    vector<vector<Point3f>> objectPointsAll;
    
    // 获取图像文件列表
    vector<String> imageFiles;
    vector<String> havChessBFiles;
    glob(image_path, imageFiles);
    
    if (imageFiles.empty()) {
        cout << "错误: 在images文件夹中没有找到PNG图片文件!" << endl;
        return -1;
    }
    
    cout << "找到 " << imageFiles.size() << " 张图片" << endl;
    
    Size imageSize;
    int validImages = 0;
    
    // 处理每张图片
    for (size_t i = 0; i < imageFiles.size(); i++) {
        Mat image = imread(imageFiles[i]);
        if (image.empty()) {
            cout << "无法读取图片: " << imageFiles[i] << endl;
            continue;
        }
        
        // 转换为灰度图
        Mat gray;
        cvtColor(image, gray, COLOR_BGR2GRAY);
        imshow("ia",gray);
        waitKey(0);
        
        if (imageSize.width == 0) {
            imageSize = gray.size();
        }
        
        // 查找棋盘角点
        vector<Point2f> corners;
        bool found = findChessboardCorners(gray, BOARD_SIZE, corners,
                                         CALIB_CB_ADAPTIVE_THRESH | 
                                         CALIB_CB_NORMALIZE_IMAGE |
                                         CALIB_CB_FAST_CHECK);
        
        if (found) {
            // 亚像素精确化角点位置
            cornerSubPix(gray, corners, Size(11, 11), Size(-1, -1),
                        TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.1));
            
            // 存储角点
            imagePointsAll.push_back(corners);
            objectPointsAll.push_back(objectPoints);

            //记录图片的名字
            havChessBFiles.push_back(imageFiles[i]);
            validImages++;
            
            // 绘制角点（可选，用于验证）
            drawChessboardCorners(image, BOARD_SIZE, corners, found);
            
            cout << "图片 " << i + 1 << "/" << imageFiles.size() 
                 << " 处理成功: " << imageFiles[i] << endl;
            
            // 显示结果（可选）
            Mat resized;
            resize(image, resized, Size(800, 600));
            imshow("棋盘角点检测", resized);
            waitKey(100);  // 短暂显示
        } else {
            cout << "图片 " << i + 1 << "/" << imageFiles.size() 
                 << " 未找到棋盘: " << imageFiles[i] << endl;
        }
    }
    
    destroyAllWindows();
    
    if (validImages < 3) {
        cout << "错误: 需要至少3张有效的棋盘图片进行标定，当前只有 " 
             << validImages << " 张" << endl;
        return -1;
    }
    
    cout << "\n开始相机标定，使用 " << validImages << " 张有效图片..." << endl;
    
    // 相机标定
    Mat cameraMatrix = Mat::eye(3, 3, CV_64F);
    Mat distCoeffs = Mat::zeros(8, 1, CV_64F);
    vector<Mat> rvecs, tvecs;
    
    double rms = calibrateCamera(objectPointsAll, imagePointsAll, imageSize,
                                cameraMatrix, distCoeffs, rvecs, tvecs);
    
    cout << "\n=== 相机标定结果 ===" << endl;
    cout << "RMS重投影误差: " << rms << " 像素" << endl;
    cout << "\n内参矩阵 (Camera Matrix):" << endl;
    cout << cameraMatrix << endl;
    cout << "\n畸变系数 (Distortion Coefficients):" << endl;
    cout << distCoeffs << endl;
    
    // // 保存标定结果到文件
    // FileStorage fs("camera_calibration.yml", FileStorage::WRITE);
    // fs << "camera_matrix" << cameraMatrix;
    // fs << "distortion_coefficients" << distCoeffs;
    // fs << "image_width" << imageSize.width;
    // fs << "image_height" << imageSize.height;
    // fs << "rms_error" << rms;
    // fs << "valid_images" << validImages;
    // fs.release();
    
    // cout << "\n标定结果已保存到 camera_calibration.yml" << endl;
    
    // 计算标定精度评估
    vector<float> perViewErrors;
    double totalError = 0;
    
    for (size_t i = 0; i < objectPointsAll.size(); i++) {
        vector<Point2f> projectedPoints;
        projectPoints(objectPointsAll[i], rvecs[i], tvecs[i], 
                     cameraMatrix, distCoeffs, projectedPoints);
        
        double error = norm(imagePointsAll[i], projectedPoints, NORM_L2);
        perViewErrors.push_back((float)(error / objectPointsAll[i].size()));
        totalError += error * error;
    }
    
    double meanError = sqrt(totalError / (validImages * BOARD_SIZE.width * BOARD_SIZE.height));
    cout << "平均重投影误差: " << meanError << " 像素" << endl;
    
    cout << "\n=== 标定完成 ===" << endl;
    cout << "建议: RMS误差小于1.0像素表示标定质量良好" << endl;

    // 开始手眼标定
    vector<Mat> Rs_world_to_camera,Ts_world_to_camera;
    Ts_world_to_camera = tvecs;
    for (size_t i = 0; i < rvecs.size(); i++) {
        Mat R;
        Rodrigues(rvecs[i], R);
        Rs_world_to_camera.push_back(R);
    }

    vector<Mat> Rs_base_to_hand,Ts_base_to_hand;
    for(size_t i=0;i<havChessBFiles.size();i++)
    {
        if(havChessBFiles[i].length()<5)
        {
            cerr<<"错误: 文件名不合法: " << havChessBFiles[i] << endl;
            return 0;
        }

        string key = havChessBFiles[i].substr(15, havChessBFiles[i].length() );
        key = key.substr(0,key.length()-4);
        double tdata[3]={0.,0.,0.};
        Mat R,T(3,1,CV_64F,tdata);
        fs[key] >> R;
        std::cout<<R<< key<<"\n";
        Rs_base_to_hand.push_back(R);

        #ifdef R_error
        auto EulerAngles = rotationMatrixToEulerAngles(R);
        std::cout<<"EulerAngles: "<<EulerAngles.x<<" "<<EulerAngles.y<<" "<<EulerAngles.z<<"\n";
        double theta_pitch = EulerAngles.y;
        double theta_yaw = EulerAngles.x;

        double z_err = -( R_err * sin(theta_pitch) );
        double L_err = R_err * (1 - cos(theta_pitch) );
        double x_err = L_err * cos(theta_yaw);
        double y_err = L_err * sin(theta_yaw);
        T = (Mat_<double>(3,1) << x_err, y_err, z_err);
        std::cout<<"误差修正: "<<x_err<<" "<<y_err<<" "<<z_err<<"\n";
        #endif

        Ts_base_to_hand.push_back(T);
    }
    
    Mat R_hand_to_cam_out, T_hand_to_cam_out,
        R_base_to_world_out, T_base_to_world_out;

    calibrateRobotWorldHandEye(Rs_world_to_camera, Ts_world_to_camera,
                               Rs_base_to_hand, Ts_base_to_hand,
                               R_base_to_world_out, T_base_to_world_out,
                               R_hand_to_cam_out, T_hand_to_cam_out,
                               CALIB_ROBOT_WORLD_HAND_EYE_SHAH
                               );

    cout<<"----------------------------------"<<endl;
    cout<<"手眼标定完成："<<endl;

    // 1. 计算眼到手的旋转矩阵 (转置)
    cv::Mat R_cam_to_hand = R_hand_to_cam_out.t();

    // 2. 计算眼到手的平移向量 (-R^T * t)
    // 注意：这里必须用矩阵乘法，不能直接减
    cv::Mat T_cam_to_hand = -R_cam_to_hand * T_hand_to_cam_out;

    cout << "手眼标定完成：" << endl;
    cout << "手到眼的旋转矩阵：" << "\n" << R_hand_to_cam_out << endl;
    cout << "手到眼的平移向量：" << "\n" << T_hand_to_cam_out << endl;

    cout << "眼到手的旋转矩阵： " << "\n" << R_cam_to_hand << endl;
    cout << "眼到手的平移向量：" << "\n" << T_cam_to_hand << endl;


    fs.release(); // 关闭文件


    cout << "----------------------------------" << endl;
    cout << "手眼标定完成，结果已输出。" << endl;

    // --- 修正后的误差分析代码 ---
// --- 修正后的误差分析代码 ---
    cout << "正在计算手眼标定误差..." << endl;

    // 1. 构造标定出的 手->眼 变换矩阵 (T_Hand_to_Cam)
    Mat T_hand_to_cam = Mat::eye(4, 4, CV_64F);
    R_hand_to_cam_out.copyTo(T_hand_to_cam(Rect(0, 0, 3, 3)));
    T_hand_to_cam_out.copyTo(T_hand_to_cam(Rect(3, 0, 1, 3)));

    vector<Point3f> board_positions_in_base;
    Point3f mean_position(0, 0, 0);
    vector<Mat> R_base_to_worlds;

    for (size_t i = 0; i < Rs_world_to_camera.size(); i++) {
        // A. 构造 基座->手
        Mat T_base_to_hand_i = Mat::eye(4, 4, CV_64F);
        Rs_base_to_hand[i].copyTo(T_base_to_hand_i(Rect(0, 0, 3, 3)));
        Ts_base_to_hand[i].copyTo(T_base_to_hand_i(Rect(3, 0, 1, 3)));

        // B. 构造 世界->相机
        Mat T_world_to_cam_i = Mat::eye(4, 4, CV_64F);
        Mat R_w2c;
        Rodrigues(rvecs[i], R_w2c);
        R_w2c.copyTo(T_world_to_cam_i(Rect(0, 0, 3, 3)));
        tvecs[i].copyTo(T_world_to_cam_i(Rect(3, 0, 1, 3)));

        // C. 核心修正：严格按照物理链条求解 World -> Base
        // T_world_to_base = T_base_to_hand.inv() * T_hand_to_cam.inv() * T_world_to_cam
        Mat T_hand_to_base_i = T_base_to_hand_i.inv();
        Mat T_cam_to_hand = T_hand_to_cam.inv();
        
        Mat T_world_to_base_calc = T_hand_to_base_i * T_cam_to_hand * T_world_to_cam_i;

        // D. 提取平移并统计
        Mat pos_mat = T_world_to_base_calc(Rect(3, 0, 1, 3));
        Point3f pos(pos_mat.at<double>(0), pos_mat.at<double>(1), pos_mat.at<double>(2));
        board_positions_in_base.push_back(pos);
        mean_position += pos;

        // E. 提取旋转并统计
        Mat R_world_to_base_calc = T_world_to_base_calc(Rect(0, 0, 3, 3));
        R_base_to_worlds.push_back(R_world_to_base_calc.t());
    }

    // --- 统计平移误差 ---
    mean_position.x /= board_positions_in_base.size();
    mean_position.y /= board_positions_in_base.size();
    mean_position.z /= board_positions_in_base.size();

    double total_dist_err = 0, max_dist_err = 0;
    for (const auto& pos : board_positions_in_base) {
        double err = norm(pos - mean_position);
        total_dist_err += err;
        if (err > max_dist_err) max_dist_err = err;
    }
    double mean_dist_err = total_dist_err / board_positions_in_base.size();

    // --- 统计旋转误差 (带单轴拆解) ---
    double total_rot_err = 0, max_rot_err = 0;
    double total_yaw_err = 0, total_pitch_err = 0, total_roll_err = 0;

    Mat R_base_to_world_truth = R_base_to_world_out;

    cout << "\n--- 单帧角度误差拆解明细 ---" << endl;
    for (size_t i = 0; i < R_base_to_worlds.size(); i++) {
        Mat R_diff = R_base_to_worlds[i] * R_base_to_world_truth.t();

        // 计算总空间误差
        Mat rvec_diff;
        Rodrigues(R_diff, rvec_diff);
        double err_deg = norm(rvec_diff) * 180.0 / CV_PI;
        total_rot_err += err_deg;

        // 计算欧拉角单轴误差
        Point3d euler_diff = rotationMatrixToEulerAngles(R_diff);
        double yaw_err   = abs(euler_diff.x * 180.0 / CV_PI);
        double pitch_err = abs(euler_diff.y * 180.0 / CV_PI);
        double roll_err  = abs(euler_diff.z * 180.0 / CV_PI);

        total_yaw_err   += yaw_err;
        total_pitch_err += pitch_err;
        total_roll_err  += roll_err;

        cout << "图片 " << i + 1 << " 误差 -> "
             << "Yaw: " << yaw_err << "° | "
             << "Pitch: " << pitch_err << "° | "
             << "Roll: " << roll_err << "°  (总空间角: " << err_deg << "°)" << endl;
    }

    double mean_rot_err   = total_rot_err / R_base_to_worlds.size();
    double mean_yaw_err   = total_yaw_err / R_base_to_worlds.size();
    double mean_pitch_err = total_pitch_err / R_base_to_worlds.size();
    double mean_roll_err  = total_roll_err / R_base_to_worlds.size();

    cout << "\n=== 云台标定精度评估 ===" << endl;
    cout << "标定板中心 (基座坐标系): [" << mean_position.x << ", " 
         << mean_position.y << ", " << mean_position.z << "] mm" << endl;
    cout << "-----------------------" << endl;
    cout << "平均平移误差: " << mean_dist_err << " mm" << endl;
    cout << "最大平移误差: " << max_dist_err << " mm" << endl;
    cout << "-----------------------" << endl;
    cout << "平均总空间误差: " << mean_rot_err << " 度 (最坏情况上限)" << endl;
    cout << "  -> 平均 Yaw (偏航) 误差: " << mean_yaw_err << " 度" << endl;
    cout << "  -> 平均 Pitch (俯仰) 误差: " << mean_pitch_err << " 度" << endl;
    cout << "  -> 平均 Roll  (横滚) 误差: " << mean_roll_err << " 度" << endl;
    cout << "-----------------------" << endl;

    return 0;
}
    
cv::Point3d rotationMatrixToEulerAngles(const cv::Mat &R)
{
    // 确保矩阵是 3x3 类型，并且是 double 精度
    assert(R.rows == 3 && R.cols == 3);
    
    // 用于检测万向锁 (Gimbal Lock) 的中间量
    // sy = sqrt(r00*r00 + r10*r10)
    double sy = std::sqrt(R.at<double>(0, 0) * R.at<double>(0, 0) + 
                          R.at<double>(1, 0) * R.at<double>(1, 0));

    // 判断是否接近奇异点 (sy close to 0)
    bool singular = sy < 1e-6; 

    double x, y, z;

    if (!singular)
    {
        // 正常情况
        // Roll (绕 X 轴) = atan2(r21, r22)
        x = std::atan2(R.at<double>(2, 1), R.at<double>(2, 2));
        
        // Pitch (绕 Y 轴) = atan2(-r20, sy)
        y = std::atan2(-R.at<double>(2, 0), sy);
        
        // Yaw (绕 Z 轴) = atan2(r10, r00)
        z = std::atan2(R.at<double>(1, 0), R.at<double>(0, 0));
    }
    else
    {
        // 万向锁情况 (Pitch = +/- 90度)
        // 此时 Yaw 和 Roll 轴重合，只能计算它们的差或和
        // 通常令 Roll = 0
        x = 0;
        y = std::atan2(-R.at<double>(2, 0), sy);
        z = std::atan2(-R.at<double>(1, 2), R.at<double>(1, 1));
    }

    // // 将弧度转换为角度
    // x = x * 180.0 / CV_PI;
    // y = y * 180.0 / CV_PI;
    // z = z * 180.0 / CV_PI;

    // 返回 Yaw(Z), Pitch(Y), Roll(X)
    return cv::Point3d(z, y, x);
}
