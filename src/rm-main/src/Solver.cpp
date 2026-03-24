#include "../include/Solver.hpp"
// #include <array>
#include <deque>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <vector>
#include <numeric>

// #define SolverDebug
#ifdef SolverDebug
#include "../include/RerunVisualizer.hpp"
extern RerunVisualizer viz;
#endif

Solver::Solver(std::string config_path)
{
    // this->cameraMatrix(3,3);
    // this->distCoeffs(5,1);
    cv::FileStorage fs;
    if (!fs.open(config_path, cv::FileStorage::READ)) 
        std::cerr << "Error: Failed to open YAML file: " << config_path << std::endl;
    
    std::cout << "Successfully opened " << config_path << std::endl;
    std::cout << "------------------------------------------" << std::endl;

    cameraMatrix = cv::Mat_<double>(3,3);
    distCoeffs = cv::Mat_<double>(5,1);

    fs["camera_matrix"] >> this->cameraMatrix;
    fs["distortion_coeffs"] >> this->distCoeffs;
    // std::cout<<cameraMatrix.size<<std::endl;

    this->R_Cam_to_gripper = cv::Mat_<double>(3,3);
    this->T_Cam_to_gripper = cv::Mat_<double>(3,1);

    // 相机到云台的旋转矩阵 (Rotation Matrix from Camera to Gripper)
    fs["R_Cam_to_gripper"] >> this->R_Cam_to_gripper;

    // 相机到云台的平移矩阵 (Translation Matrix from Camera to Gripper)
    fs["T_Cam_to_gripper"] >> this->T_Cam_to_gripper;


    this->BigArmorCenter = cv::Mat_<double>(3, 1);
    this->SmallArmorCenter = cv::Mat_<double>(3, 1);


    this->BigArmorCenter<< 11.50, 2.75, 0.0;
    this->SmallArmorCenter<< 6.75, 2.75, 0.0;
}

//解算单个装甲板的位置
std::array<ArmorPosi,2> Solver::operator () (const Armor& armor)
{
    //ArmorPosi(posi, face, toward, std::atan2(toward.z,toward.x), error);
    cv::Point3d posi0, face0, toward0, posi1, face1, toward1;
    double error0,error1;

    std::vector<cv::Mat> rvecs,tvecs;
    std::vector<double> reprojectionError;
    
    //当做小装甲板解算
    {
        int solutions = cv::solvePnPGeneric(
            this->objectSmallArmorP,
            armor.Lightcorners,
            cameraMatrix,
            distCoeffs,
            rvecs,
            tvecs,
            false,
            cv::SOLVEPNP_IPPE, // 使用 IPPE 算法获取多个解
            cv::noArray(),
            cv::noArray(),
            reprojectionError
        );

        // if(reprojectionError[0]>10||reprojectionError[1]) continue;
        // std::cerr<<reprojectionError.front()<<" "<<reprojectionError.back()<<std::endl;
        //筛选歧义解
        double Z_data[3]{0,0,10};
        cv::Mat Z_vector(cv::Size(1,3),CV_64FC1,Z_data);

        cv::Mat r_0,r_1;
        cv::Rodrigues(rvecs.front(), r_0);
        cv::Rodrigues(rvecs.back(), r_1);

        cv::Mat Z_camera_0 = r_0 * Z_vector;
        cv::Mat Z_camera_1 = r_1 * Z_vector;
        cv::Mat R,T;
        #ifdef SolverDebug
        viz.show("SmallSolvePnP0", Z_camera_0.at<double>(2,0));
        viz.show("SmallSolvePnP1", Z_camera_1.at<double>(2,0));
        #endif
        if(Z_camera_0.at<double>(2,0) > 0) {R = r_0; T = tvecs.front(); error0 = reprojectionError.front();}
        else {R = r_1; T = tvecs.back(); error0 = reprojectionError.back();}

        cv::Mat P = R * this->SmallArmorCenter + T;
        posi0 = cv::Point3d(P.at<double>(0,0),P.at<double>(1,0),P.at<double>(2,0));

        //计算朝向向量
        P = R * (cv::Mat_<double>(3,1) << 0.0, 0.0, 1.0);
        face0 = cv::Point3d(P.at<double>(0,0),P.at<double>(1,0),P.at<double>(2,0));

        P = R * (cv::Mat_<double>(3,1) << 1.0, 0.0, 0.0);
        toward0 = cv::Point3d(P.at<double>(0,0),P.at<double>(1,0),P.at<double>(2,0));
    }

    //当做大装甲板计算
    {
        int solutions = cv::solvePnPGeneric(
            this->objectBigArmorP,
            armor.Lightcorners,
            cameraMatrix,
            distCoeffs,
            rvecs,
            tvecs,
            false,
            cv::SOLVEPNP_IPPE, // 使用 IPPE 算法获取多个解
            cv::noArray(),
            cv::noArray(),
            reprojectionError
        );

        // if(reprojectionError[0]>10||reprojectionError[1]) continue;
        // std::cerr<<reprojectionError.front()<<" "<<reprojectionError.back()<<std::endl;
        //筛选歧义解
        double Z_data[3]{0,0,10};
        cv::Mat Z_vector(cv::Size(1,3),CV_64FC1,Z_data);

        cv::Mat r_0,r_1;
        cv::Rodrigues(rvecs.front(), r_0);
        cv::Rodrigues(rvecs.back(), r_1);

        cv::Mat Z_camera_0 = r_0 * Z_vector;
        cv::Mat Z_camera_1 = r_1 * Z_vector;

        cv::Mat R,T;
        // std::cerr<<Z_camera_0.at<double>(2,0)<<" "<<Z_camera_1.at<double>(2,0)<<std::endl;
        if(Z_camera_0.at<double>(2,0) > 0) {R = r_0; T = tvecs.front(); error1 = reprojectionError.front();}
        else {R = r_1; T = tvecs.back(); error1 = reprojectionError.back();}

        cv::Mat P = R * this->BigArmorCenter + T;
        posi1 = cv::Point3d(P.at<double>(0,0),P.at<double>(1,0),P.at<double>(2,0));

        //计算朝向向量
        P = R * (cv::Mat_<double>(3,1) << 0.0, 0.0, 1.0);
        face1 = cv::Point3d(P.at<double>(0,0),P.at<double>(1,0),P.at<double>(2,0));

        P = R * (cv::Mat_<double>(3,1) << 1.0, 0.0, 0.0);
        toward1 = cv::Point3d(P.at<double>(0,0),P.at<double>(1,0),P.at<double>(2,0));
    }
    
    return std::array< ArmorPosi, 2>{ArmorPosi{posi0, face0, toward0, std::atan2(toward0.z,toward0.x), error0},
                                     ArmorPosi{posi1, face1, toward1, std::atan2(toward1.z,toward1.x), error1}};
    
}

std::vector<ArmorPosi> Solver::operator()(const std::vector<YoloArmor>& armors)
{
    std::vector<ArmorPosi> results;
    if (armors.empty()) return results;
    results.reserve(armors.size());

    for (const auto& yolo_armor : armors) {
        // 1. 根据 class_id 确定类型和 3D 模型
        ArmorPosi::Type target_type;
        bool is_big = false;
        int id = yolo_armor.class_id;

        // 映射逻辑: 只有 base 和 hero 是大装甲板
        switch (id) {
            // 哨兵 (0-2)
            case 0: case 1: case 2:
                target_type = ArmorPosi::Type::guard;
                is_big = false;
                break;

            // 英雄 (3-5) - 大装甲板
            case 3: case 4: case 5:
                target_type = ArmorPosi::Type::hero;
                is_big = true;
                break;

            // 2号步兵 (6-8)
            case 6: case 7: case 8:
                target_type = ArmorPosi::Type::two;
                is_big = false;
                break;

            // 3号步兵 (9-11)
            case 9: case 10: case 11:
                target_type = ArmorPosi::Type::three;
                is_big = false;
                break;

            // 4号步兵 (12-14)
            case 12: case 13: case 14:
                target_type = ArmorPosi::Type::four;
                is_big = false;
                break;

            // 前哨站 (18-20)
            case 18: case 19: case 20:
                target_type = ArmorPosi::Type::outpost;
                is_big = false;
                break;

            // 大基地 (21-24) - 大装甲板
            case 21: case 22: case 23: case 24:
                target_type = ArmorPosi::Type::base;
                is_big = true;
                break;

            // 小基地 (25-28)
            case 25: case 26: case 27: case 28:
                target_type = ArmorPosi::Type::base;
                is_big = false;
                break;

            // 未知类型，跳过
            default:
                continue;
        }

        // 选择对应的 3D 物体坐标系参考点
        const auto& objectPoints = is_big ? this->objectBigArmorP : this->objectSmallArmorP;
        const auto& centerPoint = is_big ? this->BigArmorCenter : this->SmallArmorCenter;

        // 2. 直接进行 PnP 解算 (参考第一个函数的解算逻辑)
        std::vector<cv::Mat> rvecs, tvecs;
        std::vector<double> reprojectionError;

        cv::solvePnPGeneric(
            objectPoints,
            yolo_armor.keypoints,
            this->cameraMatrix,
            this->distCoeffs,
            rvecs,
            tvecs,
            false,
            cv::SOLVEPNP_IPPE,
            cv::noArray(),
            cv::noArray(),
            reprojectionError
        );

        if (rvecs.empty()) continue;

        // 3. 筛选歧义解 (取 Z > 0 且在相机前方的解)
        cv::Mat R, T;
        double final_error;

        // 与参考代码保持一致的歧义解筛选逻辑
        double Z_data[3]{0, 0, 10};
        cv::Mat Z_vector(cv::Size(1, 3), CV_64FC1, Z_data);

        cv::Mat r_0, r_1;
        cv::Rodrigues(rvecs.front(), r_0);
        cv::Mat Z_camera_0 = r_0 * Z_vector;

        // 如果有两个解，同时检查两个解
        if (rvecs.size() > 1) {
            cv::Rodrigues(rvecs.back(), r_1);
            cv::Mat Z_camera_1 = r_1 * Z_vector;

            // 选择 Z > 0 的解
            if (Z_camera_0.at<double>(2, 0) > 0) {
                R = r_0;
                T = tvecs.front();
                final_error = reprojectionError.front();
            } else {
                R = r_1;
                T = tvecs.back();
                final_error = reprojectionError.back();
            }
        } else {
            // 只有一个解，直接使用
            R = r_0;
            T = tvecs.front();
            final_error = reprojectionError.front();
        }

        // 4. 计算结果并填充 ArmorPosi
        cv::Mat P_posi = R * centerPoint + T;
        cv::Point3d posi(P_posi.at<double>(0, 0), P_posi.at<double>(1, 0), P_posi.at<double>(2, 0));

        cv::Mat P_face = R * (cv::Mat_<double>(3, 1) << 0.0, 0.0, 1.0);
        cv::Point3d face(P_face.at<double>(0, 0), P_face.at<double>(1, 0), P_face.at<double>(2, 0));

        cv::Mat P_toward = R * (cv::Mat_<double>(3, 1) << 1.0, 0.0, 0.0);
        cv::Point3d toward(P_toward.at<double>(0, 0), P_toward.at<double>(1, 0), P_toward.at<double>(2, 0));

        // 5. 构造并存入结果
        ArmorPosi result(posi, face, toward, std::atan2(toward.z, toward.x), final_error);
        result.type = target_type;
        result.confidence = yolo_armor.conf;

        results.emplace_back(result);
    }

    return results;
}

std::vector< std::array< ArmorPosi, 2> > Solver::operator()(const std::deque<Armor>& armors)
{
    std::vector< std::array< ArmorPosi, 2> > armors_posi;
    if(armors.empty()) return armors_posi;
    armors_posi.reserve(armors.size());

    for(const auto& armor:armors)
    {
        armors_posi.emplace_back(this->operator()(armor));//记录
    }
    return armors_posi;
}

std::vector< std::array< ArmorPosi, 2> > Solver::operator()(const std::vector<Armor>& armors)
{
    std::vector< std::array< ArmorPosi, 2> > armors_posi;
    if(armors.empty()) return armors_posi;
    armors_posi.reserve(armors.size());

    for(const auto& armor:armors)
    {
        armors_posi.emplace_back(this->operator()(armor));//记录
    }
    return armors_posi;
}




void Solver::ConverToWorld(std::array<ArmorPosi,2>& armor_posis, const cv::Quatd& gripper_to_world)
{
    for(auto& armor_posi:armor_posis)
    {
        cv::Mat R(gripper_to_world.toRotMat3x3());// 手坐标系到世界坐标系的旋转矩阵
        
        // 将装甲板位置从相机坐标系转换到手坐标系
        cv::Mat posi = this->R_Cam_to_gripper * cv::Mat(3,1,CV_64F, &armor_posi.posi) + this->T_Cam_to_gripper;
        cv::Mat face = this->R_Cam_to_gripper * cv::Mat(3,1,CV_64F, &armor_posi.face);
        cv::Mat toward = this->R_Cam_to_gripper * cv::Mat(3,1,CV_64F, &armor_posi.toward);
        
        // 将装甲板位置从手坐标系转换到世界坐标系
        posi = R * posi;
        face = R * face;
        toward = R * toward;

        // 更新装甲板位置
        armor_posi.posi = cv::Point3d(posi.at<double>(0, 0), posi.at<double>(1, 0), posi.at<double>(2, 0));
        armor_posi.face = cv::Point3d(face.at<double>(0, 0), face.at<double>(1, 0), face.at<double>(2, 0));
        armor_posi.toward = cv::Point3d(toward.at<double>(0, 0), toward.at<double>(1, 0), toward.at<double>(2, 0));
    }
}

void Solver::ConverToWorld(std::vector< std::array<ArmorPosi,2> >& armors_posis, const cv::Quatd& gripper_to_world)
{
    cv::Mat R (gripper_to_world.toRotMat3x3());// 手坐标系到世界坐标系的旋转矩阵

    for(auto& armor_posis:armors_posis)
    {
        for(auto& armor_posi:armor_posis)
        {
            // 将装甲板位置从相机坐标系转换到手坐标系
            cv::Mat posi = this->R_Cam_to_gripper * cv::Mat(3,1,CV_64F, &armor_posi.posi) + this->T_Cam_to_gripper;
            cv::Mat face = this->R_Cam_to_gripper * cv::Mat(3,1,CV_64F, &armor_posi.face);
            cv::Mat toward = this->R_Cam_to_gripper * cv::Mat(3,1,CV_64F, &armor_posi.toward);
            
            // 将装甲板位置从手坐标系转换到世界坐标系
            posi = R * posi;
            face = R * face;
            toward = R * toward;

            // 更新装甲板位置
            armor_posi.posi = cv::Point3d(posi.at<double>(0, 0), posi.at<double>(1, 0), posi.at<double>(2, 0));
            armor_posi.face = cv::Point3d(face.at<double>(0, 0), face.at<double>(1, 0), face.at<double>(2, 0));
            armor_posi.toward = cv::Point3d(toward.at<double>(0, 0), toward.at<double>(1, 0), toward.at<double>(2, 0));
        }
    }
}


void Solver::ConverToWorld(ArmorPosi& armor_posi, const cv::Quatd& gripper_to_world)
{
    cv::Mat R(gripper_to_world.toRotMat3x3());// 手坐标系到世界坐标系的旋转矩阵
    
    // 将装甲板位置从相机坐标系转换到手坐标系
    cv::Mat posi = this->R_Cam_to_gripper * cv::Mat(3,1,CV_64F, &armor_posi.posi) + this->T_Cam_to_gripper;
    cv::Mat face = this->R_Cam_to_gripper * cv::Mat(3,1,CV_64F, &armor_posi.face);
    cv::Mat toward = this->R_Cam_to_gripper * cv::Mat(3,1,CV_64F, &armor_posi.toward);
    
    // 将装甲板位置从手坐标系转换到世界坐标系
    posi = R * posi;
    face = R * face;
    toward = R * toward;

    // 更新装甲板位置
    armor_posi.posi = cv::Point3d(posi.at<double>(0, 0), posi.at<double>(1, 0), posi.at<double>(2, 0));
    armor_posi.face = cv::Point3d(face.at<double>(0, 0), face.at<double>(1, 0), face.at<double>(2, 0));
    armor_posi.toward = cv::Point3d(toward.at<double>(0, 0), toward.at<double>(1, 0), toward.at<double>(2, 0));
}

void Solver::ConverToWorld(std::vector<ArmorPosi>& armors_posi, const cv::Quatd& gripper_to_world)
{
    cv::Mat R (gripper_to_world.toRotMat3x3());// 手坐标系到世界坐标系的旋转矩阵

    for(auto& armor_posi:armors_posi)
    {
        // 将装甲板位置从相机坐标系转换到手坐标系
        cv::Mat posi = this->R_Cam_to_gripper * cv::Mat(3,1,CV_64F, &armor_posi.posi) + this->T_Cam_to_gripper;
        cv::Mat face = this->R_Cam_to_gripper * cv::Mat(3,1,CV_64F, &armor_posi.face);
        cv::Mat toward = this->R_Cam_to_gripper * cv::Mat(3,1,CV_64F, &armor_posi.toward);
        
        // 将装甲板位置从手坐标系转换到世界坐标系
        posi = R * posi;
        face = R * face;
        toward = R * toward;

        // 更新装甲板位置
        armor_posi.posi = cv::Point3d(posi.at<double>(0, 0), posi.at<double>(1, 0), posi.at<double>(2, 0));
        armor_posi.face = cv::Point3d(face.at<double>(0, 0), face.at<double>(1, 0), face.at<double>(2, 0));
        armor_posi.toward = cv::Point3d(toward.at<double>(0, 0), toward.at<double>(1, 0), toward.at<double>(2, 0));
    }
}


void Solver::ansShow(const cv::Point3d& posi,cv::Mat& image)
{
    cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64F); // 单位旋转向量
    cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64F); // 单位平移向量

    // 3. 执行投影
    // cv::projectPoints 需要一个点的向量作为输入
    std::vector<cv::Point3d> objectPoints;
    objectPoints.push_back(posi);

    // 用于存储投影结果的2D点向量
    std::vector<cv::Point2d> imagePoints;

    //重投影
    cv::projectPoints(objectPoints, rvec, tvec, cameraMatrix, distCoeffs, imagePoints);

    //在图像上绘制结果
    // 输出和可视化结果
    // 投影后的2D点坐标
    cv::Point2d projectedPoint = imagePoints[0];
    int imageWidth = image.cols;
    int imageHeight = image.rows;

    // 在图像上绘制投影点 (画一个红色的圆圈)
    // 检查点是否在图像范围内
    if (projectedPoint.x >= 0 && projectedPoint.x < imageWidth &&
        projectedPoint.y >= 0 && projectedPoint.y < imageHeight)
    {
        cv::circle(image, projectedPoint, 5, cv::Scalar(0, 0, 255), -1); // 红色实心圆
        cv::putText(image, "Projected Point", cv::Point(projectedPoint.x + 10, projectedPoint.y),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    } else {
        std::cout << "Projected point is outside the image frame." << std::endl;
    }
    // 显示图像
    // cv::imshow("Projected Point Visualization", image);
    // cv::waitKey(1); // 等待按键后退出
}

void Solver::Filter(std::vector< std::array<ArmorPosi,2> >& armors_posis,
                    std::vector<cv::Mat>& armors_pattern,
                    const cv::Quatd& gripper_to_world,
                    const Eigen::Matrix<double, 3, 1>& Gun,
                    const size_t num)
{
    std::vector< std::array<ArmorPosi,2> > armors_posis_result;
    std::vector<cv::Mat> armors_pattern_result;

    std::vector< ArmorPosi > armors_posi_small_in_world;

    armors_posis_result.reserve(armors_posis.size());
    armors_pattern_result.reserve(armors_pattern.size());
    armors_posi_small_in_world.reserve(armors_posis.size());

    for(int i = 0;i < armors_posis.size();i++)
    {
        std::array<ArmorPosi,2>& armor_posis = armors_posis[i];
        cv::Mat& pattern = armors_pattern[i];

        //解算误差筛选
        //std::cerr<<armor_posis[0].error<<" "<<armor_posis[1].error<<std::endl;
        if(armor_posis[0].error > 1 && armor_posis[1].error > 1) continue;

        //相机系下的距离筛选
        if(cv::norm(armor_posis[0].posi) > 800 && cv::norm(armor_posis[1].posi) > 800) continue;

        //坐标系变换
        auto small_armor_posis = armor_posis[0];
        auto big_armor_posis = armor_posis[1];
        this->ConverToWorld(small_armor_posis, gripper_to_world);
        this->ConverToWorld(big_armor_posis, gripper_to_world);
        // std::cout<<"small_armor_posis: "<<small_armor_posis.posi.z<<"\n";
        //高度筛选
        if( small_armor_posis.posi.z > 2000 && big_armor_posis.posi.z > 2000 ) continue;
        if( small_armor_posis.posi.z < -50 && big_armor_posis.posi.z < -50 ) continue;

        //角度筛选
        const auto& face_small = small_armor_posis.toward.cross(small_armor_posis.face);
        const auto& face_big = big_armor_posis.toward.cross(big_armor_posis.face);
        
        cv::Point3d base_small{small_armor_posis.posi.x, small_armor_posis.posi.y, 0};
        cv::Point3d base_big{big_armor_posis.posi.x, big_armor_posis.posi.y, 0};

        base_small = base_small / cv::norm(base_small);
        base_big = base_big / cv::norm(base_big);

        double angle_small = base_small.dot(face_small);
        double angle_big = base_big.dot(face_big);

        if ( ( angle_small < -0.5 || angle_small > 0.85 ) && (angle_big < -0.5 || angle_big > 0.85) ) continue;

        //储存筛选结果
        armors_posis_result.emplace_back(armor_posis);
        armors_pattern_result.emplace_back(pattern);
        armors_posi_small_in_world.emplace_back(small_armor_posis);
    }

    //选择与枪管夹角最小的num个装甲板
    if (armors_posis_result.size() > static_cast<size_t>(num))
    {
        // 1. 初始化索引数组 [0, 1, 2, ..., n-1]
        std::vector<size_t> indices(armors_posis_result.size());
        std::iota(indices.begin(), indices.end(), 0);

        // 2. 枪管方向归一化，以便后续通过点乘直接获取余弦值
        Eigen::Vector3d gun_vec = Gun.normalized();

        // 3. 按夹角部分排序（找出夹角最小的 num 个）
        std::partial_sort(indices.begin(), indices.begin() + num, indices.end(),
            [&](size_t i1, size_t i2) {
                // 取小装甲板的坐标作为代表进行比较
                const auto& p1 = armors_posi_small_in_world[i1].posi;
                const auto& p2 = armors_posi_small_in_world[i2].posi;

                // 组装为 Eigen 向量并归一化
                Eigen::Vector3d v1(p1.x, p1.y, p1.z);
                Eigen::Vector3d v2(p2.x, p2.y, p2.z);
                v1.normalize();
                v2.normalize();

                // 比较余弦值（点乘结果）。cos值越大，说明夹角越小
                return v1.dot(gun_vec) > v2.dot(gun_vec); 
            });

        // 4. 根据排序好的索引提取结果
        ArmorPosi dummy_armor(cv::Point3d(0,0,0), cv::Point3d(0,0,0), cv::Point3d(0,0,0), 0.0, 0.0);

        // 创建一个包含两个 dummy_armor 的默认 array
        std::array<ArmorPosi, 2> default_array = {dummy_armor, dummy_armor};
        armors_posis.resize(num,default_array);
        armors_pattern.resize(num);
        for (int i = 0; i < num; ++i) {
            armors_posis[i] = armors_posis_result[indices[i]];
            armors_pattern[i] = armors_pattern_result[indices[i]];
        }

        return;
    }
    //更新装甲板位置和图案
    armors_posis = std::move(armors_posis_result);
    armors_pattern = std::move(armors_pattern_result);
}

void Solver::ansShow(const ArmorPosi& armor,cv::Mat& image)
{
    double high = 27.5, width;
    if(armor.type == ArmorPosi::Type::hero || armor.type == ArmorPosi::Type::base)
        width = 115.0;
    else
        width = 67.5;

    cv::Point3d toward_w = width * armor.toward;
    cv::Point3d toward_h = high * (armor.face.cross(armor.toward)/cv::norm(armor.face.cross(armor.toward)));

    cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64F); // 单位旋转向量
    cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64F); // 单位平移向量

    // 执行投影
    // cv::projectPoints 需要一个点的向量作为输入
    std::vector<cv::Point3d> objectPoints;
    objectPoints.reserve(5);
    objectPoints.push_back(armor.posi);
    objectPoints.push_back(armor.posi - toward_w - toward_h);
    objectPoints.push_back(armor.posi + toward_w - toward_h);
    objectPoints.push_back(armor.posi + toward_w + toward_h);
    objectPoints.push_back(armor.posi - toward_w + toward_h);


    // 用于存储投影结果的2D点向量
    std::vector<cv::Point2d> imagePoints;
    imagePoints.reserve(5);

    //重投影
    cv::projectPoints(objectPoints, rvec, tvec, cameraMatrix, distCoeffs, imagePoints);

    //在图像上绘制结果
    cv::Point2d CenterPoint = imagePoints[0];
    int imageWidth = image.cols;
    int imageHeight = image.rows;

    // 在图像上绘制投影点 (画一个红色的圆圈)
    // 检查点是否在图像范围内
    if (CenterPoint.x >= 0 && CenterPoint.x < imageWidth &&
        CenterPoint.y >= 0 && CenterPoint.y < imageHeight)
    {
        cv::circle(image, CenterPoint, 5, cv::Scalar(0, 0, 255), -1); // 红色实心圆
        cv::putText(image, "Projected Point", cv::Point(CenterPoint.x + 10, CenterPoint.y),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    } else {
        std::cout << "Projected point is outside the image frame." << std::endl;
    }

    //绘制装甲板轮廓
    std::vector<cv::Point2d> points;
    points.reserve(4);
    for(int i=1;i<=4;i++)
    {
        cv::Point2d Point = imagePoints[i];
        if (Point.x >= 0 && Point.x < imageWidth &&
        Point.y >= 0 && Point.y < imageHeight)
        {
            points.push_back(Point);
        } else {
            std::cout << "Projected point is outside the image frame." << std::endl;
            return;
        }
    }

    // 绘制
    std::vector<std::vector<cv::Point2d>> contours{points};
    cv::polylines(image,contours,1,cv::Scalar(0, 255, 0),3,cv::LINE_AA);
}

void Solver::FilterAndConverToWorld(std::vector<ArmorPosi>& armors_posi,
                    const cv::Quatd& gripper_to_world,
                    const Eigen::Matrix<double, 3, 1>& Gun,
                    const size_t num)
{
    std::vector<ArmorPosi> armors_posi_result;
    armors_posi_result.reserve(armors_posi.size());

    for (auto& armor_posi : armors_posi)
    {
        // 解算误差筛选
        if (armor_posi.error > 1) continue;

        // 相机系下的距离筛选
        if (cv::norm(armor_posi.posi) > 800) continue;

        // 坐标系变换到世界坐标系
        this->ConverToWorld(armor_posi, gripper_to_world);

        // 高度筛选
        if (armor_posi.posi.z > 2000 || armor_posi.posi.z < -50) continue;

        // 角度筛选
        const auto& face = armor_posi.toward.cross(armor_posi.face);
        cv::Point3d base{armor_posi.posi.x, armor_posi.posi.y, 0};
        base = base / cv::norm(base);
        double angle = base.dot(face);

        if (angle < -0.5 || angle > 0.85) continue;

        // 储存筛选结果
        armors_posi_result.emplace_back(armor_posi);
    }

    // 选择与枪管夹角最小的num个装甲板
    if (armors_posi_result.size() > num)
    {
        // 初始化索引数组
        std::vector<size_t> indices(armors_posi_result.size());
        std::iota(indices.begin(), indices.end(), 0);

        // 枪管方向归一化
        Eigen::Vector3d gun_vec = Gun.normalized();

        // 按夹角部分排序
        std::partial_sort(indices.begin(), indices.begin() + num, indices.end(),
            [&](size_t i1, size_t i2) {
                const auto& p1 = armors_posi_result[i1].posi;
                const auto& p2 = armors_posi_result[i2].posi;

                Eigen::Vector3d v1(p1.x, p1.y, p1.z);
                Eigen::Vector3d v2(p2.x, p2.y, p2.z);
                v1.normalize();
                v2.normalize();

                return v1.dot(gun_vec) > v2.dot(gun_vec);
            });

        // 根据排序好的索引提取结果
        // 4. 根据排序好的索引提取结果
        ArmorPosi dummy_armor(cv::Point3d(0,0,0), cv::Point3d(0,0,0), cv::Point3d(0,0,0), 0.0, 0.0);

        armors_posi.resize(num, dummy_armor);
        for (size_t i = 0; i < num; ++i) {
            armors_posi[i] = armors_posi_result[indices[i]];
        }
        return;
    }

    // 更新装甲板位置
    armors_posi = std::move(armors_posi_result);
}

