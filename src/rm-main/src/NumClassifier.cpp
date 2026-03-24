#include "../include/NumClassifier.hpp"
#include <eigen3/Eigen/src/Core/Matrix.h>
#include <eigen3/Eigen/src/Core/util/Constants.h>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>

// ==========================================
// 【性能核心】固定最大 Batch Size
// 告诉 GPU 永远只申请 MAX_BATCH_SIZE 张图的显存，避免重新编译
// ==========================================
static const size_t MAX_BATCH_SIZE = 4;
static const size_t GPU_request_num = 2;
static const size_t CPU_request_num = 12;
static const size_t GPU_ENABLE_THRESHOLD = 8;

Eigen::Matrix<float, 7, 128> NumClassifier::centers;

NumClassifier::NumClassifier(std::string model_path,std::string yaml_path)
{
// --------------------------------------


    // 2. 使用 OpenCV FileStorage 读取 YAML
    cv::FileStorage fs(yaml_path, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "❌ 错误: 无法打开 standard.yaml，请检查路径或确认第一行有 %YAML:1.0" << std::endl;
        return ;
    }

    // 3. 获取 "centers" 节点
    cv::FileNode centers_node = fs["centers"];
    if (centers_node.type() != cv::FileNode::SEQ) {
        std::cerr << "❌ 错误: YAML 格式不对，'centers' 应该是一个列表(Sequence)。" << std::endl;
        return ;
    }

    // 4. 双层循环遍历解析二维列表，并存入 Eigen::Matrix
    int row_idx = 0;
    for (cv::FileNodeIterator row_it = centers_node.begin(); row_it != centers_node.end(); ++row_it, ++row_idx) {
        cv::FileNode row = *row_it;
        int col_idx = 0;
        for (cv::FileNodeIterator col_it = row.begin(); col_it != row.end(); ++col_it, ++col_idx) {
            // 读取浮点数并赋值给 Eigen 矩阵对应的 (行, 列)
            this->centers(row_idx, col_idx) = (float)(*col_it);
        }
    }
    fs.release();

    std::cout << "✅ 成功加载 ArcFace 中心向量，矩阵维度: " 
              << this->centers.rows() << " x " << this->centers.cols() << "\n";




// ---------------------------------------------------
    // 1. 读取模型
    std::shared_ptr<ov::Model> model_CPU = core.read_model(model_path);

    // 2. 配置预处理 (PrePostProcessor)
    // 这一步相当于把 blobFromImages 搬到了 GPU 内部执行
    ov::preprocess::PrePostProcessor ppp(model_CPU);
    
    // [输入定义] 告诉 OpenVINO 我们传进来的是什么：
    // 1. u8: 原始像素 0-255
    // 2. NHWC: OpenCV Mat 的默认内存布局 [Batch, Height, Width, Channel]
    // 3. BGR: OpenCV 默认颜色顺序
    ppp.input().tensor()
        .set_element_type(ov::element::u8)
        .set_layout("NHWC") 
        .set_color_format(ov::preprocess::ColorFormat::BGR);

    // [预处理步骤]
    // 这些步骤会自动在 GPU 上执行
    ppp.input().preprocess()
        .convert_color(ov::preprocess::ColorFormat::RGB) // BGR 转 RGB (匹配 PyTorch)
        .convert_element_type(ov::element::f32)          // 转 float
        .mean({123.675f, 116.280f, 103.530f})            // ImageNet Mean * 255
        // 【精度核心】这里必须使用 PyTorch ImageNet 的精确 Std * 255
        // R: 0.229*255=58.395, G: 0.224*255=57.12, B: 0.225*255=57.375
        .scale({58.395f, 57.120f, 57.375f});

    // 模型内部原本需要的输入布局 (通常 ONNX 导出是 NCHW)
    ppp.input().model().set_layout("NCHW");

    // 构建模型，将预处理步骤融入模型图层中
    model_CPU = ppp.build();

    std::map<std::string, ov::PartialShape> shapes_cpu;
    shapes_cpu[model_CPU->input().get_any_name()] = ov::PartialShape{1, 112, 112, 3}; 
    model_CPU->reshape(shapes_cpu);

    auto model_GPU = model_CPU->clone();
    std::map<std::string, ov::PartialShape> shapes;
    shapes[model_GPU->input().get_any_name()] = ov::PartialShape{MAX_BATCH_SIZE, 112, 112, 3}; 
    model_GPU->reshape(shapes);


        // 3. 编译模型
        try {
            std::cout << "[OpenVINO] 正在加载到 GPU (FP32 + NHWC Reshape)..." << std::endl;
            
            ov::AnyMap gpu_props;
            // 使用 FP32 保证 "3 vs 7" 的识别精度 (小模型上 FP32 往往比 FP16 更快)
            gpu_props[ov::hint::inference_precision.name()] = ov::element::f32; 
            // 开启低延迟模式
            gpu_props[ov::hint::performance_mode.name()] = ov::hint::PerformanceMode::LATENCY;
            // 开启模型缓存，第二次启动秒开
            gpu_props[ov::cache_dir.name()] = "./gpu_cache";

            this->compiled_model_GPU = core.compile_model(model_GPU, "GPU", gpu_props);
            this->has_gpu = true;
            std::cout << "[OpenVINO] 成功: 模型已加载到核显。" << std::endl;
        } 
        catch (const std::exception& e) {
            std::cerr << "[OpenVINO] 警告: 核显加载失败: " << e.what() << std::endl;
        }

        try {
            std::cout << "[OpenVINO] 正在加载到CPU ..." << std::endl;
            ov::AnyMap cpu_props;
            cpu_props[ov::hint::performance_mode.name()] = ov::hint::PerformanceMode::LATENCY;
            this->compiled_model_CPU = core.compile_model(model_CPU, "CPU", cpu_props);
            this->has_cpu = true;
            std::cout << "[OpenVINO] 成功: 模型已加载到 CPU。" << std::endl;
        } catch (const std::exception& ex) {
            std::cerr << "[OpenVINO] 致命错误: CPU 也无法加载模型！" << ex.what() << std::endl;
            exit(-1); 
        }

        // 4. 创建推理请求
        if (this->has_gpu) {
            std::cout << "[策略] 采用 GPU 进行推理" << std::endl;
            for(int i = 0 ;i < GPU_request_num; i++)
            this->infer_request_GPU.emplace_back(compiled_model_GPU.create_infer_request());
        }

        if (this->has_cpu) {
            std::cout << "[策略] 成功创建 " << CPU_request_num << " 个 CPU 辅助工位" << std::endl;
            for(int i = 0 ;i < CPU_request_num; i++) {
                this->infer_request_CPU.emplace_back(compiled_model_CPU.create_infer_request());
            }
        }
        else {
            std::cerr << "[致命错误] 没有任何可用设备！" << std::endl;
            exit(-1);
        }

    
    // 6. 热身 (Warm-up)
    // 跑一次空数据，消除 GPU 首帧编译带来的延迟

        try {
            if (this->has_gpu && !infer_request_GPU.empty()) {
                ov::Tensor input_tensor = infer_request_GPU[0].get_input_tensor();
                std::memset(input_tensor.data(), 0, input_tensor.get_byte_size());
                infer_request_GPU[0].infer();
            }
            if (this->has_cpu && !infer_request_CPU.empty()) {
                ov::Tensor input_tensor = infer_request_CPU[0].get_input_tensor();
                std::memset(input_tensor.data(), 0, input_tensor.get_byte_size());
                infer_request_CPU[0].infer();
            }
        } catch (...) {}
}

/**
 * @brief 将一组图像识别出数字
 * @param armors_pattern 一组图像 (每个图像的大小为 112x112x3)
 * @return std::vector<NumClassifier::Ans> 识别结果 (每个结果是一个 std::pair<int, float>，表示识别结果的数字和置信度)
 */
std::vector<NumClassifier::Ans> NumClassifier::Classify(const std::vector<cv::Mat>& armors_pattern)
{
    
    // cv::imshow("test",armors_pattern[0]);
    // cv::waitKey(1);
    std::vector<NumClassifier::Ans> ans;
    size_t N = armors_pattern.size();
    if (N == 0) return ans;
    ans.reserve(N);

    Eigen::MatrixXf features(128, N); 

    struct Task {
        ov::InferRequest* req;
    };
    std::vector<Task> tasks;

    size_t img_idx = 0;
    size_t gpu_idx = 0;
    size_t cpu_idx = 0;
    size_t num = N; 

    // ==========================================
    // 阶段 1：分发任务 (Fork) - 零拷贝绑死
    // ==========================================
    while (num > 0) {
        ov::InferRequest* req = nullptr;
        bool use_gpu = false;
        size_t count = 0;

        // 调度逻辑
        if (num >= GPU_ENABLE_THRESHOLD) {
            if (gpu_idx < infer_request_GPU.size()) {
                req = &infer_request_GPU[gpu_idx++];
                use_gpu = true;
                count = std::min(MAX_BATCH_SIZE, num);
            } else if (cpu_idx < infer_request_CPU.size()) {
                req = &infer_request_CPU[cpu_idx++];
                count = 1; // CPU 模式：一次只吃 1 张图
            }
        } else {
            if (cpu_idx < infer_request_CPU.size()) {
                req = &infer_request_CPU[cpu_idx++];
                count = 1; 
            } else if (gpu_idx < infer_request_GPU.size()) {
                req = &infer_request_GPU[gpu_idx++];
                use_gpu = true;
                count = std::min(MAX_BATCH_SIZE, num);
            }
        }

        if (!req) {
            std::cerr << "[警告] 装甲板数量过多，超出全部调度池容量！" << std::endl;
            break; 
        }

        // ------------------------------------------
        // 输入 & 输出内存绑定
        // ------------------------------------------
        if (use_gpu) {
            // [GPU 模式] 依然需要 memcpy 组装成 Batch
            uint8_t* input_ptr = req->get_input_tensor().data<uint8_t>();
            size_t img_bytes = 112 * 112 * 3;

            for (size_t j = 0; j < MAX_BATCH_SIZE; ++j) {
                if (j < count) {
                    const cv::Mat& src = armors_pattern[img_idx + j];

                    std::memcpy(input_ptr + j * img_bytes, src.data, img_bytes);
                }
            }

            // GPU 输出零拷贝：让 OpenVINO 直接写进 features 的第 img_idx 列地址中
            ov::Tensor output_tensor(ov::element::f32, {MAX_BATCH_SIZE, 128}, &features(0, img_idx));
            req->set_output_tensor(output_tensor);

        } else {
            // [CPU 模式] Batch=1，输入输出双向绝对零拷贝！
            const cv::Mat& src = armors_pattern[img_idx];
            uint8_t* data_ptr = const_cast<uint8_t*>(src.data);// 核心：最爽的直接指针映射！
            

            // CPU 输入零拷贝绑定
            ov::Tensor input_tensor(ov::element::u8, {1, 112, 112, 3}, data_ptr);
            req->set_input_tensor(input_tensor);

            // CPU 输出零拷贝绑定：直接写进 features 的当前列
            ov::Tensor output_tensor(ov::element::f32, {1, 128}, &features(0, img_idx));
            req->set_output_tensor(output_tensor);
        }

        // 异步发射
        req->start_async();
        tasks.push_back({req});
        
        num -= count;
        img_idx += count;
    }

    // ==========================================
    // 阶段 2：收集并执行矩阵计算 (Join)
    // ==========================================
    for (const auto& task : tasks) {
        task.req->wait(); 
    }

    // 神奇的事情发生了：
    // 当所有 wait() 结束时，所有图像的特征已经被 OpenVINO 乖乖地写在了 features 矩阵里！
    // 我们甚至连提取 get_output_tensor 的代码都不用写了。

    // 一把梭哈：ArcFace 全局矩阵乘法
    // centers(7x128) * valid_features(128xN) = similarities(7xN)
    Eigen::MatrixXf similarities = NumClassifier::centers * features;

    // 解析结果
    for (size_t i = 0; i < N; ++i) {
        float max_score = -2.0f;
        int max_id = 0;
        
        // 查找第 i 列（第 i 张图）的最高分
        for (int c = 0; c < 7; ++c) {
            if (similarities(c, i) > max_score) {
                max_score = similarities(c, i);
                max_id = c;
            }
        }
        ans.emplace_back(max_id, max_score);
    }

    return ans;
}



std::vector<ArmorPosi> NumClassifier::operator()(std::vector< std::array<ArmorPosi,2> >& armors,const std::vector<cv::Mat>& armors_pattern)
{
    std::vector<ArmorPosi> result;
    if(armors.empty()) return result;

    result.reserve(armors.size());

    std::vector<Ans> ans = Classify(armors_pattern);
    for(int i = 0;i < ans.size();i++)
    {
        if(ans[i].confidence > 0.70)
        {
            if(ans[i].id == 0 || ans[i].id == 1)
            {
                result.emplace_back(armors[i][1]);
                result.back().type = static_cast<ArmorPosi::Type>(ans[i].id);
                result.back().confidence = ans[i].confidence;
            } 
                
            else 
            {
                result.emplace_back(armors[i][0]);
                result.back().type = static_cast<ArmorPosi::Type>(ans[i].id);
                result.back().confidence = ans[i].confidence;
            } 
                
        }
    }
    
    return result;
}