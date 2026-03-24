#ifndef  INCLUDE_MEASURECOV_HPP
#define  INCLUDE_MEASURECOV_HPP

#include "eigen3/Eigen/Core"
#include <array>

class MeasureCov
{
private:
    std::array<double,4> mean = {0,0,0,0}; // 当前均值
    std::array<double,4> M2 = {0,0,0,0};   // 误差平方和 (Sum of squares of differences from the current mean)
    size_t count = 0;

public:
    std::array<double,4> operator()(const Eigen::Matrix<double, 4, 1>& view) 
    {
        this->update(view);
        return this->get();
    }

    void update(const Eigen::Matrix<double, 4, 1>& view)
    {
        this->count++;
        
        for (int i = 0; i < 4; ++i) {
            // Welford 算法核心更新步骤
            double value = view(i, 0);
            double delta = value - this->mean[i];
            
            // 更新均值
            this->mean[i] += delta / this->count;
            
            // 使用更新后的新均值计算 delta2
            double delta2 = value - this->mean[i];
            
            // 累加误差平方和
            this->M2[i] += delta * delta2;
        }
    }

    std::array<double,4> get() const
    {
        std::array<double, 4> variance = {0, 0, 0, 0};

        if (this->count <= 1) {
            return variance; // 防止除零
        }

        for (int i = 0; i < 4; ++i) {
            // 计算无偏样本方差
            variance[i] = this->M2[i] / (this->count - 1);
        }

        return variance;
    }

    void reset() 
    {
        this->mean = {0, 0, 0, 0};
        this->M2 = {0, 0, 0, 0};
        this->count = 0;
    }
};

#endif