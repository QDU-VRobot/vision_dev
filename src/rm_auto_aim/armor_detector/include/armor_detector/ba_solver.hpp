
#ifndef ARMOR_DETECTOR_BA_SOLVER_HPP_
#define ARMOR_DETECTOR_BA_SOLVER_HPP_

// std
#include <array>
#include <cstddef>
#include <tuple>
#include <vector>

// 3rd party
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>

// ROS2
#include <rclcpp/logger.hpp>
#include <rclcpp/logging.hpp>
#include <std_msgs/msg/float32.hpp>
// rm_auto_aim
#include "armor_detector/armor.hpp"
#include "armor_detector/pnp_solver.hpp"

// g2o
#include <g2o/core/base_multi_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/optimization_algorithm.h>
#include <g2o/core/optimization_algorithm_factory.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_factory.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/types/slam3d/types_slam3d.h>
#include <g2o/types/slam3d/vertex_pointxyz.h>

namespace rm_auto_aim
{

constexpr double FIFTTEN_DEGREE_RAD = 15 * CV_PI / 180;

// Vertex of graph optimization algorithm for the yaw angle
class VertexYaw : public g2o::BaseVertex<1, double>
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  VertexYaw() = default;
  virtual void setToOriginImpl() override { _estimate = 0; }
  virtual void oplusImpl(const double *update) override;

  virtual bool read(std::istream &in) override { return true; }
  virtual bool write(std::ostream &out) const override { return true; }
};

// Edge of graph optimization algorithm for reporjection error calculation using
// yaw angle and observation
class EdgeProjection
    : public g2o::BaseBinaryEdge<2, Eigen::Vector2d, VertexYaw, g2o::VertexPointXYZ>
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  using InfoMatrixType = Eigen::Matrix<double, 2, 2>;

  EdgeProjection(const Sophus::SO3d &R_camera_imu, const Sophus::SO3d &R_pitch,
                 const Eigen::Vector3d &t, const Eigen::Matrix3d &K);
  virtual void computeError() override;

  virtual bool read(std::istream &in) override { return true; }
  virtual bool write(std::ostream &out) const override { return true; }

 private:
  Sophus::SO3d R_camera_imu_;
  Sophus::SO3d R_pitch_;
  Eigen::Vector3d t_;
  Eigen::Matrix3d K_;
};

class BaSolver
{
 public:
  BaSolver(std::array<double, 9> &camera_matrix, std::vector<double> &dist_coeffs);

  // Solve the armor pose using the BA algorithm, return the optimized rotation
  Eigen::Matrix3d solveBa(const Armor &armor, const Eigen::Vector3d &t_camera_armor,
                          const Eigen::Matrix3d &R_camera_armor,
                          const Eigen::Matrix3d &R_imu_camera) noexcept;

 private:
  Eigen::Matrix3d K_;
  g2o::SparseOptimizer optimizer_;
  g2o::OptimizationAlgorithmProperty solver_property_;
  g2o::OptimizationAlgorithmLevenberg *lm_algorithm_;
};
};  // namespace rm_auto_aim

#endif  // ARMOR_DETECTOR_BA_SOLVER_HPP_