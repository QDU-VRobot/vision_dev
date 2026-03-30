#include "armor_detector/ba_solver.hpp"

namespace rm_auto_aim
{

void VertexYaw::oplusImpl(const double *update)
{
  Sophus::SO3d R_yaw = Sophus::SO3d::exp(Eigen::Vector3d(0, 0, update[0])) *
                       Sophus::SO3d::exp(Eigen::Vector3d(0, 0, _estimate));
  _estimate = R_yaw.log()(2);
}

EdgeProjection::EdgeProjection(const Sophus::SO3d &R_camera_imu,
                               const Sophus::SO3d &R_pitch, const Eigen::Vector3d &t,
                               const Eigen::Matrix3d &K)
    : R_camera_imu_(R_camera_imu), R_pitch_(R_pitch), t_(t), K_(K)
{
}

void EdgeProjection::computeError()
{
  // Get the rotation
  double yaw = static_cast<VertexYaw *>(_vertices[0])->estimate();
  Sophus::SO3d R_yaw = Sophus::SO3d::exp(Eigen::Vector3d(0, 0, yaw));
  Sophus::SO3d R = R_camera_imu_ * R_yaw * R_pitch_;

  // Get the 3D point
  Eigen::Vector3d p_3d = static_cast<g2o::VertexPointXYZ *>(_vertices[1])->estimate();

  // Get the observed 2D point
  Eigen::Vector2d obs(_measurement);

  // Project the 3D point to the 2D point
  Eigen::Vector3d p_2d = R * p_3d + t_;
  p_2d = K_ * (p_2d / p_2d.z());

  // Calculate the error
  _error = obs - p_2d.head<2>();
}

G2O_USE_OPTIMIZATION_LIBRARY(dense)

BaSolver::BaSolver(std::array<double, 9> &camera_matrix, std::vector<double> &dist_coeffs)
{
  K_ = Eigen::Matrix3d::Identity();
  K_(0, 0) = camera_matrix[0];
  K_(1, 1) = camera_matrix[4];
  K_(0, 2) = camera_matrix[2];
  K_(1, 2) = camera_matrix[5];

  // Optimization information
  optimizer_.setVerbose(false);
  // Optimization method
  optimizer_.setAlgorithm(g2o::OptimizationAlgorithmFactory::instance()->construct(
      "lm_dense", solver_property_));
  // Initial step size
  lm_algorithm_ = (dynamic_cast<g2o::OptimizationAlgorithmLevenberg *>(
      const_cast<g2o::OptimizationAlgorithm *>(optimizer_.algorithm())));
  lm_algorithm_->setUserLambdaInit(0.1);
}

Eigen::Matrix3d BaSolver::solveBa(const Armor &armor,
                                  const Eigen::Vector3d &t_camera_armor,
                                  const Eigen::Matrix3d &R_camera_armor,
                                  const Eigen::Matrix3d &R_imu_camera) noexcept
{
  // Reset optimizer
  optimizer_.clear();

  // Essential coordinate system transformation
  Eigen::Matrix3d r_imu_armor = R_imu_camera * R_camera_armor;
  Sophus::SO3d r_camera_imu = Sophus::SO3d(R_imu_camera.transpose());

  // Compute the initial yaw from rotation matrix
  double initial_armor_yaw = NAN;
  auto theta_by_sin = std::asin(-r_imu_armor(0, 1));
  auto theta_by_cos = std::acos(r_imu_armor(1, 1));
  if (std::abs(theta_by_sin) > 1e-5)
  {
    initial_armor_yaw = theta_by_sin > 0 ? theta_by_cos : -theta_by_cos;
  }
  else
  {
    initial_armor_yaw = r_imu_armor(1, 1) > 0 ? 0 : CV_PI;
  }

  // Get the pitch angle of the armor
  //   double armor_pitch =
  //       armor.number == "outpost" ? -FIFTTEN_DEGREE_RAD : FIFTTEN_DEGREE_RAD;
  double armor_pitch = 0.0;
  Sophus::SO3d r_pitch = Sophus::SO3d::exp(Eigen::Vector3d(0, armor_pitch, 0));

  // Get the 3D points of the armor
  auto armor_size =
      armor.type == ArmorType::SMALL
          ? Eigen::Vector2d(SMALL_ARMOR_WIDTH / 1000, SMALL_ARMOR_HEIGHT / 1000)
          : Eigen::Vector2d(LARGE_ARMOR_WIDTH / 1000, LARGE_ARMOR_HEIGHT / 1000);
  auto object_points =
      Armor::buildObjectPoints<Eigen::Vector3d>(armor_size(0), armor_size(1));

  // Fill the optimizer
  size_t id_counter = 0;

  VertexYaw *v_yaw = new VertexYaw();
  v_yaw->setId(static_cast<int>(id_counter++));
  v_yaw->setEstimate(initial_armor_yaw);
  optimizer_.addVertex(v_yaw);

  const auto &landmarks = armor.landmarks();
  for (size_t i = 0; i < Armor::N_LANDMARKS; i++)
  {
    g2o::VertexPointXYZ *v_point = new g2o::VertexPointXYZ();
    v_point->setId(static_cast<int>(id_counter++));
    v_point->setEstimate(Eigen::Vector3d(object_points[i].x(), object_points[i].y(),
                                         object_points[i].z()));
    v_point->setFixed(true);
    optimizer_.addVertex(v_point);

    EdgeProjection *edge = new EdgeProjection(r_camera_imu, r_pitch, t_camera_armor, K_);
    edge->setId(static_cast<int>(id_counter++));
    edge->setVertex(0, v_yaw);
    edge->setVertex(1, v_point);
    edge->setMeasurement(Eigen::Vector2d(landmarks[i].x, landmarks[i].y));
    edge->setInformation(EdgeProjection::InfoMatrixType::Identity());
    edge->setRobustKernel(new g2o::RobustKernelHuber);
    optimizer_.addEdge(edge);
  }

  // Start optimizing
  optimizer_.initializeOptimization();
  optimizer_.optimize(20);

  // Get yaw angle after optimization
  double yaw_optimized = v_yaw->estimate();

  if (std::isnan(yaw_optimized))
  {
    RCLCPP_ERROR(rclcpp::get_logger("armor_detector"),
                 "Yaw angle is nan after optimization");
    return R_camera_armor;
  }

  Sophus::SO3d r_yaw = Sophus::SO3d::exp(Eigen::Vector3d(0, 0, yaw_optimized));
  return (r_camera_imu * r_yaw * r_pitch).matrix();
}
}  // namespace rm_auto_aim