#include <iostream>
#include <string.h>
#include <random>
#include <thread>
#include <unistd.h>
#include <optional>
#include <mutex>
#include <queue>

#include <x86_64-linux-gnu/sys/stat.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/point_cloud_conversion.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <gps_common/GPSFix.h>
#include <sensor_msgs/PointCloud2.h>
#include <ld_msgs/ld_can.h>

#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <boost/thread/thread.hpp>


#include <pcl/pcl_macros.h>
#include "userdefine.h"
#include "radarFactor.hpp"

#include "ikd_Tree.h"
#include "radar_ego_velocity_estimator/radar_ego_velocity_estimator.h"
#include "fast_gicp/gicp/fast_gicp.hpp"
#include "fast_gicp/gicp/fast_gicp_st.hpp"
#include "aloam_velodyne/tic_toc.h"
#include "aloam_velodyne/common.h"
#include "scancontext/Scancontext.h"

#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/RadarFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/slam/dataset.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/dataset.h>
#include <gtsam/nonlinear/LevenbergMarquardtParams.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/ISAM2.h>

#define MAX_SEARCH_RADIUS 2.0f
#define RADAR_RADIUS 80
#define SONGLING

using namespace std;
// using PointVector = KD_TREE<ikdTree_PointType>::PointVector;
using PointVector = KD_TREE<PointType>::PointVector;
using namespace gtsam;
using symbol_shorthand::B; // 陀螺仪残差  (ax,ay,az,gx,gy,gz)
using symbol_shorthand::V; // 用表示      速度导数(xdot,ydot,zdot)
using symbol_shorthand::X; // 用作表示    姿态(x,y,z,r,p,y)

// clang-format off
POINT_CLOUD_REGISTER_POINT_STRUCT(RadarPointCloudType,
                                  (float, x, x)
                                  (float, y, y)
                                  (float, z, z)
                                  (float, snr_db, snr_db)
                                  (float, noise_db,   noise_db)
                                  (float, v_doppler_mps,   v_doppler_mps)
                                  )


POINT_CLOUD_REGISTER_POINT_STRUCT (mmWaveCloudType,
                                    (float, x, x)
                                    (float, y, y)
                                    (float, z, z)
                                    (float, intensity, intensity)
                                    (float, velocity, velocity))

POINT_CLOUD_REGISTER_POINT_STRUCT (OculiiPointCloudType,
                                    (float, x, x)
                                    (float, y, y)
                                    (float, z, z)
                                    (float, Doppler, Doppler)
                                    (float, Range, Range)
                                    (float, Power, Power)
                                    (float, Alpha, Alpha)
                                    (float, Beta, Beta))

POINT_CLOUD_REGISTER_POINT_STRUCT (ColoRadarPointCloudType,
                                    (float, x, x)
                                    (float, y, y)
                                    (float, z, z)
                                    (float, intensity, intensity)
                                    (float, range, range)
                                    (float, doppler, doppler))

// clang-format on

double para_q[4] = {0, 0, 0, 1};
double para_t[3] = {0, 0, 0};
double output_time = 0;
double gt_update = 0;
double radar_update = 0;

pcl::PointCloud<pcl::PointXYZI>::Ptr src(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr tar(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr SubMap(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr scan_map(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr RadarCloudMap(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr downSizeFilterMap(new pcl::PointCloud<pcl::PointXYZI>);


Eigen::Vector3d t(para_t);
Eigen::Vector4d pos({0, 0, 0, 1});

Eigen::Map<Eigen::Quaterniond> q_last_curr(para_q);
Eigen::Map<Eigen::Vector3d> t_last_curr(para_t);

// odometry
// Transformation from current frame to world frame
Eigen::Quaterniond q_w_curr(1, 0, 0, 0);
Eigen::Vector3d t_w_curr(0, 0, 0);
Vector3 v_r, sigma_v_r;
Vector3 gyro_last;
Vector3 lbr (0.0901, -0.1449, 0.0249); //colo
// 初始化值

int vel_correction_count = 0;
int icp_correction_count = 0;
std::queue<Pose3> odomBuf;
std::queue<pcl::PointCloud<PointType>::Ptr> fullResBuf;

Eigen::Matrix<double, 3, 3> Rtrans;
Eigen::Matrix<double, 3, 3> R_enu_radar;
Eigen::Matrix<double, 4, 4> T_enu_radar;
Eigen::MatrixXd currOdom;
nav_msgs::Odometry RadarOdom;
nav_msgs::Path RadarPath;

//loop detect
std::mutex mKF;
std::mutex mBuf;
SCManager scManager;
std::queue<std::pair<int, int> > scLoopICPBuf;
pcl::VoxelGrid<PointType> downSizeFilterICP;
pcl::VoxelGrid<PointType> downSizeFilterScancontext;
std::vector<pcl::PointCloud<PointType>::Ptr> keyframeLaserClouds;
std::vector<std::pair<int,Pose6D>> keyframePoses;
std::vector<std::pair<int,Pose6D>> keyframePosesUpdated;
Pose6D odom_pose_prev {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; // init 
Pose6D odom_pose_curr {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; // init pose is zero 
double keyframeMeterGap;
double movementAccumulation = 1000000.0; // large value means must add the first given frame.
bool isNowKeyFrame = false;


//rosbag
bool initialed = false;
std::mutex mutex_1;
std::queue<ImuDataStamped> queue_imu;
std::queue<sensor_msgs::PointCloud2> queue_radar;
std::queue<nav_msgs::Odometry> queue_gt;
std::queue<gps_common::GPSFix> queue_gps;

//pose graph
NonlinearFactorGraph *graph = new NonlinearFactorGraph();
gtsam::ISAM2 *isam;
gtsam::Values isamCurrentEstimate;
Values initial_values;
ImuDataStamped last_imu;
ImuDataStamped imu_data;
reve::RadarEgoVelocityEstimator radar_ego_velocity;
PreintegrationType *imu_preintegrated_;
imuBias::ConstantBias prev_bias;
Values result;
NavState prev_state;
NavState prop_state;
std::mutex mtxPosegraph;
std::mutex mtxRecentPose;
std::string bag_path;
std::string topic_imu;
std::string topic_radar_scan;
std::string topic_lidar_gt;
std::string topic_gps_gt;

ros::Publisher pubRadarCloudSurround;
ros::Publisher pubRadarSubMap;
ros::Publisher pubRadarPath;
ros::Publisher pubRadarVel;
ros::Publisher pubOdomGT;

static Eigen::Vector3d R2rpy(const Eigen::Matrix3d &R)
{
  Eigen::Vector3d n = R.col(0);
  Eigen::Vector3d o = R.col(1);
  Eigen::Vector3d a = R.col(2);

  Eigen::Vector3d rpy(3);
  double y = atan2(n(1), n(0));
  double p = atan2(-n(2), n(0) * cos(y) + n(1) * sin(y));
  double r = atan2(a(0) * sin(y) - a(1) * cos(y), -o(0) * sin(y) + o(1) * cos(y));
  rpy(0) = r;
  rpy(1) = p;
  rpy(2) = y;

  return rpy / M_PI * 180.0;
}

static double degree2rad(double theta)
{
    return theta * M_PI / 180.0;
}

double transDiff(const Pose6D& _p1, const Pose6D& _p2)
{
    return sqrt( (_p1.x - _p2.x)*(_p1.x - _p2.x) + (_p1.y - _p2.y)*(_p1.y - _p2.y) + (_p1.z - _p2.z)*(_p1.z - _p2.z) );
} // transDiff

void pointAssociateToMap(PointType const *const pi, PointType *const po)
{
  Eigen::Vector3d point_curr(pi->x, pi->y, pi->z);
  Eigen::Vector3d point_w = Rtrans * point_curr + t_w_curr;
  po->x = point_w.x();
  po->y = point_w.y();
  po->z = point_w.z();
  po->intensity = pi->intensity;
}

void pointAssociateToSubMap(PointType const *const pi, PointType *const po, Eigen::Matrix<double, 4, 4> Trans)
{
  Eigen::Vector4d point_curr(pi->x, pi->y, pi->z, 1.0);
  Eigen::Vector4d point_w = Trans.inverse() * point_curr;
  po->x = point_w.x();
  po->y = point_w.y();
  po->z = point_w.z();
  po->intensity = pi->intensity;
}

int get_file_lines(const std::string file_path)
{
  char c;
  int line = 0;
  if (boost::filesystem::exists(file_path))
  {
    std::ifstream file(file_path, std::ifstream::in);
    while (file.get(c))
    {
      if (c == '\n')
        line++;
    }
  }
  else
  {
    std::cout << "wrong file path !" << endl;
  }
  return line;
}

std::vector<float> read_radar_featurePoint(const std::string radar_data_path)
{
  if (boost::filesystem::exists(radar_data_path))
  {
    std::string value;
    std::vector<float> radar_data_buffer;
    int lines = get_file_lines(radar_data_path);
    std::ifstream radar_data_file(radar_data_path, std::ifstream::in);

    for (int i = 0; i < lines; i++)
    {
      for (int j = 0; j < 2; j++)
      {
        getline(radar_data_file, value, ' ');
        radar_data_buffer.push_back(atof(value.c_str()));
      }
      getline(radar_data_file, value, '\n'); // 换行
      radar_data_buffer.push_back(atof(value.c_str()));
    }
    return radar_data_buffer;
  }
  else
  {
    std::vector<float> radar_data_buffer(0);
    return radar_data_buffer;
  }
}

void callbackIMU(const sensor_msgs::ImuConstPtr &imu_msgs);
void callbackRadarScan(const sensor_msgs::PointCloud2ConstPtr &radar_msg);
void callbackGtPose(const nav_msgs::OdometryConstPtr &gt_msg);
void callbackGPSPose(const gps_common::GPSFixConstPtr &gps_msg);
bool pcl2msgToPcl(const sensor_msgs::PointCloud2 &pcl_msg, pcl::PointCloud<RadarPointCloudType> &scan);
void main_task();
void config_init(radar_ego_velocity_estimation::RadarEgoVelocityEstimatorConfig &config);

gtsam::Pose3 Pose6DtoGTSAMPose3(const Pose6D& p)
{
    return gtsam::Pose3( gtsam::Rot3::RzRyRx(p.roll, p.pitch, p.yaw), gtsam::Point3(p.x, p.y, p.z) );
} // Pose6DtoGTSAMPose3

Pose6D GTSAMPose3toPose6D(const gtsam::Pose3& p)
{
  Pose6D rtval;
  rtval.x = p.x();
  rtval.y = p.y();
  rtval.z = p.z();
  rtval.roll = p.rotation().roll();
  rtval.pitch = p.rotation().pitch();
  rtval.yaw = p.rotation().yaw();
  return rtval;
} // GTSAMPose3toPose6D

pcl::PointCloud<PointType>::Ptr local2global(const pcl::PointCloud<PointType>::Ptr &cloudIn, const Pose6D& tf)
{
    pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

    int cloudSize = cloudIn->size();
    cloudOut->resize(cloudSize);

    Eigen::Affine3f transCur = pcl::getTransformation(tf.x, tf.y, tf.z, tf.roll, tf.pitch, tf.yaw);

    int numberOfCores = 12;
#pragma omp parallel for num_threads(numberOfCores)
    for (int i = 0; i < cloudSize; ++i)
    {
        const auto &pointFrom = cloudIn->points[i];
        cloudOut->points[i].x = transCur(0,0) * pointFrom.x + transCur(0,1) * pointFrom.y + transCur(0,2) * pointFrom.z + transCur(0,3);
        cloudOut->points[i].y = transCur(1,0) * pointFrom.x + transCur(1,1) * pointFrom.y + transCur(1,2) * pointFrom.z + transCur(1,3);
        cloudOut->points[i].z = transCur(2,0) * pointFrom.x + transCur(2,1) * pointFrom.y + transCur(2,2) * pointFrom.z + transCur(2,3);
        cloudOut->points[i].intensity = pointFrom.intensity;
    }

    return cloudOut;
}

void initNoises( void )
{
  Eigen::Matrix<double, 9, 1> initial_state = Eigen::Matrix<double, 9, 1>::Zero();
  Point3 prior_point(initial_state.head<3>());
  Vector3 prior_euler(initial_state.segment<3>(3));
  Vector3 prior_velocity(initial_state.tail<3>());
  Rot3 prior_rotation = Rot3::ypr(degree2rad(prior_euler(0)),
                                  degree2rad(prior_euler(1)),
                                  degree2rad(prior_euler(2)));
  Pose3 prior_pose(prior_rotation, prior_point); // 初始位姿

  imuBias::ConstantBias prior_imu_bias; // 残差，默认设为0
  NavState prior_nav(prior_pose, prior_velocity);

  Eigen::Matrix3d bRn = prior_nav.R();
  Eigen::Vector3d bTn = prior_nav.t();
  Eigen::Matrix<double, 4, 4> init2nav_Trans = Eigen::Matrix<double, 4, 4>::Zero();
  init2nav_Trans.topLeftCorner<3, 3>() = bRn;
  init2nav_Trans.topRightCorner<3, 1>() = bTn;
  init2nav_Trans(3, 3) = 1;

  // 位姿
  initial_values.insert(X(vel_correction_count), prior_pose);
  // 速度
  initial_values.insert(V(vel_correction_count), prior_velocity);
  // 残差
  initial_values.insert(B(vel_correction_count), prior_imu_bias);
  cout << "initial state:\n"
       << initial_state.transpose() << endl;
  // 设置噪声模型
  // 一般为设置为对角噪声
  Vector6 pose_noise_sigma;
  pose_noise_sigma << 0.01, 0.01, 0.01, 0.1, 0.1, 0.1;
  noiseModel::Diagonal::shared_ptr pose_noise_model = noiseModel::Diagonal::Sigmas(pose_noise_sigma);
  noiseModel::Diagonal::shared_ptr velocity_noise_model = noiseModel::Isotropic::Sigma(3, 0.1);
  noiseModel::Diagonal::shared_ptr bias_noise_model = noiseModel::Isotropic::Sigma(6, 0.001);
  NonlinearFactorGraph *graph = new NonlinearFactorGraph();
  graph->add(PriorFactor<Pose3>(X(vel_correction_count), prior_pose, pose_noise_model));
  graph->add(PriorFactor<Vector3>(V(vel_correction_count), prior_velocity, velocity_noise_model));
  graph->add(PriorFactor<imuBias::ConstantBias>(B(vel_correction_count), prior_imu_bias, bias_noise_model));
  // 使用传感器信息构建IMU的噪声模型
  double accel_noise_sigma = 0.00049;
  double gyro_noise_sigma = 0.0000174444;
  double accel_bias_rw_sigma = 0.0000104050763;
  double gyro_bias_rw_sigma = 0.00000342641249;

  Matrix33 measured_acc_cov = Matrix33::Identity(3, 3) * pow(accel_noise_sigma, 2);
  Matrix33 measured_omega_cov = Matrix33::Identity(3, 3) * pow(gyro_noise_sigma, 2);
  Matrix33 integration_error_cov = Matrix33::Identity(3, 3) * 1e-8; // 速度积分误差
  Matrix33 bias_acc_cov = Matrix33::Identity(3, 3) * pow(accel_bias_rw_sigma, 2);
  Matrix33 bias_omega_cov = Matrix33::Identity(3, 3) * pow(gyro_bias_rw_sigma, 2);
  Matrix66 bias_acc_omega_int = Matrix66::Identity(6, 6) * 1e-5; // 积分骗到误差

  boost::shared_ptr<PreintegratedCombinedMeasurements::Params> p = PreintegratedCombinedMeasurements::Params::MakeSharedD();
  // MakeSharedD:NED坐标系，g默认为 9.81，这里设置为0
  // MakeSharedU：NEU坐标系，g默认为 9.81

  Vector3 gyro_last = Vector3(0.0, 0.0, 0.0);
  Vector3 lbr(0.0901, -0.1449, 0.0249); // colo

  // 设置预积分分参数
  p->accelerometerCovariance = measured_acc_cov;
  p->integrationCovariance = integration_error_cov;
  p->gyroscopeCovariance = measured_omega_cov;

  // 预计分测量值
  p->biasAccCovariance = bias_acc_cov;
  p->biasAccOmegaInt = bias_acc_omega_int;
  p->biasOmegaCovariance = bias_omega_cov;
#ifdef USE_COMBINED
  imu_preintegrated_ = new PreintegratedCombinedMeasurements(p, prior_imu_bias);
#else
  imu_preintegrated_ = new PreintegratedImuMeasurements(p, prior_imu_bias);
#endif
  // 保存上一次的imu积分值和结果
  NavState state_tmp(prior_pose,prior_velocity);
  prev_state = state_tmp;
  prop_state = prev_state;
  prev_bias = prior_imu_bias; //
}

void process_viz_map()
{
    float vizmapFrequency = 0.1;
    ros::Rate rate(vizmapFrequency);
    while (ros::ok()) {
        rate.sleep();
        if (downSizeFilterMap->size() > 1)
        {
          sensor_msgs::PointCloud2 RadarCloudSurround;
          pcl::toROSMsg(*downSizeFilterMap, RadarCloudSurround);
          RadarCloudSurround.header.stamp = ros::Time().now();
          RadarCloudSurround.header.frame_id = "/camera_init";
          pubRadarCloudSurround.publish(RadarCloudSurround);

          sensor_msgs::PointCloud2 RadarSubMap;
          pcl::toROSMsg(*SubMap, RadarSubMap);
          RadarSubMap.header.stamp = ros::Time().now();
          RadarSubMap.header.frame_id = "/camera_init";
          pubRadarSubMap.publish(RadarSubMap);
        }
    }
}

void process_viz_path(void)
{
    float hz = 10.0;
    ros::Rate rate(hz);
    
    while (ros::ok()) {
        rate.sleep();
        if (keyframePosesUpdated.size() > 1)
        {
          RadarOdom.pose.pose.position.x = currOdom(0, 3);
          RadarOdom.pose.pose.position.y = currOdom(1, 3);
          RadarOdom.pose.pose.position.z = currOdom(2, 3);
          Eigen::Quaterniond q_tmp(currOdom.topLeftCorner<3, 3>());
          RadarOdom.pose.pose.orientation.w = q_tmp.w();
          RadarOdom.pose.pose.orientation.x = q_tmp.x();
          RadarOdom.pose.pose.orientation.y = q_tmp.y();
          RadarOdom.pose.pose.orientation.z = q_tmp.z();

          geometry_msgs::PoseStamped RadarPose;
          RadarPose.pose = RadarOdom.pose.pose;
          RadarPath.header.stamp = ros::Time().now();
          RadarPath.header.frame_id = "/camera_init";
          RadarPath.poses.push_back(RadarPose);
          pubRadarPath.publish(RadarPath);

          geometry_msgs::Twist RadarVel;
          RadarVel.linear.x = v_r(0);
          RadarVel.linear.y = v_r(1);
          RadarVel.linear.z = v_r(2);
          pubRadarVel.publish(RadarVel);
        }
    }
}

void performSCLoopClosure(void)
{
    if( keyframePoses.size() < scManager.NUM_EXCLUDE_RECENT) // do not try too early 
        return;

    auto detectResult = scManager.detectLoopClosureID(); // first: nn index, second: yaw diff
    int SCclosestHistoryFrameID = detectResult.first;
    if( SCclosestHistoryFrameID != -1 ) { 
        const int prev_node_idx = SCclosestHistoryFrameID;
        const int curr_node_idx = keyframePoses.size() - 1; // because cpp starts 0 and ends n-1
        cout << "Loop detected! - between " << prev_node_idx << " and " << curr_node_idx << "" << endl;

        mBuf.lock();
        scLoopICPBuf.push(std::pair<int, int>(prev_node_idx, curr_node_idx));
        // addding actual 6D constraints in the other thread, icp_calculation.
        mBuf.unlock();
    }
} // performSCLoopClosure

void process_odom()
{
    // read bag
  rosbag::Bag source_bag;
  source_bag.open(bag_path, rosbag::bagmode::Read);
  std::vector<std::string> topics;
  topics.push_back(topic_imu);
  topics.push_back(topic_radar_scan);
  topics.push_back(topic_lidar_gt);
  topics.push_back(topic_gps_gt);

  rosbag::View view(source_bag, rosbag::TopicQuery(topics));
  sensor_msgs::ImuPtr imu_msg_new(new sensor_msgs::Imu);
  for (const rosbag::MessageInstance &m : view)
  {
    const auto topic = m.getTopic();
    if (topic == topic_imu)
    {
#ifdef SONGHONG
      const auto imu_msg_bag = m.instantiate<ld_msgs::ld_can>();
      bool Imu_update = false;
      if (imu_msg_bag != NULL && imu_msg_bag->ID == 0x605)
      {
        imu_msg_new->header = imu_msg_bag->header;
        double ax = (int16_t)((imu_msg_bag->DATA[1] << 8) + imu_msg_bag->DATA[0]) * 0.01;
        double ay = (int16_t)((imu_msg_bag->DATA[3] << 8) + imu_msg_bag->DATA[2]) * 0.01;
        double az = (int16_t)((imu_msg_bag->DATA[5] << 8) + imu_msg_bag->DATA[4]) * 0.01;
        imu_msg_new->linear_acceleration.x = ax;
        imu_msg_new->linear_acceleration.y = ay;
        imu_msg_new->linear_acceleration.z = az;
      }
      else if (imu_msg_bag != NULL && imu_msg_bag->ID == 0x608)
      {
        double wx = (int16_t)((imu_msg_bag->DATA[1] << 8) + imu_msg_bag->DATA[0]) * (0.01 * M_PI / 180.0f);
        double wy = (int16_t)((imu_msg_bag->DATA[3] << 8) + imu_msg_bag->DATA[2]) * (0.01 * M_PI / 180.0f);
        double wz = (int16_t)((imu_msg_bag->DATA[5] << 8) + imu_msg_bag->DATA[4]) * (0.01 * M_PI / 180.0f);
        imu_msg_new->angular_velocity.x = wx;
        imu_msg_new->angular_velocity.y = wy;
        imu_msg_new->angular_velocity.z = wz;
        Imu_update = true;
      }
      if (Imu_update)
      {
        callbackIMU(imu_msg_new);
      }
#endif

#ifdef SONGLING
      const auto imu_msg_bag = m.instantiate<sensor_msgs::Imu>();
      if (imu_msg_bag != NULL)
      {
        callbackIMU(imu_msg_bag);
      }
#endif
    }

    else if (topic == topic_radar_scan)
    {
      const auto radar_scan = m.instantiate<sensor_msgs::PointCloud>();
      sensor_msgs::PointCloud2Ptr New_radar_scan(new sensor_msgs::PointCloud2);
      convertPointCloudToPointCloud2(*radar_scan, *New_radar_scan);
      if (radar_scan != NULL)
        callbackRadarScan(New_radar_scan);
    }

    else if (topic == topic_lidar_gt)
    {
      const auto gt_pose = m.instantiate<nav_msgs::Odometry>();

      if (gt_pose != NULL)
      {
        gt_pose->header.frame_id = "/camera_init";
        pubOdomGT.publish(gt_pose);
        callbackGtPose(gt_pose);
      }
    }

    else if (topic == topic_gps_gt)
    {
      const auto gps_pose = m.instantiate<gps_common::GPSFix>();

      if (gps_pose != NULL)
      {
        callbackGPSPose(gps_pose);
      }
    }
    main_task();
  }
}

void process_pg()
{
  int odom_node = 0;
  while (1)
  {
    while (!odomBuf.empty() && !fullResBuf.empty())
    {
      //
      // pop and check keyframe is or not
      //
      mBuf.lock();
      while (!odomBuf.empty())
        odomBuf.pop();
      if (odomBuf.empty())
      {
        mBuf.unlock();
        break;
      }

      // Time equal check

      // TODO

      pcl::PointCloud<PointType>::Ptr thisKeyFrame(new pcl::PointCloud<PointType>());
      pcl::copyPointCloud(*fullResBuf.front(),*thisKeyFrame);
      fullResBuf.pop();

      Pose6D pose_curr = GTSAMPose3toPose6D(odomBuf.front());
      odomBuf.pop();
      odom_node++;

      //
      // Early reject by counting local delta movement (for equi-spereated kf drop)
      //
      odom_pose_prev = odom_pose_curr;
      odom_pose_curr = pose_curr;
      double delta_translation = transDiff(odom_pose_prev, odom_pose_curr);
      movementAccumulation += delta_translation;

      if (movementAccumulation > keyframeMeterGap)
      {
        isNowKeyFrame = true;
        movementAccumulation = 0.0; // reset
      }
      else
      {
        isNowKeyFrame = false;
      }

      if (!isNowKeyFrame)
        continue;

      // delete outlier by KNN
      pcl::KdTreeFLANN<pcl::PointXYZI> KNNkdtree;
      KNNkdtree.setInputCloud(thisKeyFrame);
      float radius = 1;
      for (int i = 0; i < thisKeyFrame->size(); i++)
      {
        std::vector<int> pointIndxRadiusSearch;
        std::vector<float> pointNRSquareDis;
        int size = KNNkdtree.radiusSearch(thisKeyFrame->points[i], radius, pointIndxRadiusSearch, pointNRSquareDis);
        if (size <= 15)
        {
          thisKeyFrame->erase(thisKeyFrame->begin() + i);
        }
      }

      //
      // Save data and Add consecutive node
      //
      pcl::PointCloud<PointType>::Ptr thisKeyFrameDS(new pcl::PointCloud<PointType>());
      downSizeFilterScancontext.setInputCloud(thisKeyFrame);
      downSizeFilterScancontext.filter(*thisKeyFrameDS);

      mKF.lock();
      keyframeLaserClouds.push_back(thisKeyFrameDS);
      // std::stringstream filepath;
      // filepath << "/home/invoker/catkin_ws/src/navtech-radar-slam/keyframe/" << keyframeLaserClouds.size() - 1 << ".pcd";
      // pcl::io::savePCDFileASCII(filepath.str(), *thisKeyFrameDS);
      keyframePoses.push_back(std::pair(odom_node,pose_curr));
      keyframePosesUpdated.push_back(std::pair(odom_node,pose_curr)); // init
      scManager.makeAndSaveScancontextAndKeys(*thisKeyFrameDS);
      mKF.unlock();
      const int prev_node_idx = keyframePoses.size() - 2;
      const int curr_node_idx = keyframePoses.size() - 1; // becuase cpp starts with 0 (actually this index could be any number, but for simple implementation, we follow sequential indexing)
      // gtsam::Pose3 poseFrom = Pose6DtoGTSAMPose3(keyframePoses.at(prev_node_idx).second);
      // gtsam::Pose3 poseTo = Pose6DtoGTSAMPose3(keyframePoses.at(curr_node_idx).second);

      if (curr_node_idx % 5 == 0)
        cout << "posegraph keyframe node " << curr_node_idx << " added." << endl;
    }
    // if want to print the current graph, use gtSAMgraph.print("\nFactor Graph:\n");
  

        // ps.
        // scan context detector is running in another thread (in constant Hz, e.g., 1 Hz)
        // pub path and point cloud in another thread

        // wait (must required for running the while loop)
        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
} // process_pg

void process_lcd(void)
{
    float loopClosureFrequency = 1.0; // can change
    ros::Rate rate(loopClosureFrequency);
    while (ros::ok())
    {
        rate.sleep();
        performSCLoopClosure();
        // performRSLoopClosure(); // TODO
    }
} // process_lcd

void updatePoses(void)
{
    mKF.lock();
    for (int key_idx,node_idx=0; node_idx < int(initial_values.size()); node_idx++)
    {
      if(node_idx == keyframePosesUpdated[key_idx].first && key_idx < int(keyframePosesUpdated.size()))
      {
        Pose6D& p = keyframePosesUpdated[key_idx].second;
        p.x = initial_values.at<gtsam::Pose3>(node_idx).translation().x();
        p.y = initial_values.at<gtsam::Pose3>(node_idx).translation().y();
        p.z = initial_values.at<gtsam::Pose3>(node_idx).translation().z();
        p.roll = initial_values.at<gtsam::Pose3>(node_idx).rotation().roll();
        p.pitch = initial_values.at<gtsam::Pose3>(node_idx).rotation().pitch();
        p.yaw = initial_values.at<gtsam::Pose3>(node_idx).rotation().yaw();
        key_idx++;
      }
    }
    mKF.unlock();

    mtxRecentPose.lock();
    const gtsam::Pose3& lastOptimizedPose = isamCurrentEstimate.at<gtsam::Pose3>(int(isamCurrentEstimate.size())-1);
    mtxRecentPose.unlock();
} // updatePoses

void runISAM2opt(void)
{
    // called when a variable added
    isam->update(*graph, initial_values);
    isam->update();

    graph->resize(0);
    initial_values.clear();

    isamCurrentEstimate = isam->calculateEstimate();
    updatePoses();
}

void loopFindNearKeyframesCloud( pcl::PointCloud<PointType>::Ptr& nearKeyframes, const int& key, const int& submap_size, const int& root_kf_idx)
{
    // extract and stacking near keyframes (in global coord)
    nearKeyframes->clear();
    for (int i = -submap_size; i <= submap_size; ++i) {
        int keyNear = key + i;
        if (keyNear < 0 || keyNear >= keyframeLaserClouds.size() )
            continue;

        mKF.lock();
        *nearKeyframes += * local2global(keyframeLaserClouds[keyNear], keyframePosesUpdated[root_kf_idx].second);
        mKF.unlock();
    }

    if (nearKeyframes->empty())
        return;

    // downsample near keyframes
    pcl::PointCloud<PointType>::Ptr cloud_temp(new pcl::PointCloud<PointType>());
    downSizeFilterICP.setInputCloud(nearKeyframes);
    downSizeFilterICP.filter(*cloud_temp);
    *nearKeyframes = *cloud_temp;
} // loopFindNearKeyframesCloud

std::optional<gtsam::Pose3> doICPVirtualRelative(int _loop_kf_idx, int _curr_kf_idx)
{
  // parse pointclouds
  int historyKeyframeSearchNum = 10; // enough. ex. [-25, 25] covers submap length of 50x1 = 50m if every kf gap is 1m
  pcl::PointCloud<PointType>::Ptr cureKeyframeCloud(new pcl::PointCloud<PointType>());
  pcl::PointCloud<PointType>::Ptr targetKeyframeCloud(new pcl::PointCloud<PointType>());
  loopFindNearKeyframesCloud(cureKeyframeCloud, _curr_kf_idx, 0, _loop_kf_idx); // use same root of loop kf idx
  loopFindNearKeyframesCloud(targetKeyframeCloud, _loop_kf_idx, historyKeyframeSearchNum, _loop_kf_idx);

  // loop verification
  // sensor_msgs::PointCloud2 cureKeyframeCloudMsg;
  // pcl::toROSMsg(*cureKeyframeCloud, cureKeyframeCloudMsg);
  // cureKeyframeCloudMsg.header.frame_id = "/camera_init";
  // pubLoopScanLocal.publish(cureKeyframeCloudMsg);

  // sensor_msgs::PointCloud2 targetKeyframeCloudMsg;
  // pcl::toROSMsg(*targetKeyframeCloud, targetKeyframeCloudMsg);
  // targetKeyframeCloudMsg.header.frame_id = "/camera_init";
  // pubLoopSubmapLocal.publish(targetKeyframeCloudMsg);

  // ICP Settings
  pcl::IterativeClosestPoint<PointType, PointType> icp;
  icp.setMaxCorrespondenceDistance(150); // giseop , use a value can cover 2*historyKeyframeSearchNum range in meter
  icp.setMaximumIterations(100);
  icp.setTransformationEpsilon(1e-6);
  icp.setEuclideanFitnessEpsilon(1e-6);
  icp.setRANSACIterations(0);

  // Align pointclouds
  icp.setInputSource(cureKeyframeCloud);
  icp.setInputTarget(targetKeyframeCloud);
  pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
  icp.align(*unused_result);

  float loopFitnessScoreThreshold = 2.0; // user parameter but fixed low value is safe.
  if (icp.hasConverged() == false || icp.getFitnessScore() > loopFitnessScoreThreshold)
  {
    std::cout << "[SC loop] ICP fitness test failed (" << icp.getFitnessScore() << " > " << loopFitnessScoreThreshold << "). Reject this SC loop." << std::endl;
    return std::nullopt;
  }
  else
  {
    std::cout << "[SC loop] ICP fitness test passed (" << icp.getFitnessScore() << " < " << loopFitnessScoreThreshold << "). Add this SC loop." << std::endl;
  }

  // Get pose transformation
  float x, y, z, roll, pitch, yaw;
  Eigen::Affine3f correctionLidarFrame;
  correctionLidarFrame = icp.getFinalTransformation();
  pcl::getTranslationAndEulerAngles(correctionLidarFrame, x, y, z, roll, pitch, yaw);
  gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(x, y, z));
  gtsam::Pose3 poseTo = Pose3(Rot3::RzRyRx(0.0, 0.0, 0.0), Point3(0.0, 0.0, 0.0));

  return poseFrom.between(poseTo);
} // doICPVirtualRelative

void process_LC_icp(void)
{
    while(1)
    {
		while ( !scLoopICPBuf.empty() )
        {
            if( scLoopICPBuf.size() > 30 ) {
                ROS_WARN("Too many loop clousre candidates to be ICPed is waiting ... Do process_lcd less frequently (adjust loopClosureFrequency)");
            }

            mBuf.lock();
            std::pair<int, int> loop_idx_pair = scLoopICPBuf.front();
            scLoopICPBuf.pop();
            mBuf.unlock();

            const int prev_node_idx = loop_idx_pair.first;
            const int curr_node_idx = loop_idx_pair.second;
            auto relative_pose_optional = doICPVirtualRelative(prev_node_idx, curr_node_idx);
            if(relative_pose_optional) {
                gtsam::Pose3 relative_pose = relative_pose_optional.value();
                Vector6 icp_noise_sigma;
                icp_noise_sigma << 0.001, 0.001, 0.001, 0.02, 0.02, 0.02;
                noiseModel::Diagonal::shared_ptr icp_noise_model = noiseModel::Diagonal::Sigmas(icp_noise_sigma);
                mtxPosegraph.lock();
                graph->add(gtsam::BetweenFactor<gtsam::Pose3>(keyframePoses[prev_node_idx].first, keyframePoses[curr_node_idx].first, relative_pose, icp_noise_model));
                // runISAM2opt();
                updatePoses();
                mtxPosegraph.unlock();
            }
        }

        // wait (must required for running the while loop)
        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
} // process_LC_icp

int main(int argc, char **argv)
{

  ros::init(argc, argv, "radar_odometry");
  ros::NodeHandle n("~");
  pubRadarCloudSurround = n.advertise<sensor_msgs::PointCloud2>("/radar_cloud_surround", 100);
  pubRadarSubMap = n.advertise<sensor_msgs::PointCloud2>("/radar_submap", 100);
  pubRadarPath = n.advertise<nav_msgs::Path>("/radar_path", 5);
  pubRadarVel = n.advertise<geometry_msgs::Twist>("/radar_vel", 5);
  pubOdomGT = n.advertise<nav_msgs::Odometry>("/lidar_gt", 100);
  radar_ego_velocity_estimation::RadarEgoVelocityEstimatorConfig config;
  config_init(config);
  radar_ego_velocity.configure(config);
  initNoises();
  std::size_t order = 1;
  n.getParam("bag_path", bag_path);
  n.getParam("topic_imu", topic_imu);
  n.getParam("topic_radar_scan", topic_radar_scan);
  n.getParam("topic_lidar_gt", topic_lidar_gt);
  n.getParam("topic_gps_gt", topic_gps_gt);

  // colo radar
  // R_enu_radar << 0.000796, -1, 0,
  //     1, 0.000796, 0,
  //     0, 0, 1;

#ifdef SONGHONG
  T_enu_radar << 0.99661164, -0.0819717, 0.010574, 3.7759,
    -0.081822456, -0.9965847, -0.01385872, -0.24948372,
    0.011714, 0.01290198, -0.999848363, -0.0439992834,
    0.0, 0.0, 0.0, 1.0;
#endif

#ifdef SONGLING
  T_enu_radar << 0.99984752, -0.0156398, -0.0055276, 0.2801,
  -0.0157201, -0.9997379, -0.0177237, -0.0257,
  -0.0052882, 0.01778338, -0.9998615, 0.0753,
  0.0, 0.0, 0.0, 1.0;
#endif
  

  

  currOdom = Eigen::Matrix<double, 4, 4>::Identity();

  std::thread posegraph_odom {process_odom}; // pose odom construction
  std::thread posegraph_slam {process_pg}; // pose graph construction
	std::thread lc_detection {process_lcd}; // loop closure detection 
  std::thread icp_calculation {process_LC_icp}; // loop constraint calculation via icp 
	std::thread viz_map {process_viz_map}; // visualization - map (low frequency because it is heavy)
	std::thread viz_path {process_viz_path}; // visualization - path (high frequency)

  ros::spin();
}



void main_task()
{
  pcl::PointCloud<pcl::PointXYZI>::Ptr Final(new pcl::PointCloud<pcl::PointXYZI>);

  mutex_1.lock();

  if (!queue_radar.empty())
  {
    if (initialed)
    {
      pcl::copyPointCloud(*src, *tar);
    }
    src->clear();

    auto radar_data_msg = queue_radar.front();
    sensor_msgs::PointCloud2 inlier_radar_scan;
    radar_ego_velocity.estimate(radar_data_msg, v_r, sigma_v_r, inlier_radar_scan);
    vel_correction_count++;
    R_enu_radar = T_enu_radar.topLeftCorner<3,3>();
    v_r = R_enu_radar*v_r;
    gyro_last = gyro_last - prev_bias.vector().tail<3>();
    Eigen::Matrix<double, 3, 3> gyro_x;
    gyro_x << 0, -gyro_last(2), gyro_last(1),
        gyro_last(2), 0, -gyro_last(0),
        -gyro_last(1), gyro_last(0), 0;

    v_r = v_r - gyro_x * lbr;
    noiseModel::Diagonal::shared_ptr radar_noise_model = noiseModel::Diagonal::Sigmas(sigma_v_r);
    RadarFactor radar_factor(X(vel_correction_count), V(vel_correction_count),
                             v_r,
                             radar_noise_model);

    pcl::PointCloud<OculiiPointCloudType> scan_OculliRadar;
    pcl::PCLPointCloud2 pcl_pc2;
    pcl_conversions::toPCL(inlier_radar_scan, pcl_pc2);
    pcl::fromPCLPointCloud2(pcl_pc2, scan_OculliRadar);
    int point_num = scan_OculliRadar.size();
    for (size_t i = 0; i < point_num; i++)
    {
      pcl::PointXYZI p_sel;
      p_sel.x = scan_OculliRadar.points[i].x;
      p_sel.y = scan_OculliRadar.points[i].y;
      p_sel.z = scan_OculliRadar.points[i].z;
      p_sel.intensity = scan_OculliRadar.points[i].Power;
      src->push_back(p_sel);
    }

    if (!initialed)
    {
      pcl::copyPointCloud(*src, *tar);
      initialed = true;
    }

    radar_update = radar_data_msg.header.stamp.toSec();
    queue_radar.pop();
    double time_diff = gt_update - radar_update;

      // GICP
      fast_gicp::FastGICPSingleThread<pcl::PointXYZI, pcl::PointXYZI> fgicp_st;
      fgicp_st.clearTarget();
      fgicp_st.clearSource();
      fgicp_st.setInputTarget(tar);
      fgicp_st.setInputSource(src);
      fgicp_st.setCorrespondenceRandomness(5);
      fgicp_st.align(*Final);
      double score = fgicp_st.getFitnessScore();

      std::cout << "has converged:" << fgicp_st.hasConverged() << " score: " << fgicp_st.getFitnessScore() << std::endl;
      std::cout << fgicp_st.getFinalTransformation() << std::endl;

      Eigen::Matrix<double, 4, 4> icp_result = T_enu_radar*fgicp_st.getFinalTransformation().cast<double>()*T_enu_radar.inverse();
      
      currOdom = currOdom * icp_result;
      Rtrans = currOdom.topLeftCorner<3, 3>();
      t_w_curr = currOdom.topRightCorner<3, 1>();

      // Get pose transformation
      float x, y, z, roll, pitch, yaw;
      Eigen::Affine3f correctionLidarFrame;
      correctionLidarFrame = fgicp_st.getFinalTransformation();
      pcl::getTranslationAndEulerAngles(correctionLidarFrame, x, y, z, roll, pitch, yaw);
      gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(x, y, z));
      gtsam::Pose3 poseTo = Pose3(Rot3::RzRyRx(0.0, 0.0, 0.0), Point3(0.0, 0.0, 0.0));
    
    Vector6 icp_noise_sigma;
    icp_noise_sigma << 0.001, 0.001, 0.001, 0.02, 0.02, 0.02;
    noiseModel::Diagonal::shared_ptr icp_noise_model = noiseModel::Diagonal::Sigmas(icp_noise_sigma);
    icp_correction_count = vel_correction_count;

    // 预计分测量值
    PreintegratedCombinedMeasurements *preint_imu_combined = dynamic_cast<PreintegratedCombinedMeasurements *>(imu_preintegrated_);
    // IMU 因子
    // typedef NoiseModelFactor6<Pose3, Vector3, Pose3, Vector3,imuBias::ConstantBias, imuBias::ConstantBias>
    CombinedImuFactor imu_factor(X(vel_correction_count - 1), V(vel_correction_count - 1),
                                 X(vel_correction_count), V(vel_correction_count),
                                 B(vel_correction_count - 1), B(vel_correction_count),
                                 *preint_imu_combined);

    mtxPosegraph.lock();
    graph->add(radar_factor);
    graph->add(gtsam::BetweenFactor<gtsam::Pose3>(X(icp_correction_count-1), X(icp_correction_count), poseFrom.between(poseTo), icp_noise_model));
    graph->add(imu_factor);

    // 迭代更新求解imu预测值
    prop_state = imu_preintegrated_->predict(prev_state, prev_bias);
    Eigen::Vector3d curr_euler = R2rpy(prop_state.pose().rotation().toQuaternion().toRotationMatrix());

    initial_values.insert(X(vel_correction_count), prop_state.pose());
    initial_values.insert(V(vel_correction_count), prop_state.v());
    initial_values.insert(B(vel_correction_count), prev_bias);

    LevenbergMarquardtOptimizer optimizer(*graph, initial_values);
    result = optimizer.optimize();
    mtxPosegraph.unlock();

    // 更新下一步预计分初始值
    // 导航状态
    prev_state = NavState(result.at<Pose3>(X(vel_correction_count)),
                          result.at<Vector3>(V(vel_correction_count)));
    // 偏导数
    prev_bias = result.at<imuBias::ConstantBias>(B(vel_correction_count));
    // 更新预计分值
    imu_preintegrated_->resetIntegrationAndSetBias(prev_bias);
    // 更新关键帧位姿
    updatePoses();

    mBuf.lock();
    odomBuf.push(result.at<Pose3>(X(vel_correction_count))) ;
    mBuf.unlock();

    mBuf.lock();
    fullResBuf.push(src);
    mBuf.unlock();
    
  }

  if (!queue_imu.empty())
  {
    auto imu_msg = queue_imu.front();
    imu_preintegrated_->integrateMeasurement(imu_msg.a_b_ib, imu_msg.w_b_ib, imu_msg.dt);
    gyro_last = imu_msg.w_b_ib;
    queue_imu.pop();
  }

  if (!queue_gt.empty())
  {
    auto gt_msg = queue_gt.front();
    gt_update = gt_msg.header.stamp.toSec();
    // double heading = R2rpy(q_w_curr.toRotationMatrix())(2);
    queue_gt.pop();
  }

  if (!queue_gps.empty())
  {
    auto gps_msg = queue_gps.front();
    queue_gps.pop();
  }

  mutex_1.unlock();
}

void callbackIMU(const sensor_msgs::ImuConstPtr &imu_msg)
{
  mutex_1.lock();
  double dt = 0.01;
  if (std::fabs(last_imu.dt) > 1.0e-6)
    dt = (imu_msg->header.stamp - last_imu.time_stamp).toSec();
  last_imu = ImuDataStamped(imu_msg, dt);
  queue_imu.push(last_imu);
  mutex_1.unlock();
}

void callbackRadarScan(const sensor_msgs::PointCloud2ConstPtr &radar_msg)
{
  mutex_1.lock();
  queue_radar.push(*radar_msg);
  mutex_1.unlock();
}

void callbackGtPose(const nav_msgs::OdometryConstPtr &gt_msg)
{
  mutex_1.lock();
  queue_gt.push(*gt_msg);
  mutex_1.unlock();
}

void callbackGPSPose(const gps_common::GPSFixConstPtr &gps_msg)
{
  mutex_1.lock();
  queue_gps.push(*gps_msg);
  mutex_1.unlock();
}


void config_init(radar_ego_velocity_estimation::RadarEgoVelocityEstimatorConfig &config)
{
  config.min_dist = 0.25;
  config.max_dist = 100;
  config.min_db = 0;
  config.elevation_thresh_deg = 15;
  config.azimuth_thresh_deg = 60;
  config.filter_min_z = -10;
  config.filter_max_z = 10;
  config.doppler_velocity_correction_factor = 1.0;

  config.thresh_zero_velocity = 0.05;
  config.allowed_outlier_percentage = 0.25;
  config.sigma_zero_velocity_x = 0.025;
  config.sigma_zero_velocity_y = 0.025;
  config.sigma_zero_velocity_z = 0.025;

  config.sigma_offset_radar_x = 0.05;
  config.sigma_offset_radar_y = 0.025;
  config.sigma_offset_radar_z = 0.05;

  config.max_sigma_x = 0.2;
  config.max_sigma_y = 0.2;
  config.max_sigma_z = 0.2;
  config.max_r_cond = 1000;

  config.use_cholesky_instead_of_bdcsvd = true;
  config.use_ransac = true;
  config.outlier_prob = 0.4;
  config.success_prob = 0.9999;
  config.N_ransac_points = 3;
  config.inlier_thresh = 0.15;
  config.use_odr = true;
  config.sigma_v_d = 0.125;
  config.min_speed_odr = 4.0;
  config.model_noise_offset_deg = 2.0;
  config.model_noise_scale_deg = 10.0;
}