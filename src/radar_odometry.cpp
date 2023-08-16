#include <iostream>
#include <string.h>
#include <random>
#include <unistd.h>
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
#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <boost/thread/thread.hpp>
#include <mutex>

#include <pcl/pcl_macros.h>
#include "userdefine.h"
#include "radarFactor.hpp"
#include "tic_toc.h"
#include "ikd_Tree.h"
#include "radar_ego_velocity_estimator/radar_ego_velocity_estimator.h"
#include "fast_gicp/gicp/fast_gicp.hpp"
#include "fast_gicp/gicp/fast_gicp_st.hpp"

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

#define MAX_SEARCH_RADIUS 2.0f
#define RADAR_RADIUS 80

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
NonlinearFactorGraph *graph = new NonlinearFactorGraph();
KD_TREE<pcl::PointXYZI> ikd_Tree(0.3, 0.6, 0.5);

Eigen::Vector3d t(para_t);
Eigen::Vector4d pos({0, 0, 0, 1});

Eigen::Map<Eigen::Quaterniond> q_last_curr(para_q);
Eigen::Map<Eigen::Vector3d> t_last_curr(para_t);

// Transformation from current frame to world frame
Eigen::Quaterniond q_w_curr(1, 0, 0, 0);
Eigen::Vector3d t_w_curr(0, 0, 0);
Vector3 v_r, sigma_v_r;

Eigen::Matrix<double, 3, 3> Rtrans;
Eigen::Matrix<double, 3, 3> R_enu_radar;
Eigen::MatrixXd currOdom;
nav_msgs::Odometry RadarOdom;
nav_msgs::Path RadarPath;

bool initialed = false;
std::mutex mutex_1;

std::queue<ImuDataStamped> queue_imu;
std::queue<sensor_msgs::PointCloud2> queue_radar;
std::queue<nav_msgs::Odometry> queue_gt;
std::queue<gps_common::GPSFix> queue_gps;
ImuDataStamped last_imu;
ImuDataStamped imu_data;
reve::RadarEgoVelocityEstimator radar_ego_velocity;

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

int main(int argc, char **argv)
{

  ros::init(argc, argv, "radar_odometry");
  ros::NodeHandle n("~");
  ros::Publisher pubRadarCloudSurround = n.advertise<sensor_msgs::PointCloud2>("/radar_cloud_surround", 100);
  ros::Publisher pubRadarSubMap = n.advertise<sensor_msgs::PointCloud2>("/radar_submap", 100);
  ros::Publisher pubRadarPath = n.advertise<nav_msgs::Path>("/radar_path", 5);
  ros::Publisher pubRadarVel = n.advertise<geometry_msgs::Twist>("/radar_vel", 5);
  ros::Publisher pubOdomGT = n.advertise<nav_msgs::Odometry>("/lidar_gt", 100);
  radar_ego_velocity_estimation::RadarEgoVelocityEstimatorConfig config;
  config_init(config);
  radar_ego_velocity.configure(config);
  std::size_t order = 1;

  std::string bag_path;
  std::string topic_imu;
  std::string topic_radar_scan;
  std::string topic_lidar_gt;
  std::string topic_gps_gt;
  n.getParam("bag_path", bag_path);
  n.getParam("topic_imu", topic_imu);
  n.getParam("topic_radar_scan", topic_radar_scan);
  n.getParam("topic_lidar_gt", topic_lidar_gt);
  n.getParam("topic_gps_gt", topic_gps_gt);

  ros::Rate r(100.0);

  R_enu_radar << 0.000796, -1, 0,
      1, 0.000796, 0,
      0, 0, 1;

  currOdom = Eigen::Matrix<double, 4, 4>::Identity();

  // read bag
  rosbag::Bag source_bag;
  source_bag.open(bag_path, rosbag::bagmode::Read);
  std::vector<std::string> topics;
  topics.push_back(topic_imu);
  topics.push_back(topic_radar_scan);
  topics.push_back(topic_lidar_gt);
  topics.push_back(topic_gps_gt);

  rosbag::View view(source_bag, rosbag::TopicQuery(topics));

  for (const rosbag::MessageInstance &m : view)
  {
    const auto topic = m.getTopic();
    if (topic == topic_imu)
    {
      const auto imu_msg_bag = m.instantiate<sensor_msgs::Imu>();
      if (imu_msg_bag != NULL)
      {
        callbackIMU(imu_msg_bag);
      }
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

    ros::spinOnce();

    main_task();

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

    SubMap->clear();
  }
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
    // sensor_msgs::PointCloud2 inlier_radar_scan;
    // radar_ego_velocity.estimate(radar_data_msg, v_r, sigma_v_r, inlier_radar_scan);
    pcl::PointCloud<OculiiPointCloudType> scan_OculliRadar;
    pcl::PCLPointCloud2 pcl_pc2;
    pcl_conversions::toPCL(radar_data_msg, pcl_pc2);
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
      ikd_Tree.Build(src->points);
      ikd_Tree.set_downsample_param(0.5);
      initialed = true;
    }
    radar_update = radar_data_msg.header.stamp.toSec();

    queue_radar.pop();

    // t_w_curr = t_w_curr + q_w_curr * t_last_curr;
    // q_w_curr = q_w_curr * q_last_curr;
  }

  if (!queue_imu.empty())
  {
    queue_imu.pop();
  }

  if (!queue_gt.empty())
  {
    auto gt_msg = queue_gt.front();
    t_w_curr(0) = gt_msg.pose.pose.position.x;
    t_w_curr(1) = gt_msg.pose.pose.position.y;
    t_w_curr(2) = gt_msg.pose.pose.position.z;
    q_w_curr.x() = gt_msg.pose.pose.orientation.x;
    q_w_curr.y() = gt_msg.pose.pose.orientation.y;
    q_w_curr.z() = gt_msg.pose.pose.orientation.z;
    q_w_curr.w() = gt_msg.pose.pose.orientation.w;
    // Rtrans = q_w_curr.toRotationMatrix() * R_enu_radar;
    // currOdom.topLeftCorner<3,3>() = Rtrans;
    // currOdom.topRightCorner<3,1>() = t_w_curr;
    gt_update = gt_msg.header.stamp.toSec();
    double time_diff = gt_update - radar_update;
    double heading = R2rpy(q_w_curr.toRotationMatrix())(2);
    if (time_diff < 0.1)
    {
      size_t PointNum = src->size();
      PointType p_sel;
      for (size_t i = 0; i < PointNum; i++)
      {
        pointAssociateToMap(&src->points[i], &p_sel);
        RadarCloudMap->push_back(p_sel);
        scan_map->push_back(p_sel);
      }
      ikd_Tree.Add_Points(scan_map->points, false);

      pcl::PointXYZI p_now;
      p_now.x = t_w_curr(0);
      p_now.y = t_w_curr(1);
      p_now.z = t_w_curr(2);
      ikd_Tree.Sector_Search(p_now, RADAR_RADIUS, heading, SubMap->points);

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

      Eigen::Matrix<double, 4, 4> icp_result = fgicp_st.getFinalTransformation().cast<double>();
      currOdom = icp_result * currOdom;
      Rtrans = currOdom.topLeftCorner<3, 3>();
      t_w_curr = currOdom.topRightCorner<3, 1>();

      RadarOdom.pose.pose.position.x = currOdom(0, 3);
      RadarOdom.pose.pose.position.y = currOdom(1, 3);
      RadarOdom.pose.pose.position.z = currOdom(2, 3);
      Eigen::Quaterniond q_tmp(currOdom.topLeftCorner<3, 3>());
      RadarOdom.pose.pose.orientation.w = q_tmp.w();
      RadarOdom.pose.pose.orientation.x = q_tmp.x();
      RadarOdom.pose.pose.orientation.y = q_tmp.y();
      RadarOdom.pose.pose.orientation.z = q_tmp.z();

      scan_map->clear();
    }

    pcl::VoxelGrid<pcl::PointXYZI> sor;
    sor.setInputCloud(RadarCloudMap);
    sor.setLeafSize(0.5f, 0.5f, 0.5f);
    sor.filter(*downSizeFilterMap);
    queue_gt.pop();
  }

  if (!queue_gps.empty())
  {
    auto gps_msg = queue_gps.front();
    gt_update = gps_msg.header.stamp.toSec();
    double time_diff = gt_update - radar_update;
    if (time_diff < 0.01)
    {
      // size_t PointNum = src->size();
      // PointType p_sel;
      // for (size_t i = 0; i < PointNum; i++)
      // {
      //   pointAssociateToMap(&src->points[i], &p_sel);
      //   RadarCloudMap->push_back(p_sel);
      //   scan_map->push_back(p_sel);
      // }
      // ikd_Tree.Add_Points(scan_map->points, false);


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

      Eigen::Matrix<double, 4, 4> icp_result = fgicp_st.getFinalTransformation().cast<double>();
      currOdom =  currOdom * icp_result;
      Rtrans = currOdom.topLeftCorner<3, 3>();
      t_w_curr = currOdom.topRightCorner<3, 1>();

      RadarOdom.pose.pose.position.x = currOdom(0, 3);
      RadarOdom.pose.pose.position.y = currOdom(1, 3);
      RadarOdom.pose.pose.position.z = currOdom(2, 3);
      Eigen::Quaterniond q_tmp(currOdom.topLeftCorner<3, 3>());
      RadarOdom.pose.pose.orientation.w = q_tmp.w();
      RadarOdom.pose.pose.orientation.x = q_tmp.x();
      RadarOdom.pose.pose.orientation.y = q_tmp.y();
      RadarOdom.pose.pose.orientation.z = q_tmp.z();

      scan_map->clear();
    }

    // pcl::VoxelGrid<pcl::PointXYZI> sor;
    // sor.setInputCloud(RadarCloudMap);
    // sor.setLeafSize(0.5f, 0.5f, 0.5f);
    // sor.filter(*downSizeFilterMap);

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
  config.elevation_thresh_deg = 60;
  config.azimuth_thresh_deg = 60;
  config.filter_min_z = -3;
  config.filter_max_z = 3;
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