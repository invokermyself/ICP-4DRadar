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
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/NavSatFix.h>
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
#include "gps2local.hpp"
#include "ikd_Tree.h"
#include "radar_ego_velocity_estimator/radar_ego_velocity_estimator.h"
#include "fast_gicp/gicp/fast_gicp.hpp"
#include "fast_gicp/gicp/fast_gicp_st.hpp"
#include "radar_msgs/RadarTarget.h"

#define MAX_SEARCH_RADIUS 2.0f

using namespace std;

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

pcl::PointCloud<pcl::PointXYZI>::Ptr src(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr tar(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr RadarCloudMap(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr RadarCloudLocal(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr downSizeFilterMap(new pcl::PointCloud<pcl::PointXYZI>);

// Transformation from current frame to world frame
Eigen::Quaterniond q_w_curr(1, 0, 0, 0);
Eigen::Vector3d t_w_curr(0, 0, 0);
Eigen::Matrix<double, 3, 3> Rtrans;

// init Transformation
std::mutex mutex_1;

std::queue<ImuDataStamped> queue_imu;
std::queue<sensor_msgs::PointCloud2> queue_radar;
std::queue<nav_msgs::Odometry> queue_odom_pose;
ImuDataStamped last_imu;
ImuDataStamped imu_data;
reve::RadarEgoVelocityEstimator radar_ego_velocity;

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
  Eigen::Vector4d point_w = Trans * point_curr;
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

void process_data()
{
  mutex_1.lock();

  if (!queue_imu.empty())
  {
    queue_imu.pop();
  }



  mutex_1.unlock();
}

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

int main(int argc, char **argv)
{

  ros::init(argc, argv, "radar_odometry");
  ros::NodeHandle n("~");

  ros::Subscriber subRadarScan = n.subscribe<sensor_msgs::PointCloud2>("/radar_cloud_local", 10, callbackRadarScan);
  ros::Subscriber subImu = n.subscribe<sensor_msgs::Imu>("/imu", 100, callbackIMU);
  ros::Subscriber subOdom = n.subscribe<nav_msgs::Odometry>("/gt_odom", 100, callbackGtPose);

  ros::Publisher pubRadarCloudSurround = n.advertise<sensor_msgs::PointCloud2>("/radar_cloud_surround", 100);

  radar_ego_velocity_estimation::RadarEgoVelocityEstimatorConfig config;
  config_init(config);
  radar_ego_velocity.configure(config);

  process_data();

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
  queue_odom_pose.push(*gt_msg);
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

