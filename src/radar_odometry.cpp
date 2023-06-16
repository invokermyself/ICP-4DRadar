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

#define MAX_SEARCH_RADIUS 2.0f

using namespace std;
using PointVector = KD_TREE<ikdTree_PointType>::PointVector;

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
pcl::PointCloud<pcl::PointXYZI>::Ptr RadarCloudMap(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr downSizeFilterMap(new pcl::PointCloud<pcl::PointXYZI>);
KD_TREE<pcl::PointXYZI> ikd_Tree(0.3, 0.6, 0.2);

Eigen::Vector3d t(para_t);
Eigen::Vector4d pos({0, 0, 0, 1});

Eigen::Map<Eigen::Quaterniond> q_last_curr(para_q);
Eigen::Map<Eigen::Vector3d> t_last_curr(para_t);

// Transformation from current frame to world frame
Eigen::Quaterniond q_w_curr(1, 0, 0, 0);
Eigen::Vector3d t_w_curr(0, 0, 0);

Eigen::Matrix<double, 3, 3> Rtrans;
Eigen::Matrix<double, 3, 3> R_enu_radar;
Eigen::MatrixXd currOdom;

bool initialed = false;
std::mutex mutex_1;

std::queue<ImuDataStamped> queue_imu;
std::queue<sensor_msgs::PointCloud2> queue_radar;
std::queue<nav_msgs::Odometry> queue_gt;
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
bool pcl2msgToPcl(const sensor_msgs::PointCloud2 &pcl_msg, pcl::PointCloud<RadarPointCloudType> &scan);
void main_task();
void config_init(radar_ego_velocity_estimation::RadarEgoVelocityEstimatorConfig &config);

int main(int argc, char **argv)
{

  ros::init(argc, argv, "radar_odometry");
  ros::NodeHandle n("~");
  ros::Publisher pubRadarCloudSurround = n.advertise<sensor_msgs::PointCloud2>("/radar_cloud_surround", 100);
  ros::Publisher pubOdomGT = n.advertise<nav_msgs::Odometry>("/lidar_gt", 100);
  radar_ego_velocity_estimation::RadarEgoVelocityEstimatorConfig config;
  config_init(config);
  radar_ego_velocity.configure(config);
  std::size_t order = 1;

  std::string bag_path;
  std::string topic_imu;
  std::string topic_radar_scan;
  std::string topic_lidar_gt;
  n.getParam("bag_path", bag_path);
  n.getParam("topic_imu", topic_imu);
  n.getParam("topic_radar_scan", topic_radar_scan);
  n.getParam("topic_lidar_gt", topic_lidar_gt);

  ros::Rate r(100.0);

  R_enu_radar << 0.000796, -1, 0,
  1, 0.000796, 0,
  0, 0, 1;

  // read bag
  rosbag::Bag source_bag;
  source_bag.open(bag_path, rosbag::bagmode::Read);
  std::vector<std::string> topics;
  topics.push_back(topic_imu);
  topics.push_back(topic_radar_scan);
  topics.push_back(topic_lidar_gt);

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
      const auto radar_scan = m.instantiate<sensor_msgs::PointCloud2>();
      if (radar_scan != NULL)
        callbackRadarScan(radar_scan);
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

    ros::spinOnce();

    main_task();

    sensor_msgs::PointCloud2 RadarCloudSurround;
    pcl::toROSMsg(*downSizeFilterMap, RadarCloudSurround);
    RadarCloudSurround.header.stamp = ros::Time().now();
    RadarCloudSurround.header.frame_id = "/camera_init";
    pubRadarCloudSurround.publish(RadarCloudSurround);
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
    Vector3 v_r, sigma_v_r;
    sensor_msgs::PointCloud2 inlier_radar_scan;
    radar_ego_velocity.estimate(radar_data_msg, v_r, sigma_v_r, inlier_radar_scan);
    pcl::PointCloud<ColoRadarPointCloudType> scan_ColoRadar;
    pcl::PCLPointCloud2 pcl_pc2;
    pcl_conversions::toPCL(inlier_radar_scan, pcl_pc2);
    pcl::fromPCLPointCloud2(pcl_pc2, scan_ColoRadar);
    int point_num = scan_ColoRadar.size();
    for (size_t i = 0; i < point_num; i++)
    {
      pcl::PointXYZI p_sel;
      p_sel.x = scan_ColoRadar.points[i].x;
      p_sel.y = scan_ColoRadar.points[i].y;
      p_sel.z = scan_ColoRadar.points[i].z;
      p_sel.intensity = scan_ColoRadar.points[i].intensity;
      src->push_back(p_sel);
    }

    if (!initialed)
    {
      pcl::copyPointCloud(*src, *tar);
      initialed = true;
    }
    radar_update = radar_data_msg.header.stamp.toSec();

    queue_radar.pop();

    // GICP

    fast_gicp::FastGICPSingleThread<pcl::PointXYZI, pcl::PointXYZI> fgicp_st;
    fgicp_st.clearTarget();
    fgicp_st.clearSource();
    fgicp_st.setInputTarget(tar);
    fgicp_st.setInputSource(src);
    fgicp_st.align(*Final);
    auto t2 = std::chrono::high_resolution_clock::now();
    double score = fgicp_st.getFitnessScore();

    std::cout << "has converged:" << fgicp_st.hasConverged() << " score: " << fgicp_st.getFitnessScore() << std::endl;
    std::cout << fgicp_st.getFinalTransformation() << std::endl;

    Eigen::Matrix<double, 4, 4> icp_result = fgicp_st.getFinalTransformation().cast<double>();

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
    Rtrans = R_enu_radar*q_w_curr.toRotationMatrix();
    gt_update = gt_msg.header.stamp.toSec();
    double time_diff = gt_update - radar_update;
    if (time_diff < 0.1)
    {
      size_t PointNum = src->size();
      PointType p_sel;
      for (size_t i = 0; i < PointNum; i++)
      {
        pointAssociateToMap(&src->points[i], &p_sel);
        RadarCloudMap->push_back(p_sel);
      }
    }
    pcl::VoxelGrid<pcl::PointXYZI> sor;
    sor.setInputCloud(RadarCloudMap);
    sor.setLeafSize(0.05f, 0.05f, 0.05f);
    sor.filter(*downSizeFilterMap);
    queue_gt.pop();

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

bool pcl2msgToPcl(const sensor_msgs::PointCloud2 &pcl_msg, pcl::PointCloud<RadarPointCloudType> &scan)
{
  // TODO: add support for ti_mmwave_rospkg clound type

  std::set<std::string> fields;
  std::string fields_str = "";

  for (const auto &field : pcl_msg.fields)
  {
    fields.emplace(field.name);
    fields_str += field.name + ", ";
  }

  if (fields.find("x") != fields.end() && fields.find("y") != fields.end() && fields.find("z") != fields.end() &&
      fields.find("snr_db") != fields.end() && fields.find("noise_db") != fields.end() &&
      fields.find("v_doppler_mps") != fields.end())
  {
    ROS_INFO_ONCE("[pcl2msgToPcl]: Detected rio pcl format!");
    pcl::PCLPointCloud2 pcl_pc2;
    pcl_conversions::toPCL(pcl_msg, pcl_pc2);
    pcl::fromPCLPointCloud2(pcl_pc2, scan);

    // fix format
    for (auto &p : scan)
      p.range = p.getVector3fMap().norm();

    return true;
  }
  else if (fields.find("x") != fields.end() && fields.find("y") != fields.end() && fields.find("z") != fields.end() &&
           fields.find("intensity") != fields.end() && fields.find("velocity") != fields.end())
  {
    ROS_INFO_ONCE("[pcl2msgToPcl]: Detected ti_mmwave_rospkg pcl format!");

    pcl::PointCloud<mmWaveCloudType> scan_mmwave;
    pcl::PCLPointCloud2 pcl_pc2;
    pcl_conversions::toPCL(pcl_msg, pcl_pc2);
    pcl::fromPCLPointCloud2(pcl_pc2, scan_mmwave);

    scan.clear();
    for (const auto &p : scan_mmwave)
    {
      RadarPointCloudType p_;
      p_.x = -p.y;
      p_.y = p.x;
      p_.z = p.z;
      p_.snr_db = p.intensity;
      p_.v_doppler_mps = p.velocity;
      p_.range = p.getVector3fMap().norm();
      p_.noise_db = -1.;
      scan.push_back(p_);
    }
    return true;
  }
  else if (fields.find("x") != fields.end() && fields.find("y") != fields.end() && fields.find("z") != fields.end() &&
           fields.find("Doppler") != fields.end() && fields.find("Range") != fields.end() && fields.find("Power") != fields.end() && fields.find("Alpha") != fields.end() && fields.find("Beta") != fields.end())
  {
    ROS_INFO_ONCE("[pcl2msgToPcl]: Detected oculii pcl format!");

    pcl::PointCloud<OculiiPointCloudType> scan_oculii;
    pcl::PCLPointCloud2 pcl_pc2;
    pcl_conversions::toPCL(pcl_msg, pcl_pc2);
    pcl::fromPCLPointCloud2(pcl_pc2, scan_oculii);

    scan.clear();
    for (const auto &p : scan_oculii)
    {
      RadarPointCloudType p_;
      p_.x = p.x;
      p_.y = p.y;
      p_.z = p.z;
      p_.snr_db = p.Power;
      p_.v_doppler_mps = p.Doppler;
      p_.range = p.Range;
      p_.noise_db = -1.;
      scan.push_back(p_);
    }
    return true;
  }

  else if (fields.find("x") != fields.end() && fields.find("y") != fields.end() && fields.find("z") != fields.end() &&
           fields.find("intensity") != fields.end() && fields.find("range") != fields.end() && fields.find("doppler") != fields.end())
  {
    ROS_INFO_ONCE("[pcl2msgToPcl]: Detected ColoRadar pcl format!");

    pcl::PointCloud<ColoRadarPointCloudType> scan_ColoRadar;
    pcl::PCLPointCloud2 pcl_pc2;
    pcl_conversions::toPCL(pcl_msg, pcl_pc2);
    pcl::fromPCLPointCloud2(pcl_pc2, scan_ColoRadar);

    scan.clear();
    for (const auto &p : scan_ColoRadar)
    {
      RadarPointCloudType p_;
      p_.x = p.x;
      p_.y = p.y;
      p_.z = p.z;
      p_.snr_db = p.intensity;
      p_.v_doppler_mps = p.doppler;
      p_.range = p.range;
      p_.noise_db = -1.;
      scan.push_back(p_);
    }
    return true;
  }

  else
  {
    ROS_ERROR_STREAM(
        "[pcl2msgToPcl]: Unsupported point cloud with fields: " << fields_str.substr(0, fields_str.size() - 2));
    return false;
  }
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