#include <iostream>
#include <string.h>
#include <random>
#include <unistd.h>
#include <unordered_map>
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
#include <eigen_conversions/eigen_msg.h>
#include <boost/thread/thread.hpp>
#include <mutex>

#include "userdefine.h"
#include "radar_msgs/RadarTarget.h"
#include "gps2local.hpp"

#define RADAR_NUM 6

using namespace std;

POINT_CLOUD_REGISTER_POINT_STRUCT(RadarPointCloudType,
                                  (float, x, x)
                                  (float, y, y)
                                  (float, z, z)
                                  (float, snr_db, snr_db)
                                  (float, noise_db,   noise_db)
                                  (float, v_doppler_mps,   v_doppler_mps)
                                  )


std::vector<Eigen::Matrix<double, 4, 4>> T_enu_radar(6);
Eigen::Matrix<double, 3, 3> Rtrans;
Eigen::Matrix<double, 3, 3> R_init;
Eigen::Vector3d t_w_curr(0, 0, 0);
Eigen::Quaterniond q_w_curr(1, 0, 0, 0);
CooTrans gps2xy;
std::vector<double> radar_update(6);
std::vector<uint32_t> radar_frame(6);
unordered_map<int,double> scanned_map;

std::mutex mutex_1;
std::queue<ImuDataStamped> queue_imu;
std::vector<std::queue<radar_msgs::RadarTarget>> queue_radar(6);
std::queue<nav_msgs::Odometry> queue_odom_twist;
std::queue<sensor_msgs::NavSatFix> queue_odom_pose;
ImuDataStamped last_imu;
ImuDataStamped imu_data;

std::vector< pcl::PointCloud<RadarPointCloudType>> radar_body(6);
pcl::PointCloud<RadarPointCloudType>::Ptr all_radar_body(new pcl::PointCloud<RadarPointCloudType>);

bool pose_initialed = false;
bool twist_initialed = false;

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

void callbackRadarScan(const radar_msgs::RadarTargetConstPtr &radar_msg, int radar_no)
{
  if (radar_no < 0 || radar_no > 5)
  {
    std::cout << "wrong radar no. !!!" << std::endl;
  }
  mutex_1.lock();
  queue_radar[radar_no].push(*radar_msg);
  mutex_1.unlock();
}

void callbackGtTwist(const nav_msgs::OdometryConstPtr &gt_msg)
{
  mutex_1.lock();
  queue_odom_twist.push(*gt_msg);
  mutex_1.unlock();
}

void callbackGtPose(const sensor_msgs::NavSatFixConstPtr &gt_msg)
{
  mutex_1.lock();
  queue_odom_pose.push(*gt_msg);
  mutex_1.unlock();
}

template <typename PointT>
void pointAssociateToSubMap(PointT const *const pi, PointT *const po, Eigen::Matrix<double, 4, 4> Trans)
{
  Eigen::Vector4d point_curr(pi->x, pi->y, pi->z, 1.0);
  Eigen::Vector4d point_w = Trans * point_curr;
  po->x = point_w.x();
  po->y = point_w.y();
  po->z = point_w.z();
}

void RadarPcl2Body(int radar_no)
{
  auto radar_data_msg = queue_radar[radar_no].front();
  sensor_msgs::PointCloud2 radar_ros_msg;
  int num = radar_data_msg.targetNum;
  radar_body[radar_no].clear();
  for (size_t i = 0; i < num; i++)
  {
    RadarPointCloudType p_sel;
    p_sel.x = radar_data_msg.x[i];
    p_sel.y = radar_data_msg.y[i];
    p_sel.z = radar_data_msg.z[i];
    p_sel.snr_db = radar_data_msg.power[i];
    p_sel.v_doppler_mps = radar_data_msg.v[i];
    p_sel.noise_db = radar_data_msg.snr[i];

    pointAssociateToSubMap(&p_sel, &p_sel, T_enu_radar[radar_no]);
    p_sel.range = sqrt(pow(p_sel.x, 2) + pow(p_sel.y, 2) + pow(p_sel.z, 2));
    radar_body[radar_no].push_back(p_sel);
  }

  radar_update[radar_no] = radar_data_msg.header.stamp.toSec();
  radar_frame[radar_no] = radar_data_msg.header.seq;
  

  queue_radar[radar_no].pop();
}

void process_queue()
{
  mutex_1.lock();

  if (!queue_radar[0].empty())
  {
    RadarPcl2Body(0);
  }

  if (!queue_radar[1].empty())
  {
    RadarPcl2Body(1);
  }

  if (!queue_radar[2].empty())
  {
    RadarPcl2Body(2);
  }

  if (!queue_radar[3].empty())
  {
    RadarPcl2Body(3);
  }

  if (!queue_radar[4].empty())
  {
    RadarPcl2Body(4);
  }

  if (!queue_radar[5].empty())
  {
    RadarPcl2Body(5);
  }

  if (!queue_imu.empty())
  {
    queue_imu.pop();
  }

  if (!queue_odom_twist.empty())
  {
    auto gt_msg = queue_odom_twist.front();
    q_w_curr.x() = gt_msg.pose.pose.orientation.x;
    q_w_curr.y() = gt_msg.pose.pose.orientation.y;
    q_w_curr.z() = gt_msg.pose.pose.orientation.z;
    q_w_curr.w() = gt_msg.pose.pose.orientation.w;

    if (!twist_initialed)
    {
      Rtrans = Eigen::Matrix<double, 3, 3>::Identity();
      R_init = q_w_curr.toRotationMatrix();
      twist_initialed = true;
    }
    else
    {
      Rtrans = R_init.transpose() * q_w_curr.toRotationMatrix();
    }
    queue_odom_twist.pop();
  }

  if (!queue_odom_pose.empty())
  {
    auto gt_pose = queue_odom_pose.front();
    Vector3 xyz(0, 0, 0);
    if (!pose_initialed)
    {
      gps2xy.SetECEFOw(gt_pose.latitude * M_PI / 180.0, gt_pose.longitude * M_PI / 180.0, gt_pose.altitude);
      pose_initialed = true;
    }
    else
    {
      gps2xy.getENH(gt_pose.latitude * M_PI / 180.0, gt_pose.longitude * M_PI / 180.0, gt_pose.altitude, xyz(0), xyz(1), xyz(2));
    }

    t_w_curr = xyz;
    t_w_curr = R_init.transpose() * t_w_curr;
    queue_odom_pose.pop();
  }

  mutex_1.unlock();
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "radar_mix_node");
  ros::NodeHandle n("~");
  ros::Publisher pubRadarCloudLocal = n.advertise<sensor_msgs::PointCloud2>("/radar_cloud_local", 100);
  ros::Publisher pubGtOdom = n.advertise<nav_msgs::Odometry>("/gt_odom",100);
  ros::Publisher pubimu = n.advertise<sensor_msgs::Imu>("/imu",100);

  std::string bag_path;
  std::string topic_new;
  std::string topic_imu;
  std::vector<std::string> topic_radar(6);
  std::string topic_gt_twist;
  std::string topic_gt_pose;
  n.getParam("bag_path", bag_path);
  n.getParam("topic_imu", topic_imu);
  n.getParam("topic_radar0", topic_radar[0]);
  n.getParam("topic_radar1", topic_radar[1]);
  n.getParam("topic_radar2", topic_radar[2]);
  n.getParam("topic_radar3", topic_radar[3]);
  n.getParam("topic_radar4", topic_radar[4]);
  n.getParam("topic_radar5", topic_radar[5]);
  n.getParam("topic_gt_twist", topic_gt_twist);
  n.getParam("topic_gt_pose", topic_gt_pose);

  // add calib
  T_enu_radar[0] << 0.6916, -0.7222, -0.009174, 3.65,
      0.7222, 0.6916, -0.00958, 1.02,
      0.01326, 0, 0.9999, 0.68,
      0, 0, 0, 1;

  T_enu_radar[1] << 0.9999, 0.008901, 0.01257, 3.7,
      -0.0089, 1, -0.0001119, -0.135,
      -0.01257, 0, 0.9999, 0.67,
      0, 0, 0, 1;

  T_enu_radar[2] << 0.695, 0.719, 0.01031, 3.65,
      -0.7189, 0.695, -0.01067, -1.04,
      -0.01483, 0, 0.9999, 0.66,
      0, 0, 0, 1;

  T_enu_radar[3] << -0.6785, -0.7328, 0.0522, -1.03,
      0.7306, -0.6805, -0.05622, 0.86,
      0.07672, 0, 0.9971, 0.6,
      0, 0, 0, 1;

  T_enu_radar[4] << -1, -0.004887, -0.005236, -1.165,
      0.004887, -1, 0.0000256, -0.15,
      -0.005236, 0, 1, 0.635,
      0, 0, 0, 1;

  T_enu_radar[5] << -0.7046, 0.7096, -0.0, -1,
      -0.7096, -0.7046, -0.0, -0.88,
      -0.0, 0, 1, 0.6,
      0, 0, 0, 1;

  int radar_new_index = 0;
  double radar_new_timestamp = 0;
  double gtodom_new_timestamp = 0;

  // read bag
  rosbag::Bag source_bag;
  source_bag.open(bag_path, rosbag::bagmode::Read);
  std::vector<std::string> topics;
  topics.push_back(topic_imu);
  topics.push_back(topic_radar[0]);
  topics.push_back(topic_radar[1]);
  topics.push_back(topic_radar[2]);
  topics.push_back(topic_radar[3]);
  topics.push_back(topic_radar[4]);
  topics.push_back(topic_radar[5]);
  topics.push_back(topic_gt_twist);
  topics.push_back(topic_gt_pose);

  rosbag::View view(source_bag, rosbag::TopicQuery(topics));

  for (const rosbag::MessageInstance &m : view)
  {
    const auto topic = m.getTopic();
    radar_new_index = RADAR_NUM;
    if (topic == topic_imu)
    {
      const auto imu_msg_bag = m.instantiate<sensor_msgs::Imu>();
      if (imu_msg_bag != NULL)
      {
        imu_msg_bag->header.frame_id = "/camera_init";
        pubimu.publish(imu_msg_bag);
      }
      topic_new = "imu";
    }

    else if (topic == topic_radar[0])
    {
      const auto radar_scan = m.instantiate<radar_msgs::RadarTarget>();
      if (radar_scan != NULL)
      {
        callbackRadarScan(radar_scan, 0);
        radar_new_index = 0;
        radar_new_timestamp = radar_scan->header.stamp.toSec();
      }
      topic_new = "radar";
    }

    else if (topic == topic_radar[1])
    {
      const auto radar_scan = m.instantiate<radar_msgs::RadarTarget>();
      if (radar_scan != NULL)
      {
        callbackRadarScan(radar_scan, 1);
        radar_new_index = 1;
        radar_new_timestamp = radar_scan->header.stamp.toSec();
      }
      topic_new = "radar";
    }

    else if (topic == topic_radar[2])
    {
      const auto radar_scan = m.instantiate<radar_msgs::RadarTarget>();
      if (radar_scan != NULL)
      {
        callbackRadarScan(radar_scan, 2);
        radar_new_index = 2;
        radar_new_timestamp = radar_scan->header.stamp.toSec();
      }
      topic_new = "radar";
    }

    else if (topic == topic_radar[3])
    {
      const auto radar_scan = m.instantiate<radar_msgs::RadarTarget>();
      if (radar_scan != NULL)
      {
        callbackRadarScan(radar_scan, 3);
        radar_new_index = 3;
        radar_new_timestamp = radar_scan->header.stamp.toSec();
      }
      topic_new = "radar";
    }

    else if (topic == topic_radar[4])
    {
      const auto radar_scan = m.instantiate<radar_msgs::RadarTarget>();
      if (radar_scan != NULL)
      {
        callbackRadarScan(radar_scan, 4);
        radar_new_index = 4;
        radar_new_timestamp = radar_scan->header.stamp.toSec();
      }
      topic_new = "radar";
    }

    else if (topic == topic_radar[5])
    {
      const auto radar_scan = m.instantiate<radar_msgs::RadarTarget>();
      if (radar_scan != NULL)
      {
        callbackRadarScan(radar_scan, 5);
        radar_new_index = 5;
        radar_new_timestamp = radar_scan->header.stamp.toSec();
      }
      topic_new = "radar";
    }

    else if (topic == topic_gt_twist)
    {
      const auto gt_twist = m.instantiate<nav_msgs::Odometry>();

      if (gt_twist != NULL)
      {
        callbackGtTwist(gt_twist);
        gtodom_new_timestamp = gt_twist->header.stamp.toSec();
      }
      topic_new = "gt_twist";
    }

    else if (topic == topic_gt_pose)
    {
      const auto gt_pose = m.instantiate<sensor_msgs::NavSatFix>();
      if (gt_pose != NULL)
      {
        callbackGtPose(gt_pose);
        gtodom_new_timestamp = gt_pose->header.stamp.toSec();
      }
      topic_new = "gt_pose";
    }

    if(scanned_map.count(radar_new_index) && topic_new == "radar")
    {
      for (auto p : scanned_map)
      {
        *all_radar_body += radar_body[p.first];
      }

      sensor_msgs::PointCloud2 RadarCloudLocalRos;
      pcl::toROSMsg(*all_radar_body, RadarCloudLocalRos);
      RadarCloudLocalRos.header.stamp = ros::Time().fromSec(scanned_map.begin()->second);
      RadarCloudLocalRos.header.frame_id = "/camera_init";
      pubRadarCloudLocal.publish(RadarCloudLocalRos);
      all_radar_body->clear();
      scanned_map.clear();
    }

    if(!scanned_map.count(radar_new_index) && topic_new == "radar")
    {
      std::pair<int,double> radar_curr_scan(radar_new_index,radar_new_timestamp);
      scanned_map.insert(radar_curr_scan);
    }

    if(twist_initialed && pose_initialed)
    {
      nav_msgs::Odometry CurrOdom;
      tf::quaternionEigenToMsg(q_w_curr,CurrOdom.pose.pose.orientation);
      tf::pointEigenToMsg(t_w_curr,CurrOdom.pose.pose.position);
      CurrOdom.header.frame_id = "/camera_init";
      CurrOdom.header.stamp = ros::Time().fromSec(gtodom_new_timestamp);
      pubGtOdom.publish(CurrOdom);
    }


    process_queue();

    ros::spinOnce();

  }

}