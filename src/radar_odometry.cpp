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
#include <pcl/registration/gicp.h>
#include <pcl/registration/ndt.h>
#include <pcl/pcl_macros.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/point_cloud_conversion.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/NavSatFix.h>
#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <eigen_conversions/eigen_msg.h>
#include <boost/thread/thread.hpp>
#include <mutex>

#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/RadarFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/PreintegrationBase.h>
#include <gtsam/slam/dataset.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/dataset.h>
#include <gtsam/nonlinear/LevenbergMarquardtParams.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/ISAM2.h>

#include "userdefine.h"
#include "radarFactor.hpp"
#include "gps2local.hpp"
#include "ikd_Tree.h"
#include "radar_ego_velocity_estimator/radar_ego_velocity_estimator.h"
#include "fast_gicp/gicp/fast_gicp.hpp"
#include "fast_gicp/gicp/fast_gicp_st.hpp"
#include <fast_gicp/gicp/fast_apdgicp.hpp>
#include "radar_msgs/RadarTarget.h"
#include "aloam_velodyne/tic_toc.h"
#include "aloam_velodyne/common.h"
#include "scancontext/Scancontext.h"
#include "KRFliter.h"

#define MAX_SEARCH_RADIUS 2.0f
#define USE_COMBINED
// #define USE_GICP
#define USE_NDT
#define LOOP_USE_ICP
#define RADAR_NUM 6
#define G 9.79

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

pcl::PointCloud<pcl::PointXYZI>::Ptr src(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr tar(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr RadarCloudMap(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr RadarCloudLocal(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr downSizeFilterMap(new pcl::PointCloud<pcl::PointXYZI>);

// Transformation from current frame to world frame
Eigen::Quaterniond q_w_curr(1, 0, 0, 0);
Eigen::Vector3d t_w_curr(0, 0, 0);
Vector3 v_r, sigma_v_r;
Vector3 gt_vel;
Vector3 gyro_last_ned;
Vector3 accel_last_ned;
Vector3 gyro_last_body;
Eigen::Vector3d vel_Radar;
Eigen::Vector3d vel_Radar_sigma;
Vector3 vel_Radar_last(0,0,0);
// Vector3 lbr (0.0901, -0.1449, 0.0249); //colo
#ifdef SONGHONG
Vector3 lbr (3.7759, -0.2495, -0.044); //songhong
#endif

// 初始化值
// vel esti
int radar_scan_count = 0;
int vel_correction_count = 0;
int icp_correction_count = 0;
int fault_detect_cnt = 0;
std::queue<Pose6D> odomBuf;
std::queue<pcl::PointCloud<PointType>> fullResBuf;

std::vector<Eigen::Matrix<double, 4, 4>> T_body_radar(6);
std::vector<Vector3d> l_body_radar(6);
std::vector<Vector3d> l_ned_radar(6);
Eigen::Matrix<double, 6, 1> gps = Eigen::Matrix<double, 6, 1>::Zero();
Eigen::Matrix<double, 3, 3> Rtrans;
Eigen::Matrix<double, 3, 3> R_init;
Eigen::Matrix<double, 4, 4> T_ned_body;
Eigen::Matrix<double, 3, 3> R_ned_body;
CooTrans gps2xy;
Eigen::MatrixXd currOdom;
Eigen::MatrixXd currOdom_icp;
Eigen::MatrixXd currOdom_gicp;
Eigen::MatrixXd currOdom_ndt;
Eigen::MatrixXd currOdom_apdgicp;
nav_msgs::Odometry RadarOdom;
nav_msgs::Path RadarPath;


//pose graph
NonlinearFactorGraph *graph = new NonlinearFactorGraph();
gtsam::ISAM2 *isam;
gtsam::Values isamCurrentEstimate;
Values initial_values;

//loop detect
int odom_node = 0;
int loop_currnode = 0;
std::mutex mKF;
std::mutex mBuf;
SCManager scManager;
std::queue<std::pair<int, int> > scLoopICPBuf;
pcl::VoxelGrid<PointType> downSizeFilterScan;
pcl::VoxelGrid<PointType> downSizeFilterICP;
pcl::VoxelGrid<PointType> downSizeFilterScancontext;
pcl::VoxelGrid<PointType> downSizeFilterMapPGO;
std::vector<pcl::PointCloud<PointType>::Ptr> keyframeLaserClouds;
std::vector<std::pair<int,Pose6D>> keyframePoses;
std::vector<std::pair<int,Pose6D>> keyframePosesUpdated;
Pose6D odom_pose_prev {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; // init 
Pose6D odom_pose_curr {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; // init pose is zero 
double keyframeMeterGap = 2.0;
double movementAccumulation = 1000000.0; // large value means must add the first given frame.
double scDistThres;
double output_time =0;
double odom_length=0;
bool isNowKeyFrame = false;
std::map<int, int> key_pair;


//rosbag
bool vel_initialed = false;
bool pcl_initialed = false;
bool imu_initialed = false;
bool pose_initialed = false;
bool twist_initialed = false;
std::mutex mutex_rawdata;
std::mutex mutex_imu;
std::mutex mutex_radar;
std::mutex mutex_gt;
std::queue<ImuDataStamped> queue_imu;
std::vector<std::queue<radar_msgs::RadarTarget>> queue_radar(6);
std::vector<pcl::PointCloud<OculiiPointCloudType>> radar_body(6);
std::vector<Eigen::Vector3d> radar_vel(6);
std::vector<Eigen::Vector3d> radar_vel_sigma(6);
std::queue<sensor_msgs::NavSatFix> queue_odom_pose;
std::queue<nav_msgs::Odometry> queue_odom_twist;
ImuDataStamped last_imu;
ImuDataStamped imu_data;
unordered_map<int,double> scanned_map;

std::queue<ImuDataStamped> imu_ned;
std::queue<nav_msgs::Odometry> gt_odom;
std::queue<pair<int,Eigen::Vector3d>> radar_ego_vel;
std::queue<pair<int,Eigen::Vector3d>> radar_ego_vel_sigma;
std::queue<pcl::PointCloud<OculiiPointCloudType>> all_radar_scan;
std::queue<pcl::PointCloud<OculiiPointCloudType>> fov_radar_scan;


PreintegrationType *imu_preintegrated_;
imuBias::ConstantBias prev_bias;
Values result;
NavState prev_state;
NavState prop_state;
std::mutex mtxPosegraph;
std::mutex mtxRecentPose;

// param
double bag_start;
double bag_duration;
double fov_radar_new_timestamp = 0;
int vel_correction_freq;
bool save_keyframe;
std::string bag_path;
std::string bag_name;
std::string output_path;
std::string topic_new;
std::string topic_imu;

std::vector<std::string> topic_radar(6);
std::string topic_gt_twist;
std::string topic_gt_pose;
std::string radar_vel_mode;
std::string radar_odom_mode;
std::string sensor_type;

ros::Publisher pubRadarCloudLocal;
ros::Publisher pubGtOdom;
ros::Publisher pubGtPath;
ros::Publisher pubRadarCloudSurround;
ros::Publisher pubRadarSubMap;
ros::Publisher pubRadarPath;
ros::Publisher pubRadarVel;
ros::Publisher pubOdomAftPGO;
ros::Publisher pubPathAftPGO;
ros::Publisher pubPathAftPGO_woLoop;

KalmanInfo KF_Fliter;
FILE *fp;


void pointAssociateToMap(PointType const *const pi, PointType *const po)
{
  Eigen::Vector3d point_curr(pi->x, pi->y, pi->z);
  Eigen::Vector3d point_w = Rtrans * point_curr + t_w_curr;
  po->x = point_w.x();
  po->y = point_w.y();
  po->z = point_w.z();
  po->intensity = pi->intensity;
}

template <typename PointT>
void pointAssociateToSubMap(PointT const *const pi, PointT *const po, Eigen::Matrix<double, 4, 4> Trans)
{
  Eigen::Vector4d point_curr(pi->x, pi->y, pi->z, 1.0);
  Eigen::Vector4d point_w = Trans * point_curr;
  po->x = point_w.x();
  po->y = point_w.y();
  po->z = point_w.z();
  po->Range = sqrt(pow(po->x, 2) + pow(po->y, 2) + pow(po->z, 2));
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

template <typename PointT> bool
pcl2msgToPcl(const PointT& pcl_msg, pcl::PointCloud<OculiiPointCloudType>& scan)
{
  // TODO: add support for ti_mmwave_rospkg clound type

  std::set<std::string> fields;
  std::string fields_str = "";

  for (const auto& field : pcl_msg.fields)
  {
    fields.emplace(field.name);
    fields_str += field.name + ", ";
  }

  if (fields.find("x") != fields.end() && fields.find("y") != fields.end() && fields.find("z") != fields.end() &&
      fields.find("snr_db") != fields.end() && fields.find("noise_db") != fields.end() &&
      fields.find("v_doppler_mps") != fields.end())
  {
    ROS_INFO_ONCE("[pcl2msgToPcl]: Detected rio pcl format!");
    pcl::PointCloud<RadarPointCloudType> scan_rio;
    pcl::PCLPointCloud2 pcl_pc2;
    pcl_conversions::toPCL(pcl_msg, pcl_pc2);
    pcl::fromPCLPointCloud2(pcl_pc2, scan_rio);
    scan.clear();
    for (const auto& p : scan_rio)
    {
      OculiiPointCloudType p_;
      p_.x             = p.x;
      p_.y             = p.y;
      p_.z             = p.z;
      p_.Alpha        = atan2(p.y,p.x);
      p_.Beta         = atan2(std::sqrt(p.x * p.x + p.y * p.y), p.z) - M_PI_2; 
      p_.Power        = p.snr_db;
      p_.Doppler = p.v_doppler_mps;
      p_.Range         = p.getVector3fMap().norm();
      scan.push_back(p_);
    }

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
    for (const auto& p : scan_mmwave)
    {
      OculiiPointCloudType p_;
      p_.x             = -p.y;
      p_.y             = p.x;
      p_.z             = p.z;
      p_.Alpha        = atan2(p.y,p.x);
      p_.Beta         = atan2(std::sqrt(p.x * p.x + p.y * p.y), p.z) - M_PI_2; 
      p_.Power        = p.intensity;
      p_.Doppler = p.velocity;
      p_.Range         = p.getVector3fMap().norm();
      scan.push_back(p_);
    }
    return true;
  }
  else if (fields.find("x") != fields.end() && fields.find("y") != fields.end() && fields.find("z") != fields.end() &&
          fields.find("Doppler") != fields.end() && fields.find("Range") != fields.end() && fields.find("Power") != fields.end()
          && fields.find("Alpha") != fields.end() && fields.find("Beta") != fields.end())
  {
    ROS_INFO_ONCE("[pcl2msgToPcl]: Detected oculii pcl format!");

    scan.clear();
    pcl::PCLPointCloud2 pcl_pc2;
    pcl_conversions::toPCL(pcl_msg, pcl_pc2);
    pcl::fromPCLPointCloud2(pcl_pc2, scan);

    // fix format
    for (auto& p : scan) p.Range = p.getVector3fMap().norm();

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
    for (const auto& p : scan_ColoRadar)
    {
      OculiiPointCloudType p_;
      p_.x             = p.x;
      p_.y             = p.y;
      p_.z             = p.z;
      p_.Alpha        = atan2(p.y,p.x);
      p_.Beta         = atan2(std::sqrt(p.x * p.x + p.y * p.y), p.z) - M_PI_2; 
      p_.Power        = p.intensity;
      p_.Doppler = p.doppler;
      p_.Range         = p.range;
      scan.push_back(p_);
    }
    return true;

  }
  // else if (fields.find("x") != fields.end() && fields.find("y") != fields.end() && fields.find("z") != fields.end() && fields.find("v") != fields.end()
  //         && fields.find("snr") != fields.end() && fields.find("azimuth") != fields.end() && fields.find("elevation") != fields.end())
  // {
  //   ROS_INFO_ONCE("[pcl2msgToPcl]: Detected JiheRadar pcl format!");

  //   scan.clear();
  //   for (const auto& p : pcl_msg)
  //   {
  //     OculiiPointCloudType p_;
  //     p_.x             = p.x;
  //     p_.y             = p.y;
  //     p_.z             = p.z;
  //     p_.Alpha        = p.azimuth;
  //     p_.Beta         = p.elevation; 
  //     p_.Power        = p.snr;
  //     p_.Doppler = p.v;
  //     p_.Range         = sqrt(pow(p.x,2) + pow(p.y,2) + pow(p.z,2));
  //     scan.push_back(p_);
  //   }
  //   return true;
  // }
  


  else
  {
    ROS_ERROR_STREAM(
        "[pcl2msgToPcl]: Unsupported point cloud with fields: " << fields_str.substr(0, fields_str.size() - 2));
    return false;
  }
}

bool cmp1( pair<int,vector<double>>a, pair<int,vector<double>>b)
{
  return a.second[0] < b.second[0];
} 

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
  rtval.time = fov_radar_new_timestamp;
  return rtval;
} // GTSAMPose3toPose6D

static double degree2rad(double theta)
{
    return theta * M_PI / 180.0;
}

double transDiff(const Pose6D& _p1, const Pose6D& _p2)
{
    return sqrt( (_p1.x - _p2.x)*(_p1.x - _p2.x) + (_p1.y - _p2.y)*(_p1.y - _p2.y) + (_p1.z - _p2.z)*(_p1.z - _p2.z) );
} // transDiff

pcl::PointCloud<PointType>::Ptr local2global(const pcl::PointCloud<PointType>::Ptr &cloudIn, const Pose6D& tf)
{
    pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

    int cloudSize = cloudIn->size();
    cloudOut->resize(cloudSize);

    // Eigen::Affine3f transCur = pcl::getTransformation(tf.x, tf.y, tf.z, tf.roll, tf.pitch, tf.yaw);
    Eigen::Affine3f transCur = pcl::getTransformation(tf.x, -tf.y, -tf.z, tf.roll, -tf.pitch, -tf.yaw);

    int numberOfCores = 12;
#pragma omp parallel for num_threads(numberOfCores)
    for (int i = 0; i < cloudSize; ++i)
    {
        const auto &pointFrom = cloudIn->points[i];
        cloudOut->points[i].x = transCur(0,0) * pointFrom.x + transCur(0,1) * pointFrom.y + transCur(0,2) * pointFrom.z + transCur(0,3);
        cloudOut->points[i].y = -(transCur(1,0) * pointFrom.x + transCur(1,1) * pointFrom.y + transCur(1,2) * pointFrom.z + transCur(1,3));
        cloudOut->points[i].z = -(transCur(2,0) * pointFrom.x + transCur(2,1) * pointFrom.y + transCur(2,2) * pointFrom.z + transCur(2,3));
        cloudOut->points[i].intensity = pointFrom.intensity;
    }

    return cloudOut;
}

static OculiiPointCloudType toOculiiPointCloudType(const Vector11& item)
{
  OculiiPointCloudType point;
  point.x             = item[x_r];
  point.y             = item[y_r];
  point.z             = item[z_r];
  point.Doppler = -item[v_d];
  point.Power        = item[peak_db];
  return point;
}
// v*cos(beta) = A*cos(arfa+b)
template <typename PointT> void
fitSineRansac(const pcl::PointCloud<PointT> &cloud_in,
					 double &A_best,
					 double &b_best,
					 int iterations = 200,
					 double sigma = 0.5)
{
	uint16_t num = cloud_in.points.size();
	double bestScore = 0.0;

	for (int i = 0; i < iterations; i++)
	{
		std::random_device rd;
		std::default_random_engine eng(rd());
		std::uniform_int_distribution<int> distr(0, num);
		int i1 = distr(eng);
		int i2 = distr(eng);

		const PointT p1 = cloud_in.points[i1];
		const PointT p2 = cloud_in.points[i2];

		// form the model by two points
		double k = (p1.Doppler * cos(DEG2RAD(p1.Beta))) / (p2.Doppler * cos(DEG2RAD(p2.Beta)));
		double b = atan((cos(DEG2RAD(p1.Alpha)) - k * cos(DEG2RAD(p2.Alpha))) / (sin(DEG2RAD(p1.Alpha)) - k * sin(DEG2RAD(p2.Alpha))));
		double A = cos(DEG2RAD(p1.Beta)) * p1.Doppler / cos((DEG2RAD(p1.Alpha)) + b);
		double score = 0;

		for (int j = 0; j < num; j++)
		{
			PointT pj = cloud_in.points[j];
			double delta = (cos(DEG2RAD(pj.Beta)) * pj.Doppler) - (A * cos(DEG2RAD(pj.Alpha) + b));
			if (fabs(delta) < sigma)
			{
				score += 1;
			}
		}

		if (score > bestScore)
		{
			A_best = A;
			b_best = b;
			bestScore = score;
		}
	}
}

bool solve3DLsq(const Eigen::MatrixXd& radar_data, 
                Vector3& v_r, 
                Eigen::Matrix3d P_v_r, 
                bool estimate_sigma)
{
  MatrixXd H(radar_data.rows(), 3);
  H.col(0)         = radar_data.col(0);
  H.col(1)         = radar_data.col(1);
  H.col(2)         = radar_data.col(2);
  const MatrixXd HTH = H.transpose() * H;

  const Vector y = radar_data.col(3);

  Eigen::JacobiSVD<MatrixXd> svd(HTH);
  double cond = svd.singularValues()(0) / svd.singularValues()(svd.singularValues().size() - 1);

  if (std::fabs(cond) < 1.0e3)
  {
    v_r = (HTH).ldlt().solve(H.transpose() * y);
    // v_r = H.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(y);

    if (estimate_sigma)
    {
      const Vector e    = H * v_r - y;
      P_v_r             = (e.transpose() * e).x() * (HTH).inverse() / (H.rows() - 3);
      Vector3 sigma_v_r = Vector3(P_v_r(0, 0), P_v_r(1, 1), P_v_r(2, 2));

      const Vector3 offset =
          Vector3(0.05,0.05,0.05)
              .array()
              .square();
      P_v_r += offset.asDiagonal();

      // check diagonal for valid estimation result
      if (sigma_v_r.x() >= 0.0 && sigma_v_r.y() >= 0.0 && sigma_v_r.z() >= 0.0)
      {
        sigma_v_r = sigma_v_r.array().sqrt();
        if (sigma_v_r.x() < 0.2 && sigma_v_r.y() < 0.2 &&
            sigma_v_r.z() < 0.2)
          return true;
      }
    }
    else
    {
      return true;
    }
  }

  return false;
}

bool solve2DLsq(const Eigen::MatrixXd& radar_data, 
                Vector3& v_r, 
                Vector3& sigma_v_r, 
                bool estimate_sigma)
{
  MatrixXd H(radar_data.rows(), 2);
  Eigen::Matrix3d P_v_r;
  H.col(0)         = radar_data.col(0);
  H.col(1)         = radar_data.col(1);
  const MatrixXd HTH = H.transpose() * H;

  const Vector y = radar_data.col(3);

  Eigen::JacobiSVD<MatrixXd> svd(HTH);
  double cond = svd.singularValues()(0) / svd.singularValues()(svd.singularValues().size() - 1);

  if (std::fabs(cond) < 1.0e3)
  {
    v_r.head<2>() = (HTH).ldlt().solve(H.transpose() * y);
    v_r(2)=0.0;
    // v_r = H.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(y);

    if (estimate_sigma)
    {
      const Vector e    = H * v_r.head<2>() - y;
      P_v_r.topLeftCorner<2,2>() = (e.transpose() * e).x() * (HTH).inverse() / (H.rows() - 2);
      sigma_v_r = Vector3(P_v_r(0, 0), P_v_r(1, 1), 0);

      const Vector3 offset =
          Vector3(0.05,0.05,0.05)
              .array()
              .square();
      P_v_r += offset.asDiagonal();

      // check diagonal for valid estimation result
      if (sigma_v_r.x() >= 0.0 && sigma_v_r.y() >= 0.0 && sigma_v_r.z() >= 0.0)
      {
        sigma_v_r = sigma_v_r.array().sqrt();
        if (sigma_v_r.x() < 0.2 && sigma_v_r.y() < 0.2 &&
            sigma_v_r.z() < 0.2)
          return true;
      }
    }
    else
    {
      return true;
    }
  }

  return false;
}

template <typename PointT> void
velocity_estimation(const PointT &radar_scan_msg,
                         Vector3 &v_r,
                         Vector3 &v_r_noisy,
                         pcl::PointCloud<OculiiPointCloudType>& radar_scan_inlier)
{
  Matrix3 P_v_r;
  pcl::PointCloud<OculiiPointCloudType> radar_scan;
  radar_scan_inlier.clear();
  // pcl2msgToPcl(radar_scan_msg, *radar_scan);
  int point_num = radar_scan_msg.targetNum;
  for (int i=0; i<point_num; i++)
  {
    OculiiPointCloudType p_;
    p_.x             = radar_scan_msg.x[i];
    p_.y             = radar_scan_msg.y[i];
    p_.z             = radar_scan_msg.z[i];
    p_.Alpha        = radar_scan_msg.azimuth[i];
    p_.Beta         = radar_scan_msg.elevation[i];
    p_.Power        = radar_scan_msg.snr[i];
    p_.Doppler = radar_scan_msg.v[i];
    p_.Range         = sqrt(pow(p_.x,2) + pow(p_.y,2) + pow(p_.z,2));
    radar_scan.push_back(p_);
  }

    std::vector<Vector11> valid_targets;
    for (uint i = 0; i < radar_scan.size(); ++i)
    {
      const auto target = radar_scan.points[i];
      const double r      = Vector3(target.x, target.y, target.z).norm();

      double azimuth   = target.Alpha;
      double elevation = target.Beta;

      // TODO make parameter
      Vector11 v;
      v << azimuth, elevation, target.x, target.y, target.z, target.Power, target.x / r, target.y / r,
          target.z / r, -target.Doppler, 0.0;
      valid_targets.emplace_back(v);
    }

    if (valid_targets.size() > 2)
    {
      // check for zero velocity
      std::vector<double> v_dopplers;
      for (const auto& v : valid_targets) v_dopplers.emplace_back(std::fabs(v[v_d]));
      const size_t n = v_dopplers.size() * (1.0 - 0.4);
      std::nth_element(v_dopplers.begin(), v_dopplers.begin() + n, v_dopplers.end());
      const auto median = v_dopplers[n];
      // std::cout << "Median: " << median << endl;
      if (median < 0.07)
      {
        ROS_INFO_STREAM_THROTTLE(0.5, "[velocity_estimation]:Zero velocity detected!");

        v_r = Vector3(0, 0, 0);
        P_v_r.setIdentity();
        P_v_r.diagonal() =
            Vector3(0.025, 0.025, 0.025)
                .array()
                .square();

        for (const auto& item : valid_targets)
          if (std::fabs(item[v_d]) < 0.05)
            radar_scan_inlier.push_back(toOculiiPointCloudType(item));
      }
      else
      {
        // LSQ velocity estimation
        MatrixXd radar_data(valid_targets.size(), 4); // rx, ry, rz, v
        uint itr = 0;
        for (const auto &v : valid_targets)
          radar_data.row(itr++) = Vector4(v[r_x], v[r_y], v[r_z], v[v_d]);

        std::vector<uint> inlier_idx_best;
        MatrixXd H_all(radar_data.rows(), 3);
        H_all.col(0) = radar_data.col(0);
        H_all.col(1) = radar_data.col(1);
        H_all.col(2) = radar_data.col(2);
        const Vector y_all = radar_data.col(3);

        std::vector<uint> idx(radar_data.rows());
        for (uint k = 0; k < radar_data.rows(); ++k)
          idx[k] = k;

        std::random_device rd;
        std::mt19937 g(rd());

        if (radar_data.rows() >= 3)
        {
          for (uint k = 0; k < 82; ++k)
          {
            std::shuffle(idx.begin(), idx.end(), g);
            MatrixXd radar_data_iter(3, 4);

            for (uint i = 0; i < 3; ++i)
            {
              radar_data_iter.row(i) = radar_data.row(idx.at(i));
            }

            if (solve2DLsq(radar_data_iter, v_r, v_r_noisy, false))
            {
              const Vector err = (y_all - H_all * v_r).array().abs();
              std::vector<uint> inlier_idx;
              for (uint j = 0; j < err.rows(); ++j)
              {
                if (err(j) < 0.15)
                {
                  inlier_idx.emplace_back(j);
                }
              }
              if (inlier_idx.size() > inlier_idx_best.size())
                {inlier_idx_best = inlier_idx;}
            }
          }
        }

        if (!inlier_idx_best.empty())
        {
          MatrixXd radar_data_inlier(inlier_idx_best.size(), 4);
          for (uint i = 0; i < inlier_idx_best.size(); ++i)
            radar_data_inlier.row(i) = radar_data.row(inlier_idx_best.at(i));

          solve2DLsq(radar_data_inlier, v_r, v_r_noisy, true);

        }

        for (const auto &index : inlier_idx_best)
        {
          radar_scan_inlier.push_back(toOculiiPointCloudType(valid_targets.at(index)));
        }
      }
    }
}

void updatePoses(void)
{
    mKF.lock();
    for (int key_idx=0,node_idx=0; node_idx < int(result.size()); node_idx++)
    {
      if(key_idx < int(keyframePosesUpdated.size()) && node_idx == keyframePosesUpdated[key_idx].first)
      {
        Pose6D& p = keyframePosesUpdated[key_idx].second;
        p.x = result.at<gtsam::Pose3>(X(node_idx)).translation().x();
        p.y = result.at<gtsam::Pose3>(X(node_idx)).translation().y();
        p.z = result.at<gtsam::Pose3>(X(node_idx)).translation().z();
        p.roll = result.at<gtsam::Pose3>(X(node_idx)).rotation().roll();
        p.pitch = result.at<gtsam::Pose3>(X(node_idx)).rotation().pitch();
        p.yaw = result.at<gtsam::Pose3>(X(node_idx)).rotation().yaw();
        key_idx++;
      }
    }
    mKF.unlock();

    mtxRecentPose.lock();
    // if(isamCurrentEstimate.size() > 0)
    // {
    //   const gtsam::Pose3& lastOptimizedPose = isamCurrentEstimate.at<gtsam::Pose3>(int(isamCurrentEstimate.size())-1);
    // }

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

void initNoises( void )
{
  Eigen::Matrix<double, 9, 1> initial_state = Eigen::Matrix<double, 9, 1>::Zero();
  Point3 prior_point(initial_state.head<3>());
  Vector3 prior_euler(initial_state.segment<3>(3));
  // Vector3 prior_euler(Vector3(1.11,0.09,0.58));
    // Vector3 prior_velocity(Vector3(1.35,0,0));
  Vector3 prior_velocity(initial_state.tail<3>());
  Rot3 prior_rotation = Rot3::ypr(degree2rad(prior_euler(0)),
                                  degree2rad(prior_euler(1)),
                                  degree2rad(prior_euler(2)));
  Pose3 prior_pose(prior_rotation, prior_point); // 初始位姿
  Vector6 bias_init;
  // bias_init << 0.000142,0.0002,0.018,-0.000035,-0.00008,-0.00009;
  bias_init << 0.0,0.0,0.0,-0.0,-0.0,-0.0;
  imuBias::ConstantBias prior_imu_bias(bias_init); // 残差

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
  graph->add(PriorFactor<Pose3>(X(vel_correction_count), prior_pose, pose_noise_model));
  graph->add(PriorFactor<Vector3>(V(vel_correction_count), prior_velocity, velocity_noise_model));
  graph->add(PriorFactor<imuBias::ConstantBias>(B(vel_correction_count), prior_imu_bias, bias_noise_model));
  // runISAM2opt();
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


  gyro_last_ned = Vector3(0.0, 0.0, 0.0);
  accel_last_ned = Vector3(0.0, 0.0, 0.0);
  gyro_last_body = Vector3(0.0, 0.0, 0.0);
  Vector3 lbr(0.0901, -0.1449, 0.0249); // colo

  boost::shared_ptr<PreintegratedCombinedMeasurements::Params> p = PreintegratedCombinedMeasurements::Params::MakeSharedD();
  // MakeSharedD:NED坐标系，g默认为 9.81
  // MakeSharedU：ENU坐标系，g默认为 9.81

  // 设置预积分分参数
  p->accelerometerCovariance = measured_acc_cov;
  p->integrationCovariance = integration_error_cov;
  p->gyroscopeCovariance = measured_omega_cov;

  // 预计分测量值
  p->biasAccCovariance = bias_acc_cov;
  p->biasAccOmegaInt = bias_acc_omega_int;
  p->biasOmegaCovariance = bias_omega_cov;
  Eigen::Matrix<double, 4, 4> Timu2body = Eigen::Matrix<double, 4, 4>::Identity();
  Timu2body.topRightCorner<3,1>() = Vector3(1.0, 0.0, -1.8);
  Pose3 bPs(Timu2body);
  p->body_P_sensor = bPs;

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

void performSCLoopClosure(void)
{
    if( keyframePoses.size() < scManager.NUM_EXCLUDE_RECENT) // do not try too early 
        return;

    if(loop_currnode == keyframePoses.size() - 1)
    {
      return;
    }

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
        loop_currnode = curr_node_idx;
    }
} // performSCLoopClosure

void callbackIMU(const sensor_msgs::ImuConstPtr &imu_msgs);
void callbackRadarScan(const radar_msgs::RadarTargetConstPtr &radar_msg, int radar_n);
void callbackGtPose(const sensor_msgs::NavSatFixConstPtr &gt_msg);
void callbackGtTwist(const nav_msgs::OdometryConstPtr &gt_msg);
void process_odom(std::vector<std::ofstream *>  file_list);

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
  pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
#ifdef LOOP_USE_ICP
  // ICP Settings
  pcl::IterativeClosestPoint<PointType, PointType> reg;
  reg.setMaxCorrespondenceDistance(50); // giseop , use a value can cover 2*historyKeyframeSearchNum range in meter
  reg.setMaximumIterations(100);
  reg.setTransformationEpsilon(1e-6);
  reg.setEuclideanFitnessEpsilon(1e-6);
  reg.setRANSACIterations(0);

  // Align pointclouds
  reg.setInputSource(cureKeyframeCloud);
  reg.setInputTarget(targetKeyframeCloud);
  reg.align(*unused_result);
#endif

#ifdef LOOP_USE_NDT
  // 初始化正态分布变换（NDT）
  pcl::NormalDistributionsTransform<pcl::PointXYZI, pcl::PointXYZI> reg;
  reg.setMaxCorrespondenceDistance(50);
  reg.setTransformationEpsilon(0.01);
  reg.setStepSize(0.1);
  reg.setResolution(2.0);
  reg.setMaximumIterations(100);
  reg.setInputSource(cureKeyframeCloud);
  reg.setInputTarget(targetKeyframeCloud);
  reg.align(*unused_result);

#endif 

  float loopFitnessScoreThreshold = 0.2; // user parameter but fixed low value is safe.
  if (reg.hasConverged() == false || reg.getFitnessScore() > loopFitnessScoreThreshold)
  {
    std::cout << "[SC loop] ICP fitness test failed (" << reg.getFitnessScore() << " > " << loopFitnessScoreThreshold << "). Reject this SC loop." << std::endl;
    return std::nullopt;
  }
  else
  {
    std::cout << "[SC loop] ICP fitness test passed (" << reg.getFitnessScore() << " < " << loopFitnessScoreThreshold << "). Add this SC loop." << std::endl;
  }

  // Get pose transformation
  float x, y, z, roll, pitch, yaw;
  Eigen::Affine3f correctionLidarFrame;
  correctionLidarFrame = reg.getFinalTransformation();
  pcl::getTranslationAndEulerAngles(correctionLidarFrame, x, y, z, roll, pitch, yaw);
  gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(x, y, z));
  gtsam::Pose3 poseTo = Pose3(Rot3::RzRyRx(0.0, 0.0, 0.0), Point3(0.0, 0.0, 0.0));

  return poseFrom.between(poseTo);
} // doICPVirtualRelative

void RadarPcl2Body(int radar_no)
{
  Eigen::Vector3d Vr,Vr_sigma;
  auto radar_data_msg = queue_radar[radar_no].front();
  velocity_estimation(radar_data_msg,Vr,Vr_sigma,radar_body[radar_no]);
  int num = radar_body[radar_no].size();
  // transfer to body 
  for (size_t i = 0; i < num; i++)
  {
    OculiiPointCloudType &p_sel = radar_body[radar_no].points[i];
    pointAssociateToSubMap(&p_sel, &p_sel, T_body_radar[radar_no]);
  }
  Vr = T_body_radar[radar_no].topLeftCorner<3,3>() * Vr;
  if(radar_vel_mode == "all")
  {
    gyro_last_body = gyro_last_body - prev_bias.vector().tail<3>();
    Eigen::Matrix<double, 3, 3> gyro_x;
    gyro_x << 0, -gyro_last_body(2), gyro_last_body(1),
        gyro_last_body(2), 0, -gyro_last_body(0),
        -gyro_last_body(1), gyro_last_body(0), 0;
    Vr -= gyro_x * l_body_radar[radar_no];
  }
  // else if(radar_vel_mode == "fov")
  // {
  //   ;
  // }

  radar_vel[radar_no] = Vr;
  radar_vel_sigma[radar_no] = Vr_sigma;
  queue_radar[radar_no].pop();
}

void process_queue()
{
  mutex_rawdata.lock();

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
    auto imu_msg_bag = queue_imu.front();
    swap(imu_msg_bag.a_b_ib.x(),imu_msg_bag.a_b_ib.y());
    swap(imu_msg_bag.w_b_ib.x(),imu_msg_bag.w_b_ib.y());

    // imu_msg_bag.a_b_ib.x() *= -1;
    // imu_msg_bag.w_b_ib.x() *= -1;
    // imu_msg_bag.a_b_ib.y() *= -1;
    // imu_msg_bag.w_b_ib.y() *= -1;

    imu_msg_bag.a_b_ib.z() *= -1;
    imu_msg_bag.w_b_ib.z() *= -1;
    imu_msg_bag.a_b_ib *= G;
    gyro_last_ned = imu_msg_bag.w_b_ib;
    accel_last_ned = imu_msg_bag.a_b_ib;
    gyro_last_body = R_ned_body.transpose() * imu_msg_bag.w_b_ib;

    // if((imu_msg_bag.time_stamp.toSec() > fov_radar_new_timestamp) && \
    //  (imu_msg_bag.time_stamp.toSec() - imu_msg_bag.dt < fov_radar_new_timestamp))
    // {
    //   imu_msg_bag.dt = fov_radar_new_timestamp - imu_msg_bag.time_stamp.toSec();
    //   last_imu.time_stamp.fromSec(fov_radar_new_timestamp); 
    // }
    mutex_imu.lock();
    imu_ned.push(imu_msg_bag); 
    mutex_imu.unlock();
    queue_imu.pop();
  }

  if (!queue_odom_twist.empty())
  {
    auto gt_msg = queue_odom_twist.front();
    Eigen::Quaterniond q_ned_body((Matrix3d)R_ned_body);
    q_w_curr.x() = gt_msg.pose.pose.orientation.x;
    q_w_curr.y() = gt_msg.pose.pose.orientation.y;
    q_w_curr.z() = gt_msg.pose.pose.orientation.z;
    q_w_curr.w() = gt_msg.pose.pose.orientation.w;
    gt_vel = Vector3(gt_msg.twist.twist.linear.x, gt_msg.twist.twist.linear.y, gt_msg.twist.twist.linear.z);
    // Eigen::Matrix3d R_w_curr = R_ned_body * q_w_curr.toRotationMatrix() * R_ned_body.transpose();
    Eigen::Matrix3d R_w_curr = q_w_curr.toRotationMatrix();
    gt_vel = R_w_curr.transpose() * gt_vel;
    
    if (!twist_initialed)
    {
      R_init = R_w_curr;
      twist_initialed = true;
      q_w_curr = Eigen::Quaterniond(0,0,0,1);
    }
    else
    {
      Rtrans = R_ned_body * R_init.transpose() * q_w_curr.toRotationMatrix()  * R_ned_body.transpose();
      q_w_curr = Eigen::Quaterniond(Rtrans);
    }
    queue_odom_twist.pop();
  }

  if (!queue_odom_pose.empty())
  {
    auto gt_pose = queue_odom_pose.front();
    Vector3 xyz(0, 0, 0);
    double height;
    if (!pose_initialed)
    {
      gps2xy.SetECEFOw(gt_pose.latitude * M_PI / 180.0, gt_pose.longitude * M_PI / 180.0, gt_pose.altitude);
      pose_initialed = true;
    }
    else
    {
      gps2xy.getENH(gt_pose.latitude * M_PI / 180.0, gt_pose.longitude * M_PI / 180.0, gt_pose.altitude, xyz(0), xyz(1), xyz(2));
    }
    t_w_curr(0) = xyz(0);//e
    t_w_curr(1) = xyz(1);//n
    t_w_curr(2) = xyz(2);//u
    height = t_w_curr(2);
    t_w_curr = R_init.transpose() * t_w_curr; //body
    t_w_curr(1)*=-1;
    t_w_curr(2)= height*(-1);
    queue_odom_pose.pop();
  }

  mutex_rawdata.unlock();
}

void process_data( std::vector<std::string>  path_list)
{
  nav_msgs::Path GtPath;
  std::ofstream radar_vel_esti(path_list[0], std::ios::trunc);
  std::ofstream radar_vel_filter(path_list[1], std::ios::trunc);
  std::ofstream ndt_odom(path_list[2], std::ios::trunc);
  std::ofstream gicp_odom(path_list[3], std::ios::trunc);
  std::ofstream icp_odom(path_list[4], std::ios::trunc);
  std::ofstream apdgicp_odom(path_list[5], std::ios::trunc);
  std::ofstream gt(path_list[6], std::ios::trunc);

  std::vector<std::ofstream *> file_list(5);
  file_list[0] = &radar_vel_filter;
  file_list[1] = &icp_odom;
  file_list[2] = &gicp_odom;
  file_list[3] = &ndt_odom;
  file_list[4] = &apdgicp_odom;


  radar_vel_esti.setf(std::ios::dec, std::ios::floatfield);
	radar_vel_esti.precision(15);
  // add calib
  T_body_radar[0] << 0.6916, -0.7222, -0.009174, 3.65,
      0.7222, 0.6916, -0.00958, 1.02,
      0.01326, 0, 0.9999, 0.68,
      0, 0, 0, 1;

  T_body_radar[1] << 0.9999, 0.008901, 0.01257, 3.7,
      -0.0089, 1, -0.0001119, -0.135,
      -0.01257, 0, 0.9999, 0.67,
      0, 0, 0, 1;

  // T_body_radar[1] << 0.9999, 0.008901, 0.01257, 2.7,
  //     -0.0089, 1, -0.0001119, -0.135,
  //     -0.01257, 0, 0.9999, -1.1,
  //     0, 0, 0, 1;

  T_body_radar[2] << 0.695, 0.719, 0.01031, 3.65,
      -0.7189, 0.695, -0.01067, -1.04,
      -0.01483, 0, 0.9999, 0.66,
      0, 0, 0, 1;

  T_body_radar[3] << -0.6785, -0.7328, 0.0522, -1.03,
      0.7306, -0.6805, -0.05622, 0.86,
      0.07672, 0, 0.9971, 0.6,
      0, 0, 0, 1;

  T_body_radar[4] << -1, -0.004887, -0.005236, -1.165,
      0.004887, -1, 0.0000256, -0.15,
      -0.005236, 0, 1, 0.635,
      0, 0, 0, 1;

  T_body_radar[5] << -0.7046, 0.7096, -0.0, -1,
      -0.7096, -0.7046, -0.0, -0.88,
      -0.0, 0, 1, 0.6,
      0, 0, 0, 1;
  for (size_t i = 0; i < RADAR_NUM; i++)
  {
    l_body_radar[i] = T_body_radar[i].topRightCorner<3,1>();
    l_ned_radar[i] = R_ned_body * T_body_radar[i].topRightCorner<3,1>();
  }
  
  int radar_new_index = 0;
  double radar_new_timestamp = 0;
  double gtodom_new_timestamp = 0;

      // read bag
  rosbag::Bag source_bag;
  std::string bagpath = bag_path + bag_name + ".bag";
  source_bag.open(bagpath, rosbag::bagmode::Read);
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
  auto first_timestamp = ros::TIME_MIN;
  for (const rosbag::MessageInstance &m : view)
  {

    if (first_timestamp == ros::TIME_MIN)
      first_timestamp = m.getTime();

    if ((m.getTime() - first_timestamp).toSec() < bag_start)
      continue;

    if ((m.getTime() - first_timestamp).toSec() > bag_duration)
      break;
    const auto topic = m.getTopic();
    radar_new_index = RADAR_NUM;
    if (topic == topic_imu)
    {
      const auto imu_msg_bag = m.instantiate<sensor_msgs::Imu>();
      if (imu_msg_bag != NULL)
      {
        imu_msg_bag->header.frame_id = "/camera_init";
        callbackIMU(imu_msg_bag);
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
        fov_radar_new_timestamp = radar_new_timestamp;
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

      if (gt_twist != NULL )
      {
        callbackGtTwist(gt_twist);
        // gtodom_new_timestamp = gt_twist->header.stamp.toSec();
      }
      topic_new = "gt_twist";
    }

    else if (topic == topic_gt_pose )
    {
      const auto gt_pose = m.instantiate<sensor_msgs::NavSatFix>();
      if (gt_pose != NULL  )
      {
        callbackGtPose(gt_pose);
        gtodom_new_timestamp = gt_pose->header.stamp.toSec();
      }
      topic_new = "gt_pose";
    }

    if(scanned_map.count(radar_new_index) && topic_new == "radar")
    {
      pcl::PointCloud<OculiiPointCloudType>::Ptr all_radar_body(new pcl::PointCloud<OculiiPointCloudType>);
      pair<int,Eigen::Vector3d> ego_vel;
      pair<int,Eigen::Vector3d> ego_vel_sigma;

      if(radar_odom_mode == "all")
      {
        for (auto p : scanned_map)
        {
          *all_radar_body += radar_body[p.first];
          
          radar_vel_esti << p.first << " "
          << radar_vel[p.first](0) << " "
          << radar_vel[p.first](1) << " "
          << radar_vel[p.first](2) << " " << endl;
        }
          radar_vel_esti << 6.0 << " " 
          << gt_vel(0) << " "
          << gt_vel(1) << " "
          << gt_vel(2) << " " << endl;    
      }
      else if (radar_odom_mode == "fov")
      {
        *all_radar_body += radar_body[1];
      }
      else
      {
        /* code */
      }

      if(radar_vel_mode == "all")
      {
        std::vector<pair<int,vector<double>>> temp;
        for (size_t i = 0; i < RADAR_NUM; i++)
        {
          vector<double> radar_vel_i;
          for (size_t j = 0; j < 3; j++)
          {
            radar_vel_i.push_back(radar_vel[i](j));
          }
          std::pair radar_vel_info(i,radar_vel_i);
          temp.push_back(radar_vel_info);
        }
        sort(temp.begin(), temp.end(), cmp1);
        for (size_t i = 0; i < 3; i++)
        {
          ego_vel.second[i] = temp[2].second[i];
        }
        ego_vel.first = temp[2].first;
      }
      else{
        ego_vel.first = 1;
        ego_vel.second = radar_vel[1];
        ego_vel_sigma.first = 1;
        ego_vel_sigma.second = radar_vel_sigma[1];
      }

      
      

      sensor_msgs::PointCloud2 RadarCloudLocalRos;
      pcl::toROSMsg(*all_radar_body, RadarCloudLocalRos);
      RadarCloudLocalRos.header.stamp = ros::Time().fromSec(scanned_map.begin()->second);
      RadarCloudLocalRos.header.frame_id = "/camera_init";
      pubRadarCloudLocal.publish(RadarCloudLocalRos);
      mutex_radar.lock();
      all_radar_scan.push(*all_radar_body);
      radar_ego_vel_sigma.push(ego_vel_sigma);
      radar_ego_vel.push(ego_vel);
      mutex_radar.unlock();
      scanned_map.clear();
    }

    if(!scanned_map.count(radar_new_index) && topic_new == "radar")
    {
      std::pair<int,double> radar_curr_scan(radar_new_index,radar_new_timestamp);
      scanned_map.insert(radar_curr_scan);
    }

    // if (topic == topic_radar[1])
    // {
    //   pair<int,Eigen::Vector3d> ego_vel;
    //   pair<int,Eigen::Vector3d> ego_vel_sigma;

    //   ego_vel.first = 1;
    //   ego_vel.second = radar_vel[1];
    //   ego_vel_sigma.first = 1;
    //   ego_vel_sigma.second = radar_vel_sigma[1];
    //   mutex_radar.lock();
    //   radar_ego_vel_sigma.push(ego_vel_sigma);
    //   radar_ego_vel.push(ego_vel);
    // fov_radar_scan.push(radar_body[1]);
    //   mutex_radar.unlock();
    // }


    if(twist_initialed && pose_initialed && topic_new == "gt_pose" )
    {
      nav_msgs::Odometry CurrOdom;
      tf::quaternionEigenToMsg(q_w_curr,CurrOdom.pose.pose.orientation);
      tf::pointEigenToMsg(t_w_curr,CurrOdom.pose.pose.position);
      CurrOdom.header.frame_id = "/camera_init";
      CurrOdom.header.stamp = ros::Time().now();
      pubGtOdom.publish(CurrOdom);
      gt << fixed;
      gt << setprecision(6) << gtodom_new_timestamp << " " << setprecision(9) 
          << t_w_curr(0) << ' '
          << t_w_curr(1) << ' '
          << t_w_curr(2) << ' '
          << q_w_curr.x() << " " << q_w_curr.y() << " " << q_w_curr.z() << " " << q_w_curr.w() << endl;

      geometry_msgs::PoseStamped GtPose;
      GtPose.pose = CurrOdom.pose.pose;
      GtPath.header.stamp = ros::Time().now();
      GtPath.header.frame_id = "/camera_init";
      GtPath.poses.push_back(GtPose);
      pubGtPath.publish(GtPath);
      mutex_gt.lock();
      gt_odom.push(CurrOdom);
      mutex_gt.unlock();
    }
    process_queue();
    process_odom(file_list);
  }

  std::string aftpg_path = output_path + bag_name + "_aftpg.txt";
  std::ofstream aftpg(aftpg_path, std::ios::trunc);
  aftpg << fixed;
  for (int i = 0; i < keyframePosesUpdated.size(); i++)
  {
    float cy = cos((keyframePosesUpdated[i].second.yaw) * 0.5);
    float sy = sin((keyframePosesUpdated[i].second.yaw) * 0.5);
    float cr = cos((keyframePosesUpdated[i].second.roll) * 0.5);
    float sr = sin((keyframePosesUpdated[i].second.roll) * 0.5);
    float cp = cos((keyframePosesUpdated[i].second.pitch) * 0.5);
    float sp = sin((keyframePosesUpdated[i].second.pitch) * 0.5);

    float w = cy * cp * cr + sy * sp * sr;
    float x = cy * cp * sr - sy * sp * cr;
    float y = sy * cp * sr + cy * sp * cr;
    float z = sy * cp * cr - cy * sp * sr;
    aftpg << setprecision(6) << keyframePosesUpdated[i].second.time << " " << setprecision(9) 
    // aftpg << setprecision(6) << 0.0 << " " << setprecision(9) 
          << keyframePosesUpdated[i].second.x << ' '
          << keyframePosesUpdated[i].second.y << ' '
          << keyframePosesUpdated[i].second.z << ' '
          << x << " " << y << " " << z << " " << w << endl;
          // << keyframePosesUpdated[i].second.roll * 180.0/M_PI << ' '
          // << keyframePosesUpdated[i].second.pitch * 180.0/M_PI<< ' '
          // << keyframePosesUpdated[i].second.yaw * 180.0/M_PI << std::endl;
  }
  gt.close();
  aftpg.close();
}

void process_odom(std::vector<std::ofstream *>  file_list)
{

  mutex_imu.lock();
  while (!imu_ned.empty())
  {
    auto imu_msg = imu_ned.front();
    // if (imu_msg.time_stamp.toSec() < fov_radar_new_timestamp)
    if(1)
    {
      imu_preintegrated_->integrateMeasurement(imu_msg.a_b_ib, imu_msg.w_b_ib, imu_msg.dt);
      imu_ned.pop();
    }
    else
    {
      break;
    }
  }
  mutex_imu.unlock();

  mutex_gt.lock();
  if (!gt_odom.empty())
  {
    auto gt_msg = gt_odom.front();
    gps(0) = gt_msg.pose.pose.position.x;
    gps(1) = gt_msg.pose.pose.position.y;
    gps(2) = gt_msg.pose.pose.position.z;
    gt_odom.pop();
  }
  mutex_gt.unlock();

  if(!all_radar_scan.empty())
  {
    if (pcl_initialed)
    {
      pcl::copyPointCloud(*src, *tar);
    }

    src->clear();
    mutex_radar.lock();
    pcl::PointCloud<OculiiPointCloudType>::Ptr scan_Radar(new pcl::PointCloud<OculiiPointCloudType>);
    *scan_Radar = all_radar_scan.front();
    all_radar_scan.pop();
    vel_Radar = radar_ego_vel.front().second;
    vel_Radar_sigma = radar_ego_vel_sigma.front().second;
    vel_Radar(2) = 0.0;//Vz
    int vel_Radar_No = radar_ego_vel.front().first;
    // fov_radar_scan.pop();
    radar_ego_vel.pop();
    radar_ego_vel_sigma.pop();
    mutex_radar.unlock();

    int point_num = scan_Radar->size();
    for (size_t i = 0; i < point_num; i++)
    {
      pcl::PointXYZI p_sel;
      p_sel.x = scan_Radar->points[i].x;
      p_sel.y = scan_Radar->points[i].y;
      p_sel.z = scan_Radar->points[i].z;
      p_sel.intensity = scan_Radar->points[i].Power;
      src->push_back(p_sel);
    }
    // downSizeFilterScan.setLeafSize(0.1,0.1,0.1);
    // downSizeFilterScan.setInputCloud(src);
    // downSizeFilterScan.filter(*src);

    if (!pcl_initialed)
    {
      pcl::copyPointCloud(*src, *tar);
      pcl_initialed = true;
    }

        pcl::PointCloud<pcl::PointXYZI>::Ptr Final(new pcl::PointCloud<pcl::PointXYZI>);
#ifdef USE_FGICP
    // GICP
    fast_gicp::FastGICPSingleThread<pcl::PointXYZI, pcl::PointXYZI> registration;
    registration.clearTarget();
    registration.clearSource();
    std::vector<int> mapping;
    pcl::removeNaNFromPointCloud(*src, *src, mapping);
    pcl::removeNaNFromPointCloud(*tar, *tar, mapping);
    registration.setInputTarget(tar);
    registration.setInputSource(src);
    registration.setCorrespondenceRandomness(5);
    registration.align(*Final);
    double score = registration.getFitnessScore();
#endif

    // pcl::IterativeClosestPoint<pcl::PointXYZI, pcl::PointXYZI> icp;
    // icp.setInputSource(src);
    // icp.setInputTarget(tar);
    // // icp.setMaximumIterations(200);
    // icp.align(*Final);

    // pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZI, pcl::PointXYZI> gicp;
    // gicp.setInputTarget(tar);
    // gicp.setInputSource(src);
    // gicp.setCorrespondenceRandomness(5);
    // gicp.align(*Final);

    // 初始化正态分布变换（NDT）
    pcl::NormalDistributionsTransform<pcl::PointXYZI, pcl::PointXYZI> ndt;
    ndt.setTransformationEpsilon(0.01);
    ndt.setStepSize(0.1);
    ndt.setResolution(2.0);
    ndt.setMaximumIterations(30);
    ndt.setInputSource(src);
    ndt.setInputTarget(tar);
    ndt.align(*Final);

    // fast_gicp::FastAPDGICP<pcl::PointXYZI, pcl::PointXYZI>::Ptr apdgicp(new fast_gicp::FastAPDGICP<pcl::PointXYZI, pcl::PointXYZI>());
    // apdgicp->setNumThreads(0);
    // apdgicp->setTransformationEpsilon(0.01);
    // apdgicp->setMaximumIterations(64);
    // apdgicp->setMaxCorrespondenceDistance( 2.5);
    // apdgicp->setCorrespondenceRandomness(20);
    // apdgicp->setDistVar(0.86);
    // apdgicp->setAzimuthVar(0.5);
    // apdgicp->setElevationVar(1.0);
    // apdgicp->setInputSource(src);
    // apdgicp->setInputTarget(tar);
    // apdgicp->align(*Final);

#ifdef USE_GICP
  Eigen::Matrix<double, 4, 4> icp_result = T_ned_body * gicp.getFinalTransformation().cast<double>() * T_ned_body.inverse();
#endif

#ifdef USE_NDT
  Eigen::Matrix<double, 4, 4> icp_result = T_ned_body * ndt.getFinalTransformation().cast<double>() * T_ned_body.inverse();
#endif



    // std::cout << "has converged:" << registration.hasConverged() << " score: " << registration.getFitnessScore() << std::endl;
    // std::cout << registration.getFinalTransformation() << std::endl;
    // currOdom_icp = currOdom_icp * (T_ned_body * icp.getFinalTransformation().cast<double>() * T_ned_body.inverse());
    // currOdom_gicp = currOdom_gicp * (T_ned_body * gicp.getFinalTransformation().cast<double>() * T_ned_body.inverse());
    // currOdom_ndt = currOdom_ndt * (T_ned_body * ndt.getFinalTransformation().cast<double>() * T_ned_body.inverse());
    // currOdom_apdgicp = currOdom_apdgicp * (T_ned_body * apdgicp->getFinalTransformation().cast<double>() * T_ned_body.inverse());

    // Vector3d rpy_icp = R2rpy(currOdom_icp.topLeftCorner<3,3>());
    // Vector3d rpy_gicp = R2rpy(currOdom_gicp.topLeftCorner<3,3>());
    // Vector3d rpy_ndt = R2rpy(currOdom_ndt.topLeftCorner<3,3>());
    // Vector3d rpy_apdgicp = R2rpy(currOdom_apdgicp.topLeftCorner<3,3>());

    // *file_list[1]<< currOdom_icp(0,3) << ' ' <<currOdom_icp(1,3)<< ' '<<currOdom_icp(2,3)<< ' '
    // << rpy_icp(2) <<' '<< rpy_icp(1) <<' '<< rpy_icp(0) << endl;
    // *file_list[2]<< currOdom_gicp(0,3) << ' ' <<currOdom_gicp(1,3)<< ' '<<currOdom_gicp(2,3)<< ' '
    // << rpy_gicp(2) <<' '<< rpy_gicp(1) <<' '<< rpy_gicp(0) << endl;
    // *file_list[3]<< currOdom_ndt(0,3) << ' ' <<currOdom_ndt(1,3)<< ' '<<currOdom_ndt(2,3)<< ' '
    // << rpy_ndt(2) <<' '<< rpy_ndt(1) <<' '<< rpy_ndt(0) << endl;
    // *file_list[4]<< currOdom_apdgicp(0,3) << ' ' <<currOdom_apdgicp(1,3)<< ' '<<currOdom_apdgicp(2,3)<< ' '
    // << rpy_apdgicp(2) <<' '<< rpy_apdgicp(1) <<' '<< rpy_apdgicp(0) << endl;


    currOdom = currOdom * icp_result;
    // Rtrans = currOdom.topLeftCorner<3, 3>();
    // t_w_curr = currOdom.topRightCorner<3, 1>();

    // Get pose transformation
    Eigen::Matrix3d icp_Rtrans = icp_result.topLeftCorner<3, 3>();
    Eigen::Vector3d icp_Ptrans = icp_result.topRightCorner<3, 1>();

    Rot3 icp_R(icp_Rtrans);
    Pose3 icp_pose(icp_R, icp_Ptrans);
    Pose3 curr_pose(currOdom);
    radar_scan_count ++;

    if(!imu_initialed)
    {
      vel_correction_count++;
    }
    else if(radar_scan_count % vel_correction_freq == 0)
    {
      vel_correction_count++;
    }
    else
    {
      return;
    }


    Vector6 icp_noise_sigma;
    icp_noise_sigma << 0.001, 0.001, 0.001, 0.01, 0.01, 0.01;
    noiseModel::Diagonal::shared_ptr icp_noise_model = noiseModel::Diagonal::Sigmas(icp_noise_sigma);
    icp_correction_count = vel_correction_count;
    BetweenFactor<gtsam::Pose3> odomtry_factor(X(icp_correction_count - 1), X(icp_correction_count), icp_pose, icp_noise_model);

    if(sensor_type.find("icp") != string::npos)
    {
      graph->add(odomtry_factor);
    }


    

    vel_Radar = R_ned_body * vel_Radar;

    bool static_state = false;
    if(vel_Radar(0) < 0.01)
    {
      static_state = true;
    }

    // 预计分测量值
    PreintegratedCombinedMeasurements *preint_imu_combined = dynamic_cast<PreintegratedCombinedMeasurements *>(imu_preintegrated_);
    // IMU 因子
    // typedef NoiseModelFactor6<Pose3, Vector3, Pose3, Vector3,imuBias::ConstantBias, imuBias::ConstantBias>
    CombinedImuFactor imu_factor(X(vel_correction_count - 1), V(vel_correction_count - 1),
                                 X(vel_correction_count), V(vel_correction_count),
                                 B(vel_correction_count - 1), B(vel_correction_count),
                                 *preint_imu_combined);

    // 迭代更新求解imu预测值
    prop_state = imu_preintegrated_->predict(prev_state, prev_bias);
    
    // fault detect
    bool fault_detect = true;
    Vector3 gyro_last_ = gyro_last_ned - prev_bias.vector().tail<3>();
    Eigen::Matrix<double, 3, 3> gyro_x;
    gyro_x << 0, -gyro_last_(2), gyro_last_(1),
        gyro_last_(2), 0, -gyro_last_(0),
        -gyro_last_(1), gyro_last_(0), 0;
    Vector3 vel_Radar_pred = prop_state.bodyVelocity() + gyro_x*l_ned_radar[vel_Radar_No];
    Vector3 vel_delta = vel_Radar - vel_Radar_pred;
    if(((fabs(vel_delta(0)) > 0.1 ) || \
    (fabs(vel_delta(1)) > 0.1 )) && imu_initialed)
    {
      fault_detect = true;
      fault_detect_cnt ++;
      std::cout << "vel_Radar fault detect!"<< endl;
    }
    else
    {
      fault_detect = false;
      fault_detect_cnt = 0;
    }

    if(fault_detect_cnt > 5)
    {
      fault_detect = false;
    }

    // double Vx = KalmanFilter(&KF_Fliter,vel_Radar(0));
    *file_list[0] << vel_Radar_pred(0) << " " 
    << vel_Radar_pred(1) << " "
    << vel_Radar_pred(2) << endl;

    Vector3 vel_noise_sigma;
    if(fault_detect)
    {
      vel_noise_sigma << 0.15, 0.15, 0.15;
    }
    else
    {
      vel_noise_sigma << 0.1, 0.1, 0.1;
    }

    noiseModel::Diagonal::shared_ptr radar_noise_model = noiseModel::Diagonal::Sigmas(vel_noise_sigma);


    // else if (radar_vel_mode == "fov")
    // {
      RadarFactor radar_factor(X(vel_correction_count), V(vel_correction_count), B(vel_correction_count-1),
                              vel_Radar,
                              gyro_last_ned,
                              l_ned_radar[vel_Radar_No],
                              radar_noise_model);
    // }

    if (radar_vel_mode == "all")
    {
      RadarFactor radar_factor(X(vel_correction_count), V(vel_correction_count), B(vel_correction_count-1),
                              vel_Radar,
                              gyro_last_ned,
                              Vector3(0.0, 0.0, 0.0),
                              radar_noise_model);
    }

    if(static_state)
    {

      Pose3 static_pose(Eigen::Matrix<double, 4, 4>::Identity());
      Vector6 static_pose_noisy;
      static_pose_noisy << 0.0001,0.0001,0.0001,0.0001,0.0001,0.0001;
      noiseModel::Diagonal::shared_ptr static_noise_model = noiseModel::Diagonal::Sigmas(static_pose_noisy);
      BetweenFactor<gtsam::Pose3> static_factor(X(vel_correction_count - 1), X(vel_correction_count), static_pose, static_noise_model);
      graph->add(static_factor);
    }

    if (!vel_initialed)
    {
      vel_Radar_last = vel_Radar;
      vel_initialed = true;
    }

    if(!imu_initialed && vel_correction_count > 600)
    {
      imu_initialed = true;
    }


    noiseModel::Diagonal::shared_ptr gps_correction_noise = noiseModel::Isotropic::Sigma(3, 0.1);
    GPSFactor gps_factor(X(vel_correction_count), 
    Point3(gps(0), gps(1), gps(2)), //(N,E,D)
    gps_correction_noise);

    mtxPosegraph.lock();
    if(sensor_type.find("imu") != string::npos)
    {
      graph->add(imu_factor);
    }

    // if(sensor_type.find("icp") != string::npos)
    // {
    //   graph->add(odomtry_factor);
    // }

    if(sensor_type.find("radar") != string::npos && !fault_detect)
    {
      graph->add(radar_factor);
    }

    if(sensor_type.find("gps") != string::npos)
    {
      graph->add(gps_factor);
    }



    initial_values.insert(X(vel_correction_count), prop_state.pose());
    initial_values.insert(V(vel_correction_count), prop_state.v());
    initial_values.insert(B(vel_correction_count), prev_bias);

    LevenbergMarquardtOptimizer optimizer(*graph, initial_values);
    result = optimizer.optimize();
    // 更新下一步预计分初始值
    // 导航状态 
    // runISAM2opt();
    mtxPosegraph.unlock();


    // prev_state = NavState(isamCurrentEstimate.at<Pose3>(X(vel_correction_count)),
    //                       isamCurrentEstimate.at<Vector3>(V(vel_correction_count)));
    // prev_bias = isamCurrentEstimate.at<imuBias::ConstantBias>(B(vel_correction_count));


    prev_state = NavState(result.at<Pose3>(X(vel_correction_count)),
                          result.at<Vector3>(V(vel_correction_count)));
    prev_bias = result.at<imuBias::ConstantBias>(B(vel_correction_count));
    // 更新预计分值
    imu_preintegrated_->resetIntegrationAndSetBias(prev_bias);
    // 更新关键帧位姿
    updatePoses();

    mBuf.lock();
    Pose6D pose_tmp = GTSAMPose3toPose6D(result.at<Pose3>(X(vel_correction_count)));
    odomBuf.push(pose_tmp);
    fullResBuf.push(*src);
    mBuf.unlock();

    // 计算角度误差和误差
    Vector3 gtsam_position = prev_state.pose().translation();
    // 位置误差
    Vector2 position_error = (gtsam_position - t_w_curr).head<2>();
    // 误差的范数
    double current_position_error = position_error.norm(); // 归一化

    // 姿态误差
    Eigen::Quaternion gtsam_quat = prev_state.pose().rotation().toQuaternion();
    Eigen::Quaternion gps_quat = q_w_curr;
    Eigen::Quaternion quat_error = gtsam_quat * gps_quat.inverse();
    quat_error.normalized();                                                               // 归一化
    Vector3 euler_angle_error(quat_error.x() * 2, quat_error.y() * 2, quat_error.z() * 2); // 转换为欧拉角误差
    double current_orientation_error = euler_angle_error.norm();
    Eigen::Vector3d gtsam_euler = R2rpy(gtsam_quat.toRotationMatrix());
    Eigen::Vector3d gt_euler = R2rpy(gps_quat.toRotationMatrix());
    // 输出误差
    cout << "Position error:" << current_position_error << "\t "
         << "Angular error:" << current_orientation_error << "\n";
    fprintf(fp, "%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n",
            output_time, gtsam_position(0), gtsam_position(1), gtsam_position(2),
            gtsam_euler(0), gtsam_euler(1), gtsam_euler(2),
            t_w_curr(0), t_w_curr(1), t_w_curr(2),
            gt_euler(0), gt_euler(1), gt_euler(2),
            // gyro_last_ned(0),gyro_last_ned(1),gyro_last_ned(2),
            // accel_last_ned(0), accel_last_ned(1), accel_last_ned(2));
            prev_bias.vector()(0), prev_bias.vector()(1), prev_bias.vector()(2),
            prev_bias.vector()(3), prev_bias.vector()(4), prev_bias.vector()(5));

    output_time += 1.0;
  }
}

void process_pg()
{
  std::string key_pair_path = output_path + bag_name + "_key.txt";
  std::ofstream key(key_pair_path, std::ios::trunc);

  while (1)
  {
    while (!odomBuf.empty() && !fullResBuf.empty())
    {
      // Time equal check

      // TODO
      mBuf.lock();
      pcl::PointCloud<PointType>::Ptr thisKeyFrame(new pcl::PointCloud<PointType>());
      pcl::copyPointCloud(fullResBuf.front(),*thisKeyFrame);
      fullResBuf.pop();

      Pose6D pose_curr =odomBuf.front();
      odomBuf.pop();
      mBuf.unlock();
      odom_node++;

      //
      // Early reject by counting local delta movement (for equi-spereated kf drop)
      //
      odom_pose_prev = odom_pose_curr;
      odom_pose_curr = pose_curr;
      double delta_translation = transDiff(odom_pose_prev, odom_pose_curr);
      movementAccumulation += delta_translation;
      odom_length += delta_translation;
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
      // pcl::KdTreeFLANN<pcl::PointXYZI> KNNkdtree;
      // KNNkdtree.setInputCloud(thisKeyFrame);
      // string path = "/home/invoker/catkin_ws/src/ICP-4DRadar/result/data_jihe/before_KNNFilter.pcd";
      // pcl::io::savePCDFileASCII(path,*thisKeyFrame);
      float radius = 1.0;
      // for (int i = 0; i < thisKeyFrame->size(); i++)
      // {
      //   std::vector<int> pointIndxRadiusSearch;
      //   std::vector<float> pointNRSquareDis;
      //   int size = KNNkdtree.radiusSearch(thisKeyFrame->points[i], radius, pointIndxRadiusSearch, pointNRSquareDis);
      //   if (size <= 40)
      //   {
      //     thisKeyFrame->erase(thisKeyFrame->begin() + i);
      //   }
      // }

      pcl::RadiusOutlierRemoval<pcl::PointXYZI> outrem; // 创建半径滤波器对象
      outrem.setInputCloud(thisKeyFrame);           // 设置输入点云
      outrem.setRadiusSearch(radius);                // 设置半径为1.0m
      outrem.setMinNeighborsInRadius(5);     // 设置最小邻接点个数阈值,半径范围内其他点个数少于5的点将被滤除
      outrem.filter(*thisKeyFrame);                // 执行滤波
      // path = "/home/invoker/catkin_ws/src/ICP-4DRadar/result/data_jihe/after_KNNFilter.pcd";
      // pcl::io::savePCDFileASCII(path,*thisKeyFrame);
      //
      // Save data and Add consecutive node
      //
      pcl::PointCloud<PointType>::Ptr thisKeyFrameDS(new pcl::PointCloud<PointType>());
      downSizeFilterScancontext.setInputCloud(thisKeyFrame);
      downSizeFilterScancontext.filter(*thisKeyFrameDS);

      if(save_keyframe)
      {
        std::stringstream filepath;
        filepath << bag_path << "keyframe/" <<keyframeLaserClouds.size() << ".pcd";
        pcl::io::savePCDFileASCII(filepath.str(), *thisKeyFrameDS);
      }

      mKF.lock();
      keyframeLaserClouds.push_back(thisKeyFrameDS);
      keyframePoses.push_back(std::pair<int,Pose6D>(odom_node,pose_curr));
      keyframePosesUpdated.push_back(std::pair<int,Pose6D>(odom_node,pose_curr)); // init
      scManager.makeAndSaveScancontextAndKeys(*thisKeyFrameDS);
      mKF.unlock();
      const int prev_node_idx = keyframePoses.size() - 2;
      const int curr_node_idx = keyframePoses.size() - 1; // becuase cpp starts with 0 (actually this index could be any number, but for simple implementation, we follow sequential indexing)
      key_pair.insert(std::pair(curr_node_idx,odom_node));

      key <<  curr_node_idx << ' ' << key_pair[curr_node_idx] << endl;
      
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
  key.close();
} // process_pg

void process_LC_icp(void)
{
  std::string lcd_path = output_path + bag_name + "_lcd.txt";
  std::ofstream lcd(lcd_path, std::ios::trunc);
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
                lcd << prev_node_idx << ' ' << curr_node_idx <<' '<< relative_pose.x() << ' '
                << relative_pose.y() << ' '
                << relative_pose.z() << ' '
                << relative_pose.rotation().yaw() * 180/M_PI << ' '
                << relative_pose.rotation().pitch() * 180/M_PI << ' '
                << relative_pose.rotation().roll() * 180/M_PI << endl;

                double delta = sqrt(relative_pose.x()*relative_pose.x() + relative_pose.y()*relative_pose.y() + relative_pose.z()*relative_pose.z());
                Vector6 icp_noise_sigma;
                icp_noise_sigma << 0.01, 0.01, 0.01, 0.2, 0.2, 0.2;
                noiseModel::Diagonal::shared_ptr icp_noise_model = noiseModel::Diagonal::Sigmas(icp_noise_sigma);
                mtxPosegraph.lock();
                if(keyframePoses[curr_node_idx].first <= odom_node && \
                sensor_type.find("loop") !=string::npos && \
                delta/odom_length < 0.01)
                {
                  graph->add(gtsam::BetweenFactor<gtsam::Pose3>(X(keyframePoses[prev_node_idx].first), 
                                                                X(keyframePoses[curr_node_idx].first), 
                                                                relative_pose, icp_noise_model));
                }
                // runISAM2opt();
                // updatePoses();
                mtxPosegraph.unlock();
            }
        }

        // wait (must required for running the while loop)
        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
    lcd.close();
} // process_LC_icp

void process_lcd(void)
{
    std::cout << "[process_lcd]: starting..." << endl;
    float loopClosureFrequency = 0.5; // can change
    ros::Rate rate(loopClosureFrequency);
    while (ros::ok())
    {
        rate.sleep();
        performSCLoopClosure();
        // performRSLoopClosure(); // TODO
    }
    std::cout << "[process_lcd]: stoping..." << endl;
} // process_lcd

void process_viz_map()
{
    std::cout << "[process_viz_map]: starting..." << endl;
    float vizmapFrequency = 0.2;
    int SKIP_FRAMES = 2;
    int counter = 0;
    ros::Rate rate(vizmapFrequency);
    while (ros::ok()) {
        rate.sleep();
        downSizeFilterMap->clear();
        mKF.lock();
        for (int node_idx = 0; node_idx < int(keyframePosesUpdated.size()); node_idx++)
        {
          if (counter % SKIP_FRAMES == 0)
          {
            *downSizeFilterMap += *local2global(keyframeLaserClouds[node_idx], keyframePosesUpdated[node_idx].second);
          }
          counter++;
        }
        mKF.unlock();

        downSizeFilterMapPGO.setInputCloud(downSizeFilterMap);
        downSizeFilterMapPGO.filter(*downSizeFilterMap);

        if (keyframeLaserClouds.size() > 1)
        {
          sensor_msgs::PointCloud2 RadarCloudSurround;
          pcl::toROSMsg(*downSizeFilterMap, RadarCloudSurround);
          RadarCloudSurround.header.stamp = ros::Time().now();
          RadarCloudSurround.header.frame_id = "/camera_init";
          pubRadarCloudSurround.publish(RadarCloudSurround);

          sensor_msgs::PointCloud2 RadarSubMap;
          pcl::toROSMsg(*RadarCloudMap, RadarSubMap);
          RadarSubMap.header.stamp = ros::Time().now();
          RadarSubMap.header.frame_id = "/camera_init";
          pubRadarSubMap.publish(RadarSubMap);
        }
    }
    std::cout << "[process_viz_map]: stoping..." << endl;
}

void process_viz_path(void)
{
  std::cout << "[process_viz_path]: starting..." << endl;
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

          // pub odom and path
          nav_msgs::Odometry odomAftPGO;
          nav_msgs::Path pathAftPGO;
          nav_msgs::Path pathAftPGO_woLoop;
          pathAftPGO.header.frame_id = "/camera_init";
          mKF.lock();
          for (int node_idx = 0; node_idx < int(keyframePosesUpdated.size()) - 1; node_idx++) // -1 is just delayed visualization (because sometimes mutexed while adding(push_back) a new one)
          {
            const Pose6D &pose_est = keyframePosesUpdated.at(node_idx).second; // upodated poses
            // const gtsam::Pose3& pose_est = isamCurrentEstimate.at<gtsam::Pose3>(node_idx);

            nav_msgs::Odometry odomAftPGOthis;
            odomAftPGOthis.header.frame_id = "/camera_init";
            odomAftPGOthis.child_frame_id = "/aft_pgo";
            odomAftPGOthis.header.stamp = ros::Time().now();
            odomAftPGOthis.pose.pose.position.x = pose_est.x;
            odomAftPGOthis.pose.pose.position.y = pose_est.y;
            odomAftPGOthis.pose.pose.position.z = pose_est.z;
            odomAftPGOthis.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(pose_est.roll, pose_est.pitch, pose_est.yaw);
            odomAftPGO = odomAftPGOthis;

            geometry_msgs::PoseStamped poseStampAftPGO;
            poseStampAftPGO.header = odomAftPGOthis.header;
            poseStampAftPGO.pose = odomAftPGOthis.pose.pose;

            pathAftPGO.header.stamp = odomAftPGOthis.header.stamp;
            pathAftPGO.header.frame_id = "/camera_init";
            pathAftPGO.poses.push_back(poseStampAftPGO);
          }

          for (int node_idx = 0; node_idx < int(keyframePoses.size()) - 1; node_idx++)
          {
            const Pose6D &pose_est = keyframePoses.at(node_idx).second; // upodated poses
            // const gtsam::Pose3& pose_est = isamCurrentEstimate.at<gtsam::Pose3>(node_idx);

            nav_msgs::Odometry odomAftPGOthis;
            odomAftPGOthis.header.frame_id = "/camera_init";
            odomAftPGOthis.child_frame_id = "/aft_pgo_wo_Loop";
            odomAftPGOthis.header.stamp = ros::Time().now();
            odomAftPGOthis.pose.pose.position.x = pose_est.x;
            odomAftPGOthis.pose.pose.position.y = pose_est.y;
            odomAftPGOthis.pose.pose.position.z = pose_est.z;
            odomAftPGOthis.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(pose_est.roll, pose_est.pitch, pose_est.yaw);
            odomAftPGO = odomAftPGOthis;

            geometry_msgs::PoseStamped poseStampAftPGO;
            poseStampAftPGO.header = odomAftPGOthis.header;
            poseStampAftPGO.pose = odomAftPGOthis.pose.pose;

            pathAftPGO_woLoop.header.stamp = odomAftPGOthis.header.stamp;
            pathAftPGO_woLoop.header.frame_id = "/camera_init";
            pathAftPGO_woLoop.poses.push_back(poseStampAftPGO);
          }
          
          mKF.unlock();
          pubOdomAftPGO.publish(odomAftPGO); // last pose
          pubPathAftPGO.publish(pathAftPGO); // poses
          pubPathAftPGO_woLoop.publish(pathAftPGO_woLoop);

          static tf::TransformBroadcaster br;
          tf::Transform transform;
          tf::Quaternion q;
          transform.setOrigin(tf::Vector3(odomAftPGO.pose.pose.position.x, odomAftPGO.pose.pose.position.y, odomAftPGO.pose.pose.position.z));
          q.setW(odomAftPGO.pose.pose.orientation.w);
          q.setX(odomAftPGO.pose.pose.orientation.x);
          q.setY(odomAftPGO.pose.pose.orientation.y);
          q.setZ(odomAftPGO.pose.pose.orientation.z);
          transform.setRotation(q);
          br.sendTransform(tf::StampedTransform(transform, odomAftPGO.header.stamp, "/camera_init", "/aft_pgo"));
        }
    }
    std::cout << "[process_viz_path]: stoping..." << endl;
}

int main(int argc, char **argv)
{

  ros::init(argc, argv, "radar_odometry");
  ros::NodeHandle n("~");

  n.getParam("bag_path", bag_path);
  n.getParam("bag_name", bag_name);
  n.getParam("output_path", output_path);
  n.getParam("topic_imu", topic_imu);
  n.getParam("topic_radar0", topic_radar[0]);
  n.getParam("topic_radar1", topic_radar[1]);
  n.getParam("topic_radar2", topic_radar[2]);
  n.getParam("topic_radar3", topic_radar[3]);
  n.getParam("topic_radar4", topic_radar[4]);
  n.getParam("topic_radar5", topic_radar[5]);
  n.getParam("topic_gt_twist", topic_gt_twist);
  n.getParam("topic_gt_pose", topic_gt_pose);
  n.param<string>("radar_odom_mode", radar_odom_mode, "all");
  n.param<string>("radar_vel_mode", radar_vel_mode, "all");
  n.param<string>("sensor_type", sensor_type, "imugps");
  n.param<double>("sc_dist_thres", scDistThres, 0);
  n.param<double>("bag_start", bag_start,0);
  n.param<double>("bag_duration", bag_duration, 1e6);
  n.param<int>("vel_correction_freq", vel_correction_freq, 1);
  n.param<bool>("save_keyframe", save_keyframe, false);

  pubRadarCloudLocal = n.advertise<sensor_msgs::PointCloud2>("/radar_cloud_local", 100);
  pubGtOdom = n.advertise<nav_msgs::Odometry>("/gt_odom",100);
  pubGtPath = n.advertise<nav_msgs::Path>("/gt_path",100);
  pubRadarCloudSurround = n.advertise<sensor_msgs::PointCloud2>("/radar_cloud_surround", 100);
  pubRadarSubMap = n.advertise<sensor_msgs::PointCloud2>("/radar_submap", 100);
  pubRadarPath = n.advertise<nav_msgs::Path>("/radar_path", 5);
  pubRadarVel = n.advertise<geometry_msgs::Twist>("/radar_vel", 5);
  pubOdomAftPGO = n.advertise<nav_msgs::Odometry>("/aft_pgo_odom", 100);
  pubPathAftPGO = n.advertise<nav_msgs::Path>("/aft_pgo_path", 100);
  pubPathAftPGO_woLoop = n.advertise<nav_msgs::Path>("/aft_pgo_woLoop_path", 100);

  float filter_size = 0.2;
  downSizeFilterScancontext.setLeafSize(filter_size, filter_size, filter_size);
  downSizeFilterICP.setLeafSize(filter_size, filter_size, filter_size);

  float map_vis_size = 0.2;
  downSizeFilterMapPGO.setLeafSize(map_vis_size, map_vis_size, map_vis_size);

  T_ned_body << 1, 0, 0, 0,
  0, -1, 0, 0,
  0.0, 0.0, -1, -0.0,
  0.0, 0.0, 0.0, 1.0;

  R_ned_body << 1, 0 ,0,
  0, -1, 0,
  0, 0, -1;

  // radar_ego_velocity_estimation::RadarEgoVelocityEstimatorConfig config;
  // config_init(config);
  // radar_ego_velocity.configure(config);
  currOdom = Eigen::Matrix<double, 4, 4>::Identity();
  currOdom_icp = Eigen::Matrix<double, 4, 4>::Identity();
  currOdom_gicp = Eigen::Matrix<double, 4, 4>::Identity();
  currOdom_ndt = Eigen::Matrix<double, 4, 4>::Identity();
  currOdom_apdgicp = Eigen::Matrix<double, 4, 4>::Identity();

  ISAM2Params parameters;
  parameters.relinearizeThreshold = 0.01;
  parameters.relinearizeSkip = 1;
  isam = new ISAM2(parameters);

  initNoises();
  Init_KalmanInfo(&KF_Fliter,0.1,0.1);
  scManager.setSCdistThres(scDistThres);

  std::vector<string> path_list(7);
  string outputfile = output_path  + sensor_type + "_" + radar_odom_mode +".csv";
  fp = fopen(outputfile.c_str(), "w+");
  fprintf(fp, "#time(s),x(m),y(m),z(m),roll,pitch,yaw,gt_x(m),gt_y(m),gt_z(m),roll,pitch,yaw,bias_gx,bias_gy,bias_gz,bias_wx,bias_wy,bias_wz\n");
  string radar_vel_path = output_path  + "vel_esti" + ".txt";
  path_list[0] = radar_vel_path;

  string radar_vel_filter_path = output_path + "vel_filter.txt";
  path_list[1] = radar_vel_filter_path;


  string radar_icp_path = output_path + bag_name + "_icp.txt";
  path_list[2] = radar_icp_path;


  string radar_gicp_path = output_path + bag_name + "_gicp.txt";
  path_list[3] = radar_gicp_path;


  string radar_ndt_path = output_path + bag_name + "_ndt.txt";
  path_list[4] = radar_ndt_path;
 

  string radar_apdgicp_path = output_path + bag_name + "_apdgicp.txt";
  path_list[5] = radar_apdgicp_path;

  string gt_path = output_path + bag_name + "_gt.txt";
  path_list[6] = gt_path;

  std::thread posegraph_odom {process_data, path_list}; // pose odom construction
  std::thread posegraph_slam {process_pg}; // pose graph construction
	std::thread lc_detection {process_lcd}; // loop closure detection 
  std::thread icp_calculation {process_LC_icp}; // loop constraint calculation via icp 
	std::thread viz_map {process_viz_map}; // visualization - map (low frequency because it is heavy)
	std::thread viz_path {process_viz_path}; // visualization - path (high frequency)
  ros::spin();

  posegraph_odom.join();
  posegraph_slam.join();
  lc_detection.join();
  icp_calculation.join();
  viz_map.join();
  viz_path.join();

  return 0;
}

void callbackIMU(const sensor_msgs::ImuConstPtr &imu_msg)
{
  mutex_rawdata.lock();
  double dt = 0.01;
  if (std::fabs(last_imu.dt) > 1.0e-6)
  {
    dt = (imu_msg->header.stamp - last_imu.time_stamp).toSec();
  }
  last_imu = ImuDataStamped(imu_msg, dt);
  queue_imu.push(last_imu);
  mutex_rawdata.unlock();
}

void callbackRadarScan(const radar_msgs::RadarTargetConstPtr &radar_msg, int radar_no)
{
  if (radar_no < 0 || radar_no > 5)
  {
    std::cout << "wrong radar no. !!!" << std::endl;
  }
  mutex_rawdata.lock();
  queue_radar[radar_no].push(*radar_msg);
  mutex_rawdata.unlock();
}

void callbackGtPose(const sensor_msgs::NavSatFixConstPtr &gt_msg)
{
  mutex_rawdata.lock();
  queue_odom_pose.push(*gt_msg);
  mutex_rawdata.unlock();
}

void callbackGtTwist(const nav_msgs::OdometryConstPtr &gt_msg)
{
  mutex_rawdata.lock();
  queue_odom_twist.push(*gt_msg);
  mutex_rawdata.unlock();
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

