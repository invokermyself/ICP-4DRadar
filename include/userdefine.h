
#include <sensor_msgs/Imu.h>
#include <pcl/point_types.h>
#include <Eigen/Core>


typedef Eigen::Vector2d Vector2;
typedef Eigen::Vector3d Vector3;
typedef Eigen::Vector4d Vector4;
typedef pcl::PointXYZI PointType;

typedef struct 
{
    pcl::PointXYZ point_pos;
    float RCS;
    float v_r;
    float v_r_compensated;
    float time;
}RadarPoint_Info1;

typedef struct 
{
    pcl::PointXYZI point_pos;
    float v_r;
    float distance;
    // float power;
    float arfa;
    float beta;
}RadarPoint_Info2;

struct ImuDataStamped
{
  ImuDataStamped() : dt{0} {}
  ImuDataStamped(const ros::Time& time_stamp,
                 const std::string frame_id,
                 const double dt,
                 const Vector3& a_b_ib,
                 const Vector3& w_b_ib) :
    time_stamp{time_stamp},
    frame_id{frame_id},
    dt{dt},
    a_b_ib{a_b_ib},
    w_b_ib{w_b_ib}
  {
  }

  ImuDataStamped(const sensor_msgs::ImuConstPtr& imu_msg, const double dt) :
    time_stamp{imu_msg->header.stamp},
    frame_id{imu_msg->header.frame_id},
    dt{dt},
    a_b_ib{Vector3(imu_msg->linear_acceleration.x, imu_msg->linear_acceleration.y, imu_msg->linear_acceleration.z)},
    w_b_ib{Vector3(imu_msg->angular_velocity.x, imu_msg->angular_velocity.y, imu_msg->angular_velocity.z)}
  {
  }

  sensor_msgs::Imu toImuMsg()
  {
    sensor_msgs::Imu imu_msg;
    imu_msg.header.stamp          = time_stamp;
    imu_msg.angular_velocity.x    = w_b_ib.x();
    imu_msg.angular_velocity.y    = w_b_ib.y();
    imu_msg.angular_velocity.z    = w_b_ib.z();
    imu_msg.linear_acceleration.x = a_b_ib.x();
    imu_msg.linear_acceleration.y = a_b_ib.y();
    imu_msg.linear_acceleration.z = a_b_ib.z();
    return imu_msg;
  }

  ros::Time time_stamp;  // ros::Time
  std::string frame_id;  // frame id
  double dt;               // [s]
  Vector3 a_b_ib;        // [m/s^2]
  Vector3 w_b_ib;        // [rad/s]

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct RadarPointCloudType
{
  PCL_ADD_POINT4D;      // position in [m]
  float snr_db;         // CFAR cell to side noise ratio in [dB]
  float v_doppler_mps;  // Doppler velocity in [m/s]
  float noise_db;       // CFAR noise level of the side of the detected cell in [dB]
  float range;          // range in [m]
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct mmWaveCloudType
{
  PCL_ADD_POINT4D;
  union
  {
    struct
    {
      float intensity;
      float velocity;
    };
    float data_c[4];
  };
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};


struct OculiiPointCloudType
{
  PCL_ADD_POINT4D;      // position in [m]
  float Doppler;  // Doppler velocity in [m/s]
  float Range;          // range in [m]
  float Power;
  float Alpha;
  float Beta;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct ColoRadarPointCloudType
{
  PCL_ADD_POINT4D;      // position in [m]
  float intensity;  
  float range;          // range in [m]
  float doppler;  // Doppler velocity in [m/s]
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
