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
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl_conversions/pcl_conversions.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <sensor_msgs/PointCloud2.h>
#include <ros/ros.h>
#include <rosbag/bag.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <boost/thread/thread.hpp>
#include <math.h>

#include <pcl/pcl_macros.h>
#include "../include/userdefine.h"
using namespace std;

// #define DEBUG_TEST
// #define USE_STATIC_POINTS
#define USE_BIN_FILES // select the input source
// #define USE_ICP_RESULT
// #define USE_LOCAL_MAP
#define RESULT_GAP 1

double para_q[4] = {0, 0, 0, 1};
double para_t[3] = {0, 0, 0};
double output_time = 0;

Eigen::Vector3d t(para_t);
Eigen::Vector4d pos({0, 0, 0, 1});
Eigen::Matrix<double, 3, 3> Rtrans;
Eigen::MatrixXd currOdom;

void pointAssociateToMap(PointType const *const pi, PointType *const po)
{
	Eigen::Vector3d point_curr(pi->x, pi->y, pi->z);
	Eigen::Vector3d point_w = Rtrans * point_curr + t;
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

std::vector<float> read_radar_data(const std::string radar_data_path)
{
	if (boost::filesystem::exists(radar_data_path))
	{

		std::ifstream radar_data_file(radar_data_path, std::ifstream::in | std::ifstream::binary);
		radar_data_file.seekg(0, std::ios::end);
		const size_t num_elements = radar_data_file.tellg() / sizeof(float);
		radar_data_file.seekg(0, std::ios::beg);
		std::vector<float> radar_data_buffer(num_elements);
		radar_data_file.read(reinterpret_cast<char *>(&radar_data_buffer[0]), num_elements * sizeof(float));
		return radar_data_buffer;
	}
	else
	{
		std::vector<float> radar_data_buffer(0);
		return radar_data_buffer;
	}
}

// v*cos(beta) = A*cos(arfa+b)
double fitSineRansac(const std::vector<RadarPoint_Info2> &points,
					 double &A_best,
					 double &b_best,
					 int iterations = 200,
					 double sigma = 0.5)
{
	uint16_t num = points.size();
	double bestScore = 0.0;

	for (int i = 0; i < iterations; i++)
	{
		std::random_device rd;
		std::default_random_engine eng(rd());
		std::uniform_int_distribution<int> distr(0, num);
		int i1 = distr(eng);
		int i2 = distr(eng);

		const RadarPoint_Info2 p1 = points[i1];
		const RadarPoint_Info2 p2 = points[i2];

		// form the model by two points
		double k = (p1.v_r * cos(DEG2RAD(p1.beta))) / (p2.v_r * cos(DEG2RAD(p2.beta)));
		double b = atan((cos(DEG2RAD(p1.arfa)) - k * cos(DEG2RAD(p2.arfa))) / (sin(DEG2RAD(p1.arfa)) - k * sin(DEG2RAD(p2.arfa))));
		double A = cos(DEG2RAD(p1.beta)) * p1.v_r / cos((DEG2RAD(p1.arfa)) + b);
		double score = 0;

		for (int j = 0; j < num; j++)
		{
			RadarPoint_Info2 pj = points[j];
			double delta = (cos(DEG2RAD(pj.beta)) * pj.v_r) - (A * cos(DEG2RAD(pj.arfa) + b));
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

int main(int argc, char **argv)
{

	ros::init(argc, argv, "icp_radar");
	ros::NodeHandle n("~");
	std::string dataset_folder, sequence_number;
	std::stringstream velocity_odom_path;
	std::stringstream icp_odom_path;
	std::stringstream icp_map_path;
	std::stringstream pcl_info_path;

#ifdef DEBUG_TEST
	dataset_folder = "/media/invoker/Toshiba_1T/experiment-20221211/round/";
	// dataset_folder = "/home/invoker/suzhouluce_test/";
	sequence_number = "01";
#else
	n.getParam("dataset_folder", dataset_folder);
	n.getParam("sequence_number", sequence_number);
#endif

	velocity_odom_path << dataset_folder << "radar";
	if (access(velocity_odom_path.str().c_str(), 0))
	{
		std::cout << "folder dose not exist!! Will create a new one!" << std::endl;
		std::string commmand = "sudo mkdir " + velocity_odom_path.str();
		system(commmand.c_str());
	}
	// velocity_odom_path << '/' + sequence_number;
	// if (access(velocity_odom_path.str().c_str(), 0))
	// {
	// 	std::cout << "folder dose not exist!! Will create a new one!" << std::endl;
	// 	std::string commmand = "sudo mkdir " + velocity_odom_path.str();
	// 	system(commmand.c_str());
	// }
	velocity_odom_path << "/"
					   << "velocity.txt";
	std::ofstream velocity_odom(velocity_odom_path.str(), std::ios::trunc);
	velocity_odom.setf(std::ios::dec, std::ios::floatfield);
	velocity_odom.precision(15);

	icp_odom_path << dataset_folder << "radar/"
				  << "icp.txt";
	std::ofstream icp_odom(icp_odom_path.str(), std::ios::trunc);
	icp_odom.setf(std::ios::dec, std::ios::floatfield);
	icp_odom.precision(15);

	icp_map_path << dataset_folder << "radar/"
				 << "icp_map.txt";
	std::ofstream icp_map(icp_map_path.str(), std::ios::trunc);
	icp_map.setf(std::ios::dec, std::ios::floatfield);
	icp_map.precision(15);

	pcl_info_path << dataset_folder << "radar/"
				 << "pcl_info.txt";
	std::ofstream pcl_info(pcl_info_path.str(), std::ios::trunc);
	pcl_info.setf(std::ios::dec, std::ios::floatfield);

#ifndef USE_ICP_RESULT
	string outputfile = "/home/invoker/catkin_ws/src/ICP-4DRadar/output_result.csv";
	FILE *fp = fopen(outputfile.c_str(), "w+");
	// 输出
	fprintf(fp, "#time(s),Rtrans00,Rtrans01,Rtrans02,Rtrans03,Rtrans10,Rtrans11,Rtrans12,Rtrans13,Rtrans20,Rtrans21,Rtrans22,Rtrans23,Rtrans00,Rtrans00,Rtrans00,Rtrans00,score,A,b\n");
#else
	string resultfile = "/home/invoker/catkin_ws/src/ICP-4DRadar/output_result.csv";
	// 解析 CSV
	ifstream file(resultfile.c_str());
	string value;

	Eigen::Matrix<double, 20, 1> firstline = Eigen::Matrix<double, 20, 1>::Zero();
	for (int i = 0; i < 19; i++)
	{
		getline(file, value, ',');
		firstline(i) = atof(value.c_str()); // 转为浮点型
	}
	getline(file, value, '\n'); // 换行
	firstline(19) = atof(value.c_str());
#endif

	ros::Publisher pubRadarCloudSurround = n.advertise<sensor_msgs::PointCloud2>("/radar_cloud_surround", 100);
	ros::Publisher pubRadarCloudLocal = n.advertise<sensor_msgs::PointCloud2>("/radar_cloud_local", 100);
	ros::Publisher pubRadarSubMap = n.advertise<sensor_msgs::PointCloud2>("/radar_cloud_submap", 100);
	ros::Publisher pubOdomGT = n.advertise<nav_msgs::Odometry>("/radar_odom", 100);
	nav_msgs::Odometry odomGT;
	odomGT.header.frame_id = "/camera_init";
	odomGT.child_frame_id = "/ground_truth";

	ros::Publisher pubPathGT = n.advertise<nav_msgs::Path>("/radar_path", 5);
	nav_msgs::Path pathGT;
	pathGT.header.frame_id = "/camera_init";

	pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_src_in(new pcl::PointCloud<pcl::PointXYZI>);
	pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_src_out(new pcl::PointCloud<pcl::PointXYZI>);
	pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_tar_in(new pcl::PointCloud<pcl::PointXYZI>);
	pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_tar_out(new pcl::PointCloud<pcl::PointXYZI>);
	pcl::PointCloud<pcl::PointXYZI>::Ptr Final(new pcl::PointCloud<pcl::PointXYZI>);
	pcl::PointCloud<pcl::PointXYZI>::Ptr RadarCloudSurround(new pcl::PointCloud<pcl::PointXYZI>);
	pcl::PointCloud<pcl::PointXYZI>::Ptr RadarCloudLocal(new pcl::PointCloud<pcl::PointXYZI>);
	pcl::PointCloud<pcl::PointXYZI>::Ptr RadarSubMap(new pcl::PointCloud<pcl::PointXYZI>);
	pcl::PointCloud<pcl::PointXYZI>::Ptr RadarSubMapLocal(new pcl::PointCloud<pcl::PointXYZI>);
	std::vector<pcl::PointCloud<pcl::PointXYZI>> RadarLocalMap;
	pcl::PointCloud<pcl::PointXYZI>::Ptr LocalMap_cloud_src(new pcl::PointCloud<pcl::PointXYZI>);
	pcl::PointCloud<pcl::PointXYZI>::Ptr LocalMap_cloud_tar(new pcl::PointCloud<pcl::PointXYZI>);
	std::vector<Eigen::Vector3d> Vxyz_result;
	std::vector<Eigen::Matrix<double, 3, 3>> Icp_Rtrans_result;
	std::vector<Eigen::Vector3d> Icp_Ttrans_result;
	std::vector<Eigen::Matrix<double, 3, 3>> Icp_map_Rtrans_result;
	std::vector<Eigen::Vector3d> Icp_map_Ttrans_result;
	std::vector<Eigen::Matrix<double, 4, 4>> Odom_result;
	std::vector<Eigen::Matrix<double, 4, 4>> SubMap_Odom_result;

	pcl::PointXYZI pointSel;
	std::vector<int> mapping;

	std::cout << "Reading sequences " << sequence_number << " from " << dataset_folder << '\n';

	Eigen::Matrix3d R_transform;
	R_transform << 0, 0, 1, -1, 0, 0, 0, -1, 0;
	Eigen::Quaterniond q_transform(R_transform);

	Rtrans = Eigen::Matrix<double, 3, 3>::Identity();
	currOdom = Eigen::Matrix<double, 4, 4>::Identity();

	//  q = q_transform * q;
	//  t = q_transform * t;

	std::size_t order = 0;
	std::size_t submap_cnt = 0;
	std::size_t mapstart_order = 0;
	ros::Rate r(100.0);
	std::string cloud_src_file, cloud_tar_file;

	SubMap_Odom_result.push_back(currOdom);

	while (ros::ok())
	{
		std::stringstream radar_currdata_path;
		std::stringstream radar_lastdata_path;
		std::stringstream radar_nextdata_path;

#ifdef USE_PCD_FILES
		radar_currdata_path << dataset_folder << "train/sequences/" + sequence_number + "/pcd/"
							<< std::setfill('0') << std::setw(5) << order << ".pcd";

		if (order == 0)
		{
			radar_lastdata_path << dataset_folder << "train/sequences/" + sequence_number + "/pcd/"
								<< std::setfill('0') << std::setw(5) << order << ".pcd";
		}
		else
		{
			radar_lastdata_path << dataset_folder << "train/sequences/" + sequence_number + "/pcd/"
								<< std::setfill('0') << std::setw(5) << order - 1 << ".pcd";
		}
		radar_nextdata_path << dataset_folder << "train/sequences/" + sequence_number + "/pcd/"
							<< std::setfill('0') << std::setw(5) << order + 1 << ".pcd";

		cloud_tar_file = radar_lastdata_path.str();
		cloud_src_file = radar_currdata_path.str();

		if (pcl::io::loadPCDFile<pcl::PointXYZ>(cloud_src_file, *cloud_src_in) == -1) //*打开点云文件
		{
			PCL_ERROR("Couldn't read file cloud_in.pcd\n");
			return (-1);
		}

		if (pcl::io::loadPCDFile<pcl::PointXYZ>(cloud_tar_file, *cloud_tar_in) == -1) //*打开点云文件
		{
			PCL_ERROR("Couldn't read file cloud_in.pcd\n");
			return (-1);
		}
#endif

#ifdef USE_BIN_FILES
		radar_currdata_path << dataset_folder << "data/" << "radar_pointcloud_"
							<< order << ".bin";

		if (order == 0)
		{
			radar_lastdata_path << dataset_folder << "data/" << "radar_pointcloud_"
								<< order << ".bin";
		}
		else
		{
			radar_lastdata_path << dataset_folder << "data/" << "radar_pointcloud_"
								<< order - 1 << ".bin";
		}
		radar_nextdata_path << dataset_folder << "data/" << "radar_pointcloud_"
							<< order + 1 << ".bin";	

		std::vector<float> radarcurr_data = read_radar_data(radar_currdata_path.str());

		std::vector<float> radarlast_data = read_radar_data(radar_lastdata_path.str());
		std::cout << "totally " << radarcurr_data.size() / 5.0 << " points in " << std::setfill('0')
				  << std::setw(5) << order << ".bin" << '\n';

		pcl_info << radarcurr_data.size() / 5.0 << std::endl;

		// /* add cloud_src */
		// for (std::size_t i = 0; i < radarcurr_data.size(); i += 7)
		// {
		// 	RadarPoint_Info1 point_src;
		// 	point_src.point_pos.x = radarcurr_data[i];
		// 	point_src.point_pos.y = radarcurr_data[i + 1];
		// 	point_src.point_pos.z = radarcurr_data[i + 2];
		// 	point_src.RCS = radarcurr_data[i + 3];
		// 	point_src.v_r = radarcurr_data[i + 4];
		// 	point_src.v_r_compensated = radarcurr_data[i + 5];
		// 	point_src.time = radarcurr_data[i + 6];
		// 	cloud_src_in->push_back(point_src.point_pos);
		// }

		// /* add cloud_tar */
		// for (std::size_t i = 0; i < radarlast_data.size(); i += 7)
		// {
		// 	RadarPoint_Info1 point_tar;
		// 	point_tar.point_pos.x = radarlast_data[i];
		// 	point_tIcp_Rtrans_resultar.v_r = radarlast_data[i + 4];
		// 	point_tar.v_r_compensated = radarlast_data[i + 5];
		// 	point_tar.time = radarlast_data[i + 6];
		// 	cloud_tar_in->push_back(point_tar.point_pos);
		// }

		/* add cloud_src */
		// int PointsNum = radarcurr_data.size() / 8.0;
		int PointsNum = radarcurr_data.size() / 5.0;
		std::vector<RadarPoint_Info2> point_src_cloud(PointsNum);
		std::vector<RadarPoint_Info2> static_point_src_cloud;
		std::vector<RadarPoint_Info2> dynamic_point_src_cloud;
		for (std::size_t i = 0; i < radarcurr_data.size(); i += 5)
		{
			RadarPoint_Info2 point_src;
			// point_src.point_pos.x = radarcurr_data[i];
			// point_src.point_pos.y = radarcurr_data[i + 1];
			// point_src.point_pos.z = radarcurr_data[i + 2];
			// point_src.v_r = radarcurr_data[i + 3];
			// point_src.distance = radarcurr_data[i + 4];
			// point_src.point_pos.intensity = radarcurr_data[i + 5];
			// // point_src.power = radarcurr_data[i + 5];
			// point_src.arfa = radarcurr_data[i + 6];
			// point_src.beta = radarcurr_data[i + 7];
			// point_src_cloud[i / 8] = point_src;


			point_src.point_pos.x = radarcurr_data[i];
			point_src.point_pos.y = radarcurr_data[i + 1];
			point_src.point_pos.z = radarcurr_data[i + 2];
			point_src.point_pos.intensity = radarcurr_data[i + 3];
			point_src.v_r = radarcurr_data[i + 4];
			point_src.distance = std::sqrt(radarcurr_data[i]*radarcurr_data[i] + \
			radarcurr_data[i+1]*radarcurr_data[i+1] + radarcurr_data[i+2]*radarcurr_data[i+2]);
			
			// point_src.power = radarcurr_data[i + 5];
			point_src.arfa = std::atan2(point_src.point_pos.y,point_src.point_pos.x)*180/M_PI;
			point_src.beta = std::asin(point_src.point_pos.z/point_src.distance)*180/M_PI;
			point_src_cloud[i / 5] = point_src;
		}

		double A_src = 0;
		double b_src = 0;
		fitSineRansac(point_src_cloud, A_src, b_src, PointsNum * 0.2);

		// devide the points
		for (int i = 0; i < point_src_cloud.size(); i++)
		{
			double delta = (cos(DEG2RAD(point_src_cloud[i].beta)) * point_src_cloud[i].v_r) -
						   (A_src * cos(DEG2RAD(point_src_cloud[i].arfa) + b_src));
			if (delta > 0.2)
			{
				dynamic_point_src_cloud.push_back(point_src_cloud[i]);
			}
			else
			{
				static_point_src_cloud.push_back(point_src_cloud[i]);
			}
#ifndef USE_STATIC_POINTS
			cloud_src_in->push_back(point_src_cloud[i].point_pos);
#endif
		}

		// use the static points to esitimate velocity
		int StaticPointNum = static_point_src_cloud.size();

		Eigen::MatrixX3d K(StaticPointNum, 3);
		Eigen::VectorXd Vr(StaticPointNum);
		Eigen::VectorXd Vxyz;

		for (int i = 0; i < StaticPointNum; i++)
		{
			double Ki0 = cos(DEG2RAD(static_point_src_cloud[i].arfa)) * cos(DEG2RAD(static_point_src_cloud[i].beta));
			double Ki1 = sin(DEG2RAD(static_point_src_cloud[i].arfa)) * cos(DEG2RAD(static_point_src_cloud[i].beta));
			double Ki2 = sin(DEG2RAD(static_point_src_cloud[i].beta));
			K(i, 0) = Ki0;
			K(i, 1) = Ki1;
			K(i, 2) = Ki2;
			Vr(i) = static_point_src_cloud[i].v_r;
#ifdef USE_STATIC_POINTS
			cloud_src_in->push_back(static_point_src_cloud[i].point_pos);
#endif
		}
		Vxyz = (K.transpose() * K).inverse() * K.transpose() * Vr;
		Vxyz_result.push_back(Vxyz);
		cout << Vxyz(0) << '\t' << Vxyz(1) << '\t' << Vxyz(2) << endl;

		/* add cloud_tar */
		PointsNum = radarlast_data.size() / 5.0;
		std::vector<RadarPoint_Info2> point_tar_cloud(PointsNum);
		std::vector<RadarPoint_Info2> static_point_tar_cloud;
		std::vector<RadarPoint_Info2> dynamic_point_tar_cloud;
		for (std::size_t i = 0; i < radarlast_data.size(); i += 5)
		{
			RadarPoint_Info2 point_tar;
			// point_tar.point_pos.x = radarlast_data[i];
			// point_tar.point_pos.y = radarlast_data[i + 1];
			// point_tar.point_pos.z = radarlast_data[i + 2];
			// point_tar.v_r = radarlast_data[i + 3];
			// point_tar.distance = radarlast_data[i + 4];
			// point_tar.point_pos.intensity = radarlast_data[i + 5];
			// // point_tar.power = radarlast_data[i + 5];
			// point_tar.arfa = radarlast_data[i + 6];
			// point_tar.beta = radarcurr_data[i + 7];
			// point_tar_cloud[i / 8] = point_tar;

			point_tar.point_pos.x = radarlast_data[i];
			point_tar.point_pos.y = radarlast_data[i + 1];
			point_tar.point_pos.z = radarlast_data[i + 2];
			point_tar.point_pos.intensity = radarlast_data[i + 3];
			point_tar.v_r = radarlast_data[i + 4];
			point_tar.distance = std::sqrt(radarlast_data[i]*radarlast_data[i] + \
			radarlast_data[i+1]*radarlast_data[i+1] + radarlast_data[i+2]*radarlast_data[i+2]);
			
			// point_src.power = radarcurr_data[i + 5];
			point_tar.arfa = std::atan2(point_tar.point_pos.y,point_tar.point_pos.x)*180/M_PI;
			point_tar.beta = std::asin(point_tar.point_pos.z/point_tar.distance)*180/M_PI;
			point_tar_cloud[i / 5] = point_tar;
		}
		double A_tar = 0;
		double b_tar = 0;
		fitSineRansac(point_tar_cloud, A_tar, b_tar, PointsNum * 0.2);

		// devide the points
		for (int i = 0; i < point_tar_cloud.size(); i++)
		{
			double delta = (cos(DEG2RAD(point_tar_cloud[i].beta)) * point_tar_cloud[i].v_r) -
						   (A_tar * cos(DEG2RAD(point_tar_cloud[i].arfa) + b_tar));
			if (delta > 0.2)
			{
				dynamic_point_tar_cloud.push_back(point_tar_cloud[i]);
			}
			else
			{
				static_point_tar_cloud.push_back(point_tar_cloud[i]);
			}
#ifndef USE_STATIC_POINTS
			cloud_tar_in->push_back(point_tar_cloud[i].point_pos);
#endif
		}

		StaticPointNum = static_point_tar_cloud.size();
		for (int i = 0; i < StaticPointNum; i++)
		{
#ifdef USE_STATIC_POINTS
			cloud_tar_in->push_back(static_point_tar_cloud[i].point_pos);
#endif
		}

#endif

		/*** 	remove nan points		***/
		// pcl::removeNaNFromPointCloud(*cloud_src_in, *cloud_src_out, mapping);
		// pcl::removeNaNFromPointCloud(*cloud_tar_in, *cloud_tar_out, mapping);
		// std::stringstream filepath;
		// filepath << "/home/invoker/data_tju/velodyne/sequences/00/static_pcl/" << std::setfill('0')
		// 		 << std::setw(5) << order << ".pcd";
		// pcl::io::savePCDFileASCII(filepath.str(), *cloud_src_in);

		if (cloud_tar_in->size() && cloud_src_in->size())
		{

#ifndef USE_ICP_RESULT
			// use ICP method and set the parameters of ICP
			pcl::IterativeClosestPoint<pcl::PointXYZI, pcl::PointXYZI> icp;
			icp.setInputSource(cloud_src_in);
			icp.setInputTarget(cloud_tar_in);
			// icp.setMaximumIterations(200);
			icp.align(*Final);

			std::cout << "has converged:" << icp.hasConverged() << " score: " << icp.getFitnessScore() << std::endl;

			// output the transformation matrix
			std::cout << icp.getFinalTransformation() << std::endl;
			double score = icp.getFitnessScore();
			Eigen::Matrix<double, 4, 4> icp_result = icp.getFinalTransformation().cast<double>();

#else
			Eigen::Matrix<double, 20, 1> data = Eigen::Matrix<double, 20, 1>::Zero();
			Eigen::Matrix<double, 4, 4> icp_result;
			for (int i = 0; i < 19; i++)
			{
				getline(file, value, ',');
				data(i) = atof(value.c_str()); // 转为浮点型
			}
			getline(file, value, '\n'); // 换行
			data(19) = atof(value.c_str());
			for (int i = 0; i < 4; i++)
			{
				for (int j = 0; j < 4; j++)
				{
					icp_result(i, j) = data(4 * i + j + 1, 1);
				}
			}
#endif
			Eigen::Matrix<double, 4, 4> Tradar2nav = Eigen::Matrix<double, 4, 4>::Identity();
			Tradar2nav(1, 1) = -1;
			Tradar2nav(2, 2) = -1;
			// icp_result = Tradar2nav * icp_result * Tradar2nav.inverse();
			Eigen::Matrix<double, 3, 3> odom_Rtrans = icp_result.topLeftCorner<3, 3>();
			Eigen::Vector3d odom_Ptrans = icp_result.topRightCorner<3, 1>();
			if (1)
			{
				Icp_Rtrans_result.push_back(odom_Rtrans);
				Icp_Ttrans_result.push_back(odom_Ptrans);
			}
			currOdom = currOdom * icp_result;
			Odom_result.push_back(currOdom);
			t = t + Rtrans * odom_Ptrans;

			Rtrans = Rtrans * odom_Rtrans;
			Eigen::Quaterniond q(Rtrans);
			q.normalize();

			odomGT.header.stamp = ros::Time().now();
			odomGT.pose.pose.orientation.x = q.x();
			odomGT.pose.pose.orientation.y = q.y();
			odomGT.pose.pose.orientation.z = q.z();
			odomGT.pose.pose.orientation.w = q.w();
			odomGT.pose.pose.position.x = t(0);
			odomGT.pose.pose.position.y = t(1);
			odomGT.pose.pose.position.z = t(2);
			pubOdomGT.publish(odomGT);

			geometry_msgs::PoseStamped poseGT;
			poseGT.header = odomGT.header;
			poseGT.pose = odomGT.pose.pose;
			pathGT.header.stamp = odomGT.header.stamp;
			pathGT.poses.push_back(poseGT);
			pubPathGT.publish(pathGT);

			// publish cloudsurround
			RadarCloudSurround->clear();
			RadarCloudLocal->clear();
			size_t PointNum = cloud_src_in->size();
			for (int iterCount = 0; iterCount < PointNum; iterCount++)
			{
				RadarCloudLocal->push_back(cloud_src_in->points[iterCount]);
				pointAssociateToMap(&cloud_src_in->points[iterCount], &pointSel);
				RadarCloudSurround->push_back(pointSel);
				RadarSubMap->push_back(pointSel);
			}
			submap_cnt += 1;

			if (submap_cnt == 20)
			{
				// Eigen::Vector4f centroid;
				// pcl::compute3DCentroid(*RadarSubMap,centroid);
				SubMap_Odom_result.push_back(currOdom);
				Eigen::Vector3d t_submap = SubMap_Odom_result.back().topRightCorner<3, 1>() -
										   (*(++SubMap_Odom_result.rbegin())).topRightCorner<3, 1>();
				// t_submap(0) = centroid(0);
				// t_submap(1) = centroid(1);
				// t_submap(2) = centroid(2);
				t_submap.normalize();
				t_submap = t_submap * 100 + t;
				Eigen::Matrix<double, 4, 4> wTcurr = *(++SubMap_Odom_result.rbegin());
				// wTcurr(0,3)  = wTcurr(0,3) + 100;
				// wTcurr(1,3)  = wTcurr(1,3);
				// wTcurr(2,3)  = wTcurr(2,3);
				PointNum = RadarSubMap->size();
				for (int iterCount = 0; iterCount < PointNum; iterCount++)
				{
					pointAssociateToSubMap(&RadarSubMap->points[iterCount], &pointSel, wTcurr);
					// pointSel.x -= 100;
					RadarSubMapLocal->push_back(pointSel);
				}

				sensor_msgs::PointCloud2 RadarCloudSubMapMsg;
				pcl::toROSMsg(*RadarSubMapLocal, RadarCloudSubMapMsg);
				RadarCloudSubMapMsg.header.stamp = ros::Time().now();
				RadarCloudSubMapMsg.header.frame_id = "/camera_init";
				pubRadarSubMap.publish(RadarCloudSubMapMsg);
				RadarSubMap->clear();
				RadarSubMapLocal->clear();

				// odomGT.header.stamp = ros::Time().now();
				// odomGT.pose.pose.orientation.x = q.x();
				// odomGT.pose.pose.orientation.y = q.y();
				// odomGT.pose.pose.orientation.z = q.z();
				// odomGT.pose.pose.orientation.w = q.w();
				// odomGT.pose.pose.position.x = t(0);
				// odomGT.pose.pose.position.y = t(1);
				// odomGT.pose.pose.position.z = t(2);
				// pubOdomGT.publish(odomGT);

				submap_cnt = 0;
			}

// add local map
// RadarLocalMap.push_back(*RadarCloudSurround);
#ifdef USE_LOCAL_MAP

			if ((order - mapstart_order % 30) < 15)
			{
				size_t PointNum = RadarCloudSurround->size();
				for (int iterCount = 0; iterCount < PointNum; iterCount++)
				{
					LocalMap_cloud_tar->push_back(RadarCloudSurround->points[iterCount]);
				}
			}
			else
			{
				size_t PointNum = RadarCloudSurround->size();
				for (int iterCount = 0; iterCount < PointNum; iterCount++)
				{
					LocalMap_cloud_src->push_back(RadarCloudSurround->points[iterCount]);
				}
			}

			if ((order > 20) && ((order + 1) % 15 == 0))
			{
				// use ICP method and set the parameters of ICP
				pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp2;
				icp2.setInputSource(LocalMap_cloud_src);
				icp2.setInputTarget(LocalMap_cloud_tar);
				// icp.setMaximumIterations(200);
				icp2.align(*Final);
				// output the transformation matrix
				Eigen::Matrix<double, 4, 4> icp_map_result = icp2.getFinalTransformation().cast<double>();
				Eigen::Matrix<double, 3, 3> odom_map_Rtrans = icp_map_result.topLeftCorner<3, 3>();
				Eigen::Vector3d odom_map_Ptrans = icp_map_result.topRightCorner<3, 1>();

				// LocalMap_cloud_tar = LocalMap_cloud_src;
				LocalMap_cloud_tar->clear();
				size_t PointNum = LocalMap_cloud_src->size();
				for (int iterCount = 0; iterCount < PointNum; iterCount++)
				{
					LocalMap_cloud_tar->push_back(LocalMap_cloud_src->points[iterCount]);
				}

				LocalMap_cloud_src->clear();

				Icp_map_Rtrans_result.push_back(odom_map_Rtrans);
				Icp_map_Ttrans_result.push_back(odom_map_Ptrans);

				mapstart_order += 15;
			}
#endif

			sensor_msgs::PointCloud2 RadarCloudSurround2;
			pcl::toROSMsg(*RadarCloudSurround, RadarCloudSurround2);
			RadarCloudSurround2.header.stamp = ros::Time().now();
			RadarCloudSurround2.header.frame_id = "/camera_init";
			pubRadarCloudSurround.publish(RadarCloudSurround2);

			sensor_msgs::PointCloud2 RadarCloudLocalMsg;
			pcl::toROSMsg(*RadarCloudLocal, RadarCloudLocalMsg);
			RadarCloudLocalMsg.header.stamp = ros::Time().now();
			RadarCloudLocalMsg.header.frame_id = "/camera_init";
			pubRadarCloudLocal.publish(RadarCloudLocalMsg);

			cloud_src_in->clear();
			cloud_tar_in->clear();
#ifndef USE_ICP_RESULT
			fprintf(fp, "%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n",
					output_time, icp_result(0, 0), icp_result(0, 1), icp_result(0, 2), icp_result(0, 3),
					icp_result(1, 0), icp_result(1, 1), icp_result(1, 2), icp_result(1, 3),
					icp_result(2, 0), icp_result(2, 1), icp_result(2, 2), icp_result(2, 3),
					icp_result(3, 0), icp_result(3, 1), icp_result(3, 2), icp_result(3, 3),
					score, A_src, b_src);
#endif
			output_time += 1.0;
		}

		if (boost::filesystem::exists(radar_nextdata_path.str()))
		{
			order = order + 1;

			r.sleep();
		}
		else
		{
			break;
		}
	}

	/***	visual present   ***/

	/*
	pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));

	int view_port_1 = 1;
	viewer->createViewPort(0.0, 0.0, 0.5, 1.0, view_port_1);
	viewer->setBackgroundColor(0, 0, 0, view_port_1);
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color1_1(cloud_src_out, 255, 0, 0);
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color1_2(cloud_tar_out, 0, 255, 0);
	viewer->addPointCloud<pcl::PointXYZ>(cloud_src_out, color1_1, "source", view_port_1);
	viewer->addPointCloud<pcl::PointXYZ>(cloud_tar_out, color1_2, "target", view_port_1);

	int view_port_2 = 2;
	viewer->createViewPort(0.5, 0.0, 1.0, 1.0, view_port_2);
	viewer->setBackgroundColor(0, 0, 0, view_port_2);
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color2_1(Final, 255, 0, 0);
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color2_2(cloud_tar_out, 0, 255, 0);
	viewer->addPointCloud<pcl::PointXYZ>(Final, color2_1, "final", view_port_2);
	viewer->addPointCloud<pcl::PointXYZ>(cloud_tar_out, color2_2, "target2", view_port_2);
	// viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");

	viewer->addText("before ICP", 10, 10, 20, 1, 0, 0, "view_port_1", view_port_1);
	viewer->addText("after ICP", 10, 10, 20, 0, 1, 0, "view_port_2", view_port_2);
	viewer->addCoordinateSystem(1.0);
	viewer->initCameraParameters();

	while (!viewer->wasStopped())
	{
		viewer->spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
	*/

	/*write to the files*/
	Eigen::Vector3d V_sum(0, 0, 0);
	for (int i = 0; i < Vxyz_result.size(); i += RESULT_GAP)
	{
		velocity_odom << Vxyz_result[i](0) << ' ' << Vxyz_result[i](1) << ' '
					  << Vxyz_result[i](2) << std::endl;
		V_sum = V_sum + Vxyz_result[i] * 0.2;
	}
	velocity_odom.close();

	Eigen::Vector3d T_sum(0, 0, 0);
	for (int i = 0; i < Icp_Rtrans_result.size() ; i += RESULT_GAP)
	{
		Eigen::Matrix<double, 3, 3> R_curr;
		Eigen::Vector3d T_curr;
		// R_curr = Icp_Rtrans_result[i] * Icp_Rtrans_result[i + 1] * Icp_Rtrans_result[i + 2];
		// T_curr = Icp_Ttrans_result[i] + Icp_Rtrans_result[i] * Icp_Ttrans_result[i + 1] +
		// 		 Icp_Rtrans_result[i] * Icp_Rtrans_result[i + 1] * Icp_Ttrans_result[i + 2];
		R_curr = Icp_Rtrans_result[i];
		T_curr = Icp_Ttrans_result[i];
		T_sum = T_sum + T_curr;
		icp_odom << R_curr(0, 0) << ' '
				 << R_curr(0, 1) << ' '
				 << R_curr(0, 2) << ' '
				 << T_curr(0) << ' '
				 << R_curr(1, 0) << ' '
				 << R_curr(1, 1) << ' '
				 << R_curr(1, 2) << ' '
				 << T_curr(1) << ' '
				 << R_curr(2, 0) << ' '
				 << R_curr(2, 1) << ' '
				 << R_curr(2, 2) << ' '
				 << T_curr(2) << std::endl;
	}
	icp_odom.close();

	for (int i = 0; i < Icp_map_Rtrans_result.size(); i += 1)
	{
		Eigen::Matrix<double, 3, 3> R_curr;
		Eigen::Vector3d T_curr;
		R_curr = Icp_map_Rtrans_result[i];
		T_curr = Icp_map_Ttrans_result[i];
		icp_map << R_curr(0, 0) << ' '
				<< R_curr(0, 1) << ' '
				<< R_curr(0, 2) << ' '
				<< T_curr(0) << ' '
				<< R_curr(1, 0) << ' '
				<< R_curr(1, 1) << ' '
				<< R_curr(1, 2) << ' '
				<< T_curr(1) << ' '
				<< R_curr(2, 0) << ' '
				<< R_curr(2, 1) << ' '
				<< R_curr(2, 2) << ' '
				<< T_curr(2) << std::endl;
	}
	icp_map.close();
	pcl_info.close();
#ifndef USE_ICP_RESULT
	fclose(fp);
#endif
	std::cout << "velocity result:" << V_sum << std::endl;
	std::cout << "icp result:" << T_sum << std::endl;
	return (0);
}
