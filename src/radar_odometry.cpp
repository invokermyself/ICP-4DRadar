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

#include <pcl/pcl_macros.h>
#include "userdefine.h"
#include "radarFactor.hpp"
#include "tic_toc.h"
#include "ikd_Tree.h"
#include "radar_ego_velocity_estimator/radar_ego_velocity_estimator.h"

#define MAX_SEARCH_RADIUS 2.0f

using namespace std;

double para_q[4] = {0, 0, 0, 1};
double para_t[3] = {0, 0, 0};
double output_time = 0;

pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreeEdgeFeatureLast(new pcl::KdTreeFLANN<pcl::PointXYZI>());

Eigen::Vector3d t(para_t);
Eigen::Vector4d pos({0, 0, 0, 1});

Eigen::Map<Eigen::Quaterniond> q_last_curr(para_q);
Eigen::Map<Eigen::Vector3d> t_last_curr(para_t);

// Transformation from current frame to world frame
Eigen::Quaterniond q_w_curr(1, 0, 0, 0);
Eigen::Vector3d t_w_curr(0, 0, 0);

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

int main(int argc, char **argv)
{

	ros::init(argc, argv, "radar_odometry");
	ros::NodeHandle n("~");
	std::size_t order = 1;

	std::string dataset_folder;
	n.getParam("dataset_folder", dataset_folder);

	pcl::PointCloud<pcl::PointXYZI>::Ptr featurePointsLast(new pcl::PointCloud<pcl::PointXYZI>);
	pcl::PointCloud<pcl::PointXYZI>::Ptr featurePointsCurr(new pcl::PointCloud<pcl::PointXYZI>);
	pcl::PointCloud<pcl::PointXYZI>::Ptr Final(new pcl::PointCloud<pcl::PointXYZI>);

	std::stringstream radar_odom_path;
	radar_odom_path << "/home/invoker/catkin_ws/src/ICP-4DRadar/radar_odometry.txt";
	std::ofstream odom(radar_odom_path.str(), std::ios::trunc);
	odom.setf(std::ios::dec, std::ios::floatfield);
	odom.precision(15);

	ros::Rate r(100.0);

	while (ros::ok())
	{
		// ceres::LossFunction *loss_function = NULL;
		std::stringstream radar_featurelast_path;
		std::stringstream radar_featurecurr_path;

		radar_featurelast_path << dataset_folder << std::setfill('0') << std::setw(4) << order << "-1.txt";
		radar_featurecurr_path << dataset_folder << std::setfill('0') << std::setw(4) << (order + 1) << "-1.txt";

		std::vector<float> radarlast_data = read_radar_featurePoint(radar_featurelast_path.str());
		std::vector<float> radarcurr_data = read_radar_featurePoint(radar_featurecurr_path.str());
		ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
		ceres::LocalParameterization *q_parameterization =
			new ceres::EigenQuaternionParameterization();
		ceres::Problem::Options problem_options;

		ceres::Problem problem(problem_options);
		problem.AddParameterBlock(para_q, 4, q_parameterization);
		problem.AddParameterBlock(para_t, 3);

		pcl::PointXYZI pointSel;
		std::vector<int> pointSearchInd;
		std::vector<float> pointSearchSqDis;
		featurePointsLast->clear();
		featurePointsCurr->clear();

		for (std::size_t i = 0; i < radarlast_data.size(); i += 3)
		{
			pcl::PointXYZI point;
			point.x = radarlast_data[i];
			point.y = radarlast_data[i + 1];
			point.z = radarlast_data[i + 2];
			featurePointsLast->push_back(point);
		}

		for (std::size_t i = 0; i < radarcurr_data.size(); i += 3)
		{
			pcl::PointXYZI point;
			point.x = radarcurr_data[i];
			point.y = radarcurr_data[i + 1];
			point.z = radarcurr_data[i + 2];
			featurePointsCurr->push_back(point);
		}


		pcl::IterativeClosestPoint<pcl::PointXYZI, pcl::PointXYZI> icp;
		icp.setInputSource(featurePointsCurr);
		icp.setInputTarget(featurePointsLast);
		// icp.setMaximumIterations(200);
		icp.align(*Final);

		std::cout << "has converged:" << icp.hasConverged() << " score: " << icp.getFitnessScore() << std::endl;

		// output the transformation matrix
		std::cout << icp.getFinalTransformation() << std::endl;
		double score = icp.getFitnessScore();
		Eigen::Matrix<double, 4, 4> icp_result = icp.getFinalTransformation().cast<double>();

		kdtreeEdgeFeatureLast->setInputCloud(featurePointsLast);

		for (int i = 0; i < featurePointsCurr->size(); ++i)
		{
			int closestPointInd = -1, minPointInd2 = -1;
			pointSel = featurePointsCurr->points[i];
			kdtreeEdgeFeatureLast->nearestKSearch(pointSel, 2, pointSearchInd, pointSearchSqDis);
			if (*max_element(pointSearchSqDis.begin(), pointSearchSqDis.end()) < MAX_SEARCH_RADIUS)
			{
				Eigen::Vector3d curr_point(featurePointsCurr->points[i].x,
										   featurePointsCurr->points[i].y,
										   featurePointsCurr->points[i].z);
				Eigen::Vector3d last_point_a(featurePointsLast->points[pointSearchInd[0]].x,
											 featurePointsLast->points[pointSearchInd[0]].y,
											 featurePointsLast->points[pointSearchInd[0]].z);
				Eigen::Vector3d last_point_b(featurePointsLast->points[pointSearchInd[1]].x,
											 featurePointsLast->points[pointSearchInd[1]].y,
											 featurePointsLast->points[pointSearchInd[1]].z);

				double s = 1.0;
				ceres::CostFunction *cost_function = RadarEdgeFactor::Create(curr_point, last_point_a, last_point_b, s);
				problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
			}
		}

		TicToc t_solver;
		ceres::Solver::Options options;
		options.linear_solver_type = ceres::DENSE_QR;
		options.max_num_iterations = 4;
		options.minimizer_progress_to_stdout = false;
		ceres::Solver::Summary summary;
		ceres::Solve(options, &problem, &summary);
		printf("solver time %f ms \n", t_solver.toc());

		t_w_curr = t_w_curr + q_w_curr * t_last_curr;
		q_w_curr = q_w_curr * q_last_curr;
		odom << t_w_curr(0) << " "
			 << t_w_curr(1) << " "
			 << t_w_curr(2) << std::endl;

		order++;
		r.sleep();
	}
	odom.close();
}