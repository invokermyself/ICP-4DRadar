

#include <pcl/point_types.h>

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
