
# Radar Odometry

## need to change

param in radar_odometry.launch
R_enu_radar: trans from radar to body(eg. 右前上)

## dependency

see .gitmodules

## how to use

roslaunch odometry_4dradar radar_odometry.launch

## output

topic:"/radar_path","/radar_vel"
