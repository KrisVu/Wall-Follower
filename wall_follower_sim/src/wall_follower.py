#!/usr/bin/env python2

import numpy as np
import math
from scipy import stats

import rospy
from rospy.numpy_msg import numpy_msg
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from sklearn.linear_model import LinearRegression
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker

class WallFollower:
    SCAN_TOPIC = rospy.get_param("wall_follower/scan_topic")
    DRIVE_TOPIC = rospy.get_param("wall_follower/drive_topic")
    SIDE = rospy.get_param("wall_follower/side")
    VELOCITY = rospy.get_param("wall_follower/velocity")
    DESIRED_DISTANCE = rospy.get_param("wall_follower/desired_distance")


    def __init__(self):
        """
        Initialize publishers
        """

        self.pub = rospy.Publisher(self.DRIVE_TOPIC, AckermannDriveStamped, queue_size=10)
        self.sub = rospy.Subscriber(self.SCAN_TOPIC, LaserScan, self.callback)

        self.last_error = 0

        self.rate = rospy.Rate(10)

        self.line_pub = rospy.Publisher('/line', Marker, queue_size=10)



    def callback(self, data):
        """
        Callback for subscriber to scan topic. Uses Lidar scan data to
        publish estimated line for the wall and publish drive commands to
        follow it.
        """
        # Data preprocessing: slicing
        ranges = np.asarray(data.ranges)

        # Converting to Cartesian
        angles = data.angle_min*np.ones(len(ranges)) + (data.angle_increment *  np.arange(1, len(ranges) + 1))

        if self.SIDE == -1:
            side_array = angles[:int((3.0/5)*len(ranges))]
            side_ranges = ranges[:int((3.0/5)*len(ranges))]
        else:
            side_array = angles[int((2.0/5)*len(ranges)):]
            side_ranges = ranges[int((2.0/5)*len(ranges)):]



        x_coords = np.array(side_ranges * np.cos(side_array)).reshape(-1, 1)
        y_coords = np.array(side_ranges * np.sin(side_array)).reshape(-1, 1)

        count = 0
        for i, distance in enumerate(side_ranges):
            if distance >= np.mean(side_ranges)*1.5:
                np.delete(x_coords, i - count)
                np.delete(y_coords, i - count)
                count += 1

        #Ordinary Leas Squares Regression
        line = stats.linregress(x_coords.reshape(-1), y_coords.reshape(-1))
        smoothed_wall = LinearRegression().fit(x_coords, y_coords)
        intercept = smoothed_wall.intercept_
        slope = smoothed_wall.coef_
        self.plot_line(x_coords, (x_coords)*slope + intercept, self.line_pub, color = (1.0, 0.0, 0.0))


        #PD Controller


        actual = np.abs(line.intercept/math.sqrt(1 + line.slope**2))

        Kp = 2.0
        Kd = 0.5
        error =self.SIDE*(self.DESIRED_DISTANCE - actual)
        dedt = (error - self.last_error)

        u = Kp*error + Kd*dedt
        self.last_error = error


        ack_msg = AckermannDriveStamped()
        ack_msg.header.stamp = rospy.Time.now()
        ack_msg.header.frame_id = 'map'
        ack_msg.drive.steering_angle = u
        ack_msg.drive.speed = self.VELOCITY
        self.pub.publish(ack_msg)


    def plot_line(self, x,  y, publisher, color=(1., 0., 0.), frame="/base_link"):
        """
        Publishes a marker in rviz.
        """

        line_strip = Marker()
        line_strip.type = Marker.LINE_STRIP
        line_strip.header.frame_id = frame

        line_strip.scale.x = 0.1
        line_strip.scale.y = 0.1
        line_strip.color.a = 1.
        line_strip.color.r = color[0]
        line_strip.color.b = color[1]
        line_strip.color.g = color[2]

        for xi, yi in zip(np.array(x), np.array(y)):
            p = Point()
            p.x = xi
            p.y = yi
            line_strip.points.append(p)

        publisher.publish(line_strip)


if __name__ == "__main__":
    rospy.init_node('wall_follower')
    wall_follower = WallFollower()
    rospy.spin()
