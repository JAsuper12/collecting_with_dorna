#! /usr/bin/env python

import rospy
import actionlib
from actions_quiz.msg import CustomActionMsgAction, CustomActionMsgFeedback, CustomActionMsgResult
from geometry_msgs.msg import Twist

class TakeoffAndLand:
    def __init__(self):
        self.vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size = 1)
        self.vel_msg = Twist()
        self.action_server = actionlib.SimpleActionServer("/action_custom_msg_as", CustomActionMsgAction, self.goal_callback, False)
        self.action_server.start()
        rospy.loginfo("Action server active")
        self.rate = rospy.Rate(1)
        self.feedback = CustomActionMsgFeedback()
        self.result = CustomActionMsgResult()
        self.ctrl_c = False
        self.publish_once_in_cmd_vel()

    def publish_once_in_cmd_vel(self):
        while not self.ctrl_c:
            connections = self.vel_pub.get_num_connections()

            if connections > 0:
                self.vel_pub.publish(self.vel_msg)
                rospy.loginfo("Cmd Published")
                break

            else:
                self.rate.sleep()

    def takeoff(self):
        self.seconds = 0
        rospy.loginfo("Taking off")
        self.feedback.feedback = "taking off"

        while self.seconds <= 5:
            self.check_preemted()
            self.action_server.publish_feedback(self.feedback)
            self.vel_msg.linear.z = 1
            self.vel_pub.publish(self.vel_msg)
            self.rate.sleep()
            self.seconds += 1

    def land(self):
        self.seconds = 0
        rospy.loginfo("Landing")
        self.feedback.feedback = "landing"

        while self.seconds <= 5:
            self.check_preemted()
            self.action_server.publish_feedback(self.feedback)
            self.vel_msg.linear.z = -1
            self.vel_pub.publish(self.vel_msg)
            self.rate.sleep()
            self.seconds += 1

    def stop(self):
        rospy.loginfo("Stopping.")
        self.vel_msg.linear.z = 0
        self.vel_pub.publish(self.vel_msg)
        self.rate.sleep()

    def goal_callback(self, goal):
        self.motion = goal.goal
        self.success = True

        if self.motion == "TAKEOFF":
            self.takeoff()
            self.stop()

        elif self.motion == "LAND":
            self.land()

        if self.success:
            rospy.loginfo("Action completed.")
            self.action_server.set_succeeded

    def check_preemted(self):
        if self.action_server.is_preempt_requested():
                rospy.loginfo("The goal has been cancelled")
                self.action_server.set_preempted(self.result)
                self.success = False

if __name__ == "__main__":
    rospy.init_node("drone_takeoff_land_action_server")
    TakeoffAndLand()
    rospy.spin()
