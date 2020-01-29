import rospy
import actionlib
from actionlib.msg import TestAction, TestFeedback, TestResult
from geometry_msgs.msg import Twist

class MoveSquare:
    def __init__(self):
        self.vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size = 1)
        self.vel_msg = Twist()
        self.action_server = actionlib.SimpleActionServer("/drone_move_square", TestAction, self.goal_callback, False)
        self.action_server.start()
        rospy.loginfo("Action server active")
        self.rate_long = rospy.Rate(1)
        self.rate_short = rospy.Rate(10)
        self.feedback = TestFeedback()
        self.result = TestResult()
        self.ctrl_c = False
        self.publish_once_in_cmd_vel()

    def goal_callback(self, goal):
        self.size = goal.goal
        rospy.loginfo("Goal recieved. Performing square of size %f" %self.size)
        self.executions = 0
        self.success = True
        self.seconds_total = 0

        self.takeoff()
        self.move_square()
        self.land()

        if self.success:
            self.result.result = self.seconds_total
            rospy.loginfo("Action completed.")
            self.action_server.set_succeeded(self.result)

    def move_square(self):
        while self.executions < 4:

            if self.action_server.is_preempt_requested():
                rospy.loginfo("The goal has been cancelled")
                self.action_server.set_preempted()
                self.success = False
                break

            self.feedback.feedback = self.executions
            self.action_server.publish_feedback(self.feedback)
            self.move_straight()
            self.stop(self.rate_long)
            self.turn()
            self.stop(self.rate_short)
            self.executions += 1

    def move_straight(self):
        rospy.loginfo("Moving straight.")
        self.lenght = 0

        while self.lenght <= self.size:
            self.vel_msg.linear.x = 1
            self.vel_msg.angular.z = 0
            self.vel_msg.linear.z = 0
            self.vel_pub.publish(self.vel_msg)
            self.rate_long.sleep()
            self.lenght += 1
            self.seconds_total += 1

    def stop(self, duration):
        rospy.loginfo("Stopping.")
        self.seconds = 0

        while self.seconds <= 1:
            self.vel_msg.linear.x = 0
            self.vel_msg.angular.z = 0
            self.vel_msg.linear.z = 0

            self.vel_pub.publish(self.vel_msg)
            duration.sleep()
            self.seconds += 1

            if duration == self.rate_long:
                self.seconds_total += 1

    def turn(self):
        rospy.loginfo("Turning.")
        self.seconds = 0

        while self.seconds <= 2:
            self.vel_msg.linear.x = 0
            self.vel_msg.angular.z = 0.65
            self.vel_msg.linear.z = 0
            self.vel_pub.publish(self.vel_msg)
            self.rate_long.sleep()
            self.seconds += 1
            self.seconds_total += 1

    def publish_once_in_cmd_vel(self):

        while not self.ctrl_c:
            connections = self.vel_pub.get_num_connections()

            if connections > 0:
                self.vel_pub.publish(self.vel_msg)
                rospy.loginfo("Cmd Published")
                break

            else:
                self.rate_long.sleep()

    def takeoff(self):
        self.seconds = 0
        rospy.loginfo("Taking off")

        while self.seconds <= 3:
            self.vel_msg.linear.z = 1
            self.vel_msg.linear.x = 0
            self.vel_msg.angular.z = 0
            self.vel_pub.publish(self.vel_msg)
            self.rate_long.sleep()
            self.seconds += 1
            self.seconds_total += 1

    def land(self):
        self.seconds = 0
        rospy.loginfo("Landing")

        while self.seconds <= 3:
            self.vel_msg.linear.z = -1
            self.vel_msg.linear.x = 0
            self.vel_msg.angular.z = 0
            self.vel_pub.publish(self.vel_msg)
            self.rate_long.sleep()
            self.seconds += 1
            self.seconds_total += 1

if __name__ == "__main__":
    rospy.init_node("drone_perform_square_action_server")
    MoveSquare()
    rospy.spin()
