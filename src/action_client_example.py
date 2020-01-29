import rospy
import actionlib
from ardrone_as.msg import ArdroneAction, ArdroneGoal, ArdroneFeedback, ArdroneResult
from geometry_msgs.msg import Twist

class MoveDrone:
    def __init__(self):
        self.client = actionlib.SimpleActionClient("/ardrone_action_server", ArdroneAction)
        self.client.wait_for_server()
        self.goal = ArdroneGoal()
        self.goal.nseconds = 10
        self.nImage = 1
        self.rate = rospy.Rate(1)
        self.vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size = 1)
        self.vel_msg = Twist()
        self.ctrl_c = False

    def send_goal(self):
        print("Taking off.")
        self.takeoff()
        self.client.send_goal(self.goal, feedback_cb = self.feedback_callback)
        print("Goal send.")
        self.state = self.client.get_state()

        while self.state < 2:
            self.circle()
            self.state = self.client.get_state()

        print("Landing.")
        self.land()

    def feedback_callback(self, feedback):
        print("[Feedback] image n.%d received"%self.nImage)
        self.nImage += 1

    def takeoff(self):
        self.seconds = 0

        while self.seconds <= 3:
            self.vel_msg.linear.z = 1
            self.vel_msg.linear.x = 0
            self.vel_msg.angular.z = 0
            self.vel_pub.publish(self.vel_msg)
            self.rate.sleep()
            self.seconds += 1

    def circle(self):
        self.vel_msg.linear.z = 0
        self.vel_msg.linear.x = 0.5
        self.vel_msg.angular.z = 0.5
        self.vel_pub.publish(self.vel_msg)

    def land(self):
        self.seconds = 0

        while self.seconds <= 3:
            self.vel_msg.linear.z = -1
            self.vel_msg.linear.x = 0
            self.vel_msg.angular.z = 0
            self.vel_pub.publish(self.vel_msg)
            self.rate.sleep()
            self.seconds += 1

    def publish_once_in_cmd_vel(self):
        while not self.ctrl_c:
            connections = self.vel_pub.get_num_connections()
            if connections > 0:
                self.vel_pub.publish(self.vel_msg)
                rospy.loginfo("Cmd Published")
                break
            else:
                self.rate.sleep()

rospy.init_node("move_drone_while_taking_pictures_action_client")
move_drone = MoveDrone()
move_drone.publish_once_in_cmd_vel()
move_drone.send_goal()
