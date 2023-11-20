#!/usr/bin/env python3
import math
from dataclasses import dataclass, field
import cvxpy
import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from scipy.linalg import block_diag
from scipy.sparse import block_diag, csc_matrix
from mpc.mpc_utils import nearest_point
from transforms3d.euler import quat2euler

# TODO CHECK: include needed ROS msg type headers and libraries

# Note on reference trajectories: So, with a B-spline for instance (where you
# use one of 3 points to control the curvature), you could just represent in
# memory that trajectory using the 3 points that describe that spline.

# OR, using some resolution "waypoints per meter," you could sample a waypoint
# from the trajectory / spline at every distance along the spline. Can do this
# according to velocity and timestep (like what happens inside the MPC
# controller), or could do this based on some fixed resolution, or based on
# velocity, etc.


@dataclass
class mpc_config:
    NXK: int = 4  # length of kinematic state vector: z = [x, y, v, yaw]
    NU: int = 2  # length of input vector: u = = [steering, acceleration]
    TK: int = 8  # finite time horizon length kinematic

    # ---------------------------------------------------
    # TODO: you may need to tune the following matrices
    # This is the vector of control input weights. I.e., how much we penalize
    # changing each of our control inputs' values by. 
    Rk: list = field(
        default_factory=lambda: np.diag([0.01, 10.0])
    )  # input cost matrix, penalty for inputs - [accel, steering]
    # This is the vector of control input DIFFERENCE weights. I.e., how much we
    # penalize large changes for each of our control input values. 
    Rdk: list = field(
        default_factory=lambda: np.diag([0.01, 10.0])
    )  # input difference cost matrix, penalty for change of inputs - [accel, steering]
    # This is the vector of weights that defines how much we want to weight each
    # of the state vector differences. I.e., how much cost do we add for large
    # differences in each our state variables? If we care more about one state
    # variable than another, then we might place high weight here, as we want
    # the difference to result in a high cost--therefore hopefully solving for a
    # control value that yields a smaller difference for that variable.
    Qk: list = field(
        default_factory=lambda: np.diag([80.0, 80.0, 80, 5.5])
    )  # state error cost matrix, for the the next (T) prediction time steps [x, y, v, yaw]
    # This is the vector of weights that defines how important for our state
    # variables are at the end of the time horizon. I.e., if we want the control
    # values we pick to get us as close as possible to the desired position T
    # timesteps away, then more weight should go to those values.
    Qfk: list = field(
        default_factory=lambda: np.diag([80.0, 80.0, 80, 5.5])
    )  # final state error matrix, penalty  for the final state constraints: [x, y, v, yaw]
    # ---------------------------------------------------

    N_IND_SEARCH: int = 20  # Search index number
    DTK: float = 0.1  # time step [s] kinematic
    dlk: float = 0.03  # dist step [m] kinematic
    LENGTH: float = 0.58  # Length of the vehicle [m]
    WIDTH: float = 0.31  # Width of the vehicle [m]
    WB: float = 0.33  # Wheelbase [m]
    MIN_STEER: float = -0.4189  # maximum steering angle [rad]
    MAX_STEER: float = 0.4189  # maximum steering angle [rad]
    MAX_DSTEER: float = np.deg2rad(180.0)  # maximum steering speed [rad/s]
    MAX_SPEED: float = 6.0  # maximum speed [m/s]
    MIN_SPEED: float = 0.0  # minimum backward speed [m/s]
    MAX_ACCEL: float = 3.0  # maximum acceleration [m/ss]


@dataclass
class State:
    x: float = 0.0
    y: float = 0.0
    v: float = 0.0
    yaw: float = 0.0
    
class MPC(Node):
    """ 
    Implement Kinematic MPC on the car
    This is just a template, you are free to implement your own node!
    """
    def __init__(self):
        super().__init__('mpc')

        # Create subscriber for car's odometry topic.
        self.__odom_subscriber = self.create_subscription(msg_type=Odometry,
                                                          topic="odom",
                                                          callback=self.__odom_callback,
                                                          qos_profile=10)
        # Create publisher for drive messages.
        self.__drive_publisher = self.create_publisher(msg_type=AckermannDriveStamped,
                                                       topic="drive",
                                                       qos_profile=10)
        
        # self.__path_subscriber = self.create_subscription(msg_type=Path,
        #                                                   topic="path",
        #                                                   callback=self.__path_callback,
        #                                                   qos_profile=10)
        # self.__path = None
        # TODO: TEMPORARILY hardcoding the path into the node so that we have
        # something to work with. In the long run, I really want a separate node
        # to publish a joint trajectory--containing the position of each
        # waypoint, the velocity, and the heading angle/orientation each one
        # should have.
        # NOTE: Should just run the existing path publisher with this path
        # anyway, just so that I can tune the controller to better track the
        # path. I.e., just use the existing path publisher to make the path
        # visible in RVIZ.
        self.__max_longitudinal_velocity = 4.0
        self.__min_longitudinal_velocity = 0.3
        self.__trajectory = [
            [13.560638427734375,2.7555694580078125],
            [13.00737476348877,2.4709153175354004],
            [12.806493759155273,1.7395272254943848],
            [12.502553939819336,1.0535345077514648],
            [11.965315818786621,0.6088647842407227],
            [11.178140640258789,0.31683921813964844],
            [10.536487579345703,-0.44613075256347656],
            [10.518616676330566,-1.3454322814941406],
            [11.093320846557617,-2.232316017150879],
            [11.869260787963867,-2.6306419372558594],
            [12.653507232666016,-2.681591033935547],
            [13.429366111755371,-2.5058536529541016],
            [14.09465503692627,-2.23138427734375],
            [14.64012336730957,-1.8782730102539062],
            [15.353914260864258,-1.2399787902832031],
            [16.188953399658203,-0.6803398132324219],
            [17.26852798461914,0.560297966003418],
            [18.104114532470703,0.9899187088012695],
            [18.948312759399414,1.8311080932617188],
            [19.143470764160156,2.903562068939209],
            [19.13678741455078,3.862104892730713],
            [18.421070098876953,4.616009712219238],
            [17.445926666259766,4.943364143371582],
            [16.599119186401367,4.811524391174316],
            [16.227771759033203,4.384362697601318],
            [15.65716552734375,3.8408212661743164],
            [15.263104438781738,3.125494956970215],
            [14.539806365966797,2.9095115661621094],
            [13.69031047821045,2.8717823028564453],
            [13.113306045532227,2.4780378341674805]
        ]
        # Compute heading angle between waypoints.
        for s in range(len(self.__trajectory) - 1):
            point1 = self.__trajectory[s]
            point2 = self.__trajectory[s+1]
            yaw_between_points_rad = np.arctan2(point2[1] - point1[1], point2[0] - point1[0])
            self.__trajectory[s].append(yaw_between_points_rad)
        # Make sure we also compute that angle between the last point and the
        # first point as well.
        point1 = self.__trajectory[-1]
        point2 = self.__trajectory[0]
        yaw_between_points_rad = np.arctan2(point2[1] - point1[1], point2[0] - point1[0])
        self.__trajectory[-1].append(yaw_between_points_rad)

        # For now, add a constant velocity to each waypoint.
        self.__temp_longitudinal_velocity = 3.0
        for s in range(len(self.__trajectory)):
            self.__trajectory[s].append(self.__temp_longitudinal_velocity)

        # Convert the trajectory to a numpy array.
        self.__trajectory = np.array(self.__trajectory)

        # Instantiate config dataclass.
        self.config = mpc_config()
        self.odelta = None
        self.oa = None
        self.init_flag = 0

        # initialize MPC problem
        # This sets up (defines/initializes) the linearized vehicle model, sets
        # up all other (not part of the vehicle model) linearized constraints,
        # etc. Also defines the cost funtion. Sets up the CVXPY solver. Only
        # need to run this once in the beginning to initialize everything.
        self.mpc_prob_init()

    # def __path_callback(self, path_msg: Path) -> None:
    #     """Simple callback to store most recently received path message.

    #     Args:
    #         path_msg (Path): Received Path message.
    #     """
    #     self.__path = path_msg

    def __odom_callback(self, odom_msg: Odometry) -> None:

        self.get_logger().info("Received new odom message!!!")

        # TODO: extract pose from ROS msg
        
        # TODO: Update node to receive odometry message instead of pose. I.e.,
        # we need an estimate of the vehicle's speed--and while you CAN get that
        # from TF (based on the changes in transformation)--apparently that's
        # going to be very noisy and not very helpful (unless you'd pass it
        # through some kind of bayes/kalman filter). Soooo, instead, I think we
        # need to rely on the same paradigm here where we continously use the
        # estimated speed from odometry (whether visual, intertial, whatever)
        # and then correct that based on some sort of localization or GPS or
        # something like that. Again, I think realistically, all these things
        # would come out of a KALMAN filter responsible for combining all those
        # values and spitting out its best estimate of the current pose and
        # twist???
        # NOTE: I actually probably don't need to take the norm of this vector.
        # Rather, I think because this is an ackerman vehicle, from the
        # perspective of the base_link, it's ONLY ever going to have velocity in
        # the x direction--however, I'm a little fuzzy on this, so it'd be nice
        # to clarify.
        # current_speed = np.linalg.norm([odom_msg.twist.twist.linear.x,
        #                                 odom_msg.twist.twist.linear.y,
        #                                 odom_msg.twist.twist.linear.z])
        current_longitudinal_velocity = odom_msg.twist.twist.linear.x
        # TODO: Extract Roll, Pitch, and Yaw of the base_link's frame with
        # respect to the parent frame. Have to construct the quaternian in the
        # format the transforms3d function is expecting (w,x,y,z). In this case,
        # only really care about heading angle==yaw.
        orientation_quat = (
            odom_msg.pose.pose.orientation.w,
            odom_msg.pose.pose.orientation.x,
            odom_msg.pose.pose.orientation.y,
            odom_msg.pose.pose.orientation.z
        )
        _, _, yaw = quat2euler(quaternion=orientation_quat)
        
        vehicle_state = State(x=odom_msg.pose.pose.position.x,
                              y=odom_msg.pose.pose.position.y,
                              v=current_longitudinal_velocity,
                              yaw=yaw)

        # TODO: Calculate the next reference trajectory for the next T steps
        #       with current vehicle pose.
        #       ref_x, ref_y, ref_yaw, ref_v are columns of self.waypoints

        # Before doing anything, make sure there's a valid trajectory to work
        # with. Otherwise, just return--can't do anything without a reference
        # trajectory from planner, as we need a path to track!
        # if self.__trajectory is None:
        #     return

        # Otherwise, if we do have a valid path, then extract the x, y, heading
        # angle, and velocity from each waypoint on the path.
        waypoint_x_values = self.__trajectory[:, 0]
        waypoint_y_values = self.__trajectory[:, 1]
        waypoint_yaw_values = self.__trajectory[:, 2]
        waypoint_longitudinal_velocity_values = self.__trajectory[:, 3]

        # Use calc_ref_trajectory to only get a relevant subset of the provided
        # waypoints from the original trajectory.
        ref_path = self.calc_ref_trajectory(state=vehicle_state, 
                                            cx=waypoint_x_values,
                                            cy=waypoint_y_values,
                                            cyaw=waypoint_yaw_values,
                                            sp=waypoint_longitudinal_velocity_values)
        self.get_logger().info(f"ref path: {ref_path}")
        x0 = [vehicle_state.x, vehicle_state.y, vehicle_state.v, vehicle_state.yaw]

        # TODO: solve the MPC control problem
        (
            oa,
            odelta,
            ox,
            oy,
            oyaw,
            ov,
            state_predict,
        ) = self.linear_mpc_control(ref_path, x0, self.oa, self.odelta)

        # Store the optimal acceleration and optimal steering angle sequences
        # from this most recent solve. These will be used in the next solve to
        # compute the linearized state model over TK timesteps!
        self.oa = oa
        self.odelta = odelta

        # TODO Construct and publish a new ackermann drive message with these
        # optimal control values--but only if the solve was successful.
        if oa is None or odelta is None:
            self.get_logger().error(f"Failed to solve MPC problem -- no drive message published.")
            return

        # TODO: Grab only the first value in the sequence of TK optimal control
        # values.
        # NOTE: We may only be able to control the car's velocity--not its
        # acceleration. Not sure about this yet.
        # new_acceleration = oa[0]
        new_speed = vehicle_state.v + self.oa[0] * self.config.DTK
        new_steering_angle = self.odelta[0]

        new_drive_message = AckermannDriveStamped()
        # new_drive_message.drive.acceleration = new_acceleration
        new_drive_message.drive.speed = new_speed
        new_drive_message.drive.steering_angle = new_steering_angle
        self.__drive_publisher.publish(new_drive_message)

        # # TODO: publish drive message.
        # steer_output = self.odelta[0]
        # speed_output = vehicle_state.v + self.oa[0] * self.config.DTK

    # ONLY RUNS ONCE--doesn't get run every time during solving.
    def mpc_prob_init(self):
        """
        Create MPC quadratic optimization problem using cvxpy, solver: OSQP
        Will be solved every iteration for control.
        More MPC problem information here: https://osqp.org/docs/examples/mpc.html
        More QP example in CVXPY here: https://www.cvxpy.org/examples/basic/quadratic_program.html
        """
        # Initialize and create vectors for the optimization problem
        # Vehicle State Vector
        self.xk = cvxpy.Variable(
            (self.config.NXK, self.config.TK + 1)
        )
        # Control Input vector
        self.uk = cvxpy.Variable(
            (self.config.NU, self.config.TK)
        )
        cost_function = 0.0  # Cost function description that we'll build up below.
        constraints = []  # Create constraints array

        # Initialize reference vectors
        self.x0k = cvxpy.Parameter((self.config.NXK,))
        self.x0k.value = np.zeros((self.config.NXK,))

        # Initialize reference trajectory parameter
        self.ref_traj_k = cvxpy.Parameter((self.config.NXK, self.config.TK + 1))
        self.ref_traj_k.value = np.zeros((self.config.NXK, self.config.TK + 1))

        # --------------------------------------------------------
        # TODO: Define the objective (cost function) that CVXPY will find values
        # for variables xk and uk to minimize the value of. 
        #
        # NOTE that we're not computing a value for objective here. Rather,
        # we're building up a description / mathematical expression of cvxpy
        # variables, parameters, and constants to compute the value of the cost
        # function. I.e., we're telling CVXPY which variable/parameter/constant
        # values to use and what operations to perform with them.
        # 
        # Also NOTE that the (atomic) operations (and aliases of operations)
        # that we use (provided by cvpyx) will ultimately determine whether the
        # cost function we develop is convex, concave, or neither. If we use
        # CVXPY atomic operations, then we can be certain that DCP rules are
        # followed and that our cost function will be able minimized/maximized.
        # I.e., if our cost function ends up being convex (gobal minimum, facing
        # up), our problem's OBJECTIVE will be to MINIMIZE the value of the COST
        # FUNCTION. If our cost function ends up being concave (global maximum,
        # "facing down"), the our problem's OBJECTIVE will be to MAXIMIZE the
        # value of the COST FUNCTION. In both cases, this involves finding the
        # values of the problem's variables that minimize or maximize the value
        # of the function (depending on the cost function's curvature). See
        # https://www.cvxpy.org/tutorial/dcp/index.html#disciplined-convex-programming
        
        # TODO: Objective part 1: Influence of the control inputs: Inputs u
        # multiplied by the penalty R
        control_value_cost = 0
        for t in range(self.config.TK):
            control_value_cost += cvxpy.quad_form(x=self.uk[:, t], P=self.config.Rk)

        # TODO: Objective part 2: Deviation of the vehicle from the reference
        # trajectory weighted by Q, including final Timestep T weighted by Qf
        # Have to add two parts here: one is the weights for the terminal state,
        # and one component for all the rest.
        # First, calculate the tracking cost for the first 0--T-1 states.
        tracking_cost = 0
        for t in range(self.config.TK):
            tracking_cost += cvxpy.quad_form(x=self.ref_traj_k[:, t]-self.xk[:, t], P=self.config.Qk)
        # Then, calculate the tracking cost associated with how far the
        # last/final "optimal" state deviates from the last reference state.
        tracking_cost += cvxpy.quad_form(x=self.ref_traj_k[:, self.config.TK]-self.xk[:, self.config.TK], P=self.config.Qfk) 

        # TODO: Objective part 3: Telling CVXPY how to compute the cost
        # associated with changes in the control value vector from one timestep
        # to the next.
        control_value_change_cost = 0
        for t in range(self.config.TK - 1):
            control_value_change_cost += cvxpy.quad_form(self.uk[:, t+1] - self.uk[:, t], self.config.Rdk)

        # Add all the above cost function components to the actual cost
        # function.
        cost_function = control_value_cost + tracking_cost + control_value_change_cost

        # # --------------------------------------------------------

        # # Constraints 1: Calculate the future vehicle behavior/states based on the vehicle dynamics model matrices
        # # Evaluate vehicle Dynamics for next T timesteps
        # A_block = []
        # B_block = []
        # C_block = []
        # # init path to zeros
        # # Creates 4x9 matrix. 9 columns for 9 timesteps, where each row in each
        # # column is a different state variable.
        # path_predict = np.zeros((self.config.NXK, self.config.TK + 1))
        # # For each timestep, compute a new linearized model matrix set.
        # for t in range(self.config.TK):
        #     # Pass the velocity (v at index 2) and heading (phi at index 3) and
        #     # the steering angle as 0 into the get_model_matrix function.
        #     A, B, C = self.get_model_matrix(
        #         path_predict[2, t], path_predict[3, t], 0.0
        #     )
        #     A_block.append(A)
        #     B_block.append(B)
        #     C_block.extend(C)

        # A_block = block_diag(tuple(A_block))
        # B_block = block_diag(tuple(B_block))
        # # C_block = np.array(C_block)
        # # TODO JUST FOR NOW, going to try to form C_block as a 2D array to match
        # # A and B. C should be NXKxTK--that's what this reshape accomplishes.
        # C_block = np.reshape(np.array(C_block), (self.config.NXK, -1), order='F')

        # # [AA] Sparse matrix to CVX parameter for proper stuffing
        # # Reference: https://github.com/cvxpy/cvxpy/issues/1159#issuecomment-718925710
        # m, n = A_block.shape
        # # NOTE: we create these model coefficient matrices as CVXPY parameters,
        # # as they're not Variables whose value we're trying to optimize, but
        # # instead values that we're simply using in the optimization process.
        # self.Annz_k = cvxpy.Parameter(A_block.nnz)
        # data = np.ones(self.Annz_k.size)
        # rows = A_block.row * n + A_block.col
        # cols = np.arange(self.Annz_k.size)
        # Indexer = csc_matrix((data, (rows, cols)), shape=(m * n, self.Annz_k.size))

        # # Setting sparse matrix data
        # self.Annz_k.value = A_block.data

        # # Now we use this sparse version instead of the old A_ block matrix
        # self.Ak_ = cvxpy.reshape(Indexer @ self.Annz_k, (m, n), order="C")

        # # Same as A
        # m, n = B_block.shape
        # self.Bnnz_k = cvxpy.Parameter(B_block.nnz)
        # data = np.ones(self.Bnnz_k.size)
        # rows = B_block.row * n + B_block.col
        # cols = np.arange(self.Bnnz_k.size)
        # Indexer = csc_matrix((data, (rows, cols)), shape=(m * n, self.Bnnz_k.size))
        # self.Bk_ = cvxpy.reshape(Indexer @ self.Bnnz_k, (m, n), order="C")
        # self.Bnnz_k.value = B_block.data

        # # No need for sparse matrices for C as most values are parameters
        # self.Ck_ = cvxpy.Parameter(C_block.shape)
        # self.Ck_.value = C_block

        # # -------------------------------------------------------------
        # # TODO: Constraint part 1:
        # #       Add dynamics constraints to the optimization problem
        # #       This constraint should be based on a few variables:
        # #       self.xk, self.Ak_, self.Bk_, self.uk, and self.Ck_. I.e.,
        # #       constrain the optimization to produce xk and uk values that
        # #       conform to the system model--meaning the uk and xk that are
        # #       computed will be kinematically feasible.

        # # NOTE: This could be wrong, depending on how Ak_ and Bk_ are really
        # # laid out in memory.
        # # NOTE: On that note as well, I need to make sure that A, B, and C are
        # # structured like I'm expecting. I'm expecting to be able to multiply A
        # # by an nx1 matrix like "A @ (nx1)". If A isn't laid out such that each
        # # row is the kth coefficient corresponding to the kth variable of the
        # # state, then this won't work.

        # for t in range(self.config.TK):
        #     # Get the "t-th" a and b matrices from the block versions above
        #     a_t = self.Ak_[self.config.NXK*t:self.config.NXK*t + self.config.NXK, self.config.NXK*t:self.config.NXK*t + self.config.NXK]
        #     b_t = self.Bk_[self.config.NXK*t:self.config.NXK*t + self.config.NXK, self.config.NU*t:self.config.NU*t + self.config.NU]
        #     # c_t =
        #     # self.Ck_[self.config.NXK*t:self.config.NXK*t:self.config.NXK*t +
        #     # self.config.NXK]
        #     c_t = self.Ck_[:, t]
        #     # Define the "t-th" component/constraint of the system / state
        #     # model.
        #     # Assuming b_t is 4x2 and uk[:, t] is 2x1, that part should be okay
        #     # and make sense. I think the issue is in the slicing for b_t and
        #     # c_t.
        #     constraints.append(self.xk[:, t+1] == a_t @ self.xk[:, t] + b_t @ self.uk[:, t] + c_t)

        # # NOTE Only problem with this, however, is that after getting "blocked,"
        # # the A-matrix gets put into a sparse representation from
        # # scipy--therefore, I don't think indexing in this way is going to work.
        # # NOTE BUT--we reshape back to mxn, so I feel like that means it IS IN
        # # that block form somehow still?

        # # Is there a more compact of making this work?
        # # What if we just multiplied the diagonal Ak_ matrix by a flattened
        # # version of x? If you did this, you'd have to reshape the result--which
        # # is fine, but that doesn't really feel any cleaner than what I did
        # # above.
        # # constraints.append()
        
        # TODO: Constraint part 2:
        #       Add constraints on steering, change in steering angle
        #       cannot exceed steering angle speed limit. Should be based on:
        #       self.uk, self.config.MAX_DSTEER, self.config.DTK
        min_steering_angle_constraint = self.uk[0:] >= self.config.MIN_STEER
        constraints.append(min_steering_angle_constraint)
        max_steering_angle_constraint = self.uk[0:] <= self.config.MAX_STEER
        constraints.append(max_steering_angle_constraint)
        # The change per unit of time must not exceed the maximum rate of change
        # of the steering angle from one computed steering angle to the next.
        for t in range(self.config.TK-1):
            new_max_steering_rate_constraint = (self.uk[0, t+1] - self.uk[0, t])/self.config.DTK <= self.config.MAX_DSTEER
            constraints.append(new_max_steering_rate_constraint)

        # TODO: Constraint part 3:
        #       Add constraints on upper and lower bounds of states and inputs
        #       and initial state constraint, should be based on:
        #       self.xk, self.x0k, self.config.MAX_SPEED, self.config.MIN_SPEED,
        #       self.uk, self.config.MAX_ACCEL, self.config.MAX_STEER
        
        # -------------------------------------------------------------
        
        # Add constraint specifying that the first predicted state is equal to
        # the current state.
        constraints.append(self.xk[:, 0] == self.x0k)
        # Add constraint specifying the max speed (v) we should be able to reach
        # (as velocity is part of our state).
        constraints.append(self.xk[2,:] <= self.config.MAX_SPEED)
        # Add constraint specifying minimum allowable velocity.
        constraints.append(self.xk[2,:] >= self.config.MIN_SPEED)
        # Add constraint specifying the maximum acceleration--which is
        # essentially just the difference in velocity divided by the time in
        # between each timestep.
        for t in range(self.config.TK):
            constraints.append((self.xk[2,t+1] - self.xk[2,t])/self.config.DTK <= self.config.MAX_ACCEL)

        # Create the optimization problem in CVXPY and setup the workspace
        # Optimization goal (I.e., the "objective"): Find the values of xk and
        # uk that minimize the value of the cost_function, subject to the list
        # of constraints.
        self.MPC_prob = cvxpy.Problem(cvxpy.Minimize(cost_function), constraints)

    def calc_ref_trajectory(self, state, cx, cy, cyaw, sp):
        """
        calc referent trajectory ref_traj in T steps: [x, y, v, yaw]
        using the current velocity, calc the T points along the reference path
        :param cx: Course X-Position
        :param cy: Course y-Position
        :param cyaw: Course Heading
        :param sp: speed profile
        :dl: distance step
        :pind: Setpoint Index
        :return: reference trajectory ref_traj, reference steering angle
        """

        # Create placeholder Arrays for the reference trajectory for T steps
        ref_traj = np.zeros((self.config.NXK, self.config.TK + 1))
        ncourse = len(cx)

        # Find nearest index/setpoint from where the trajectories are calculated
        _, _, _, ind = nearest_point(np.array([state.x, state.y]), np.array([cx, cy]).T)

        # Load the initial parameters from the setpoint into the trajectory
        ref_traj[0, 0] = cx[ind]
        ref_traj[1, 0] = cy[ind]
        ref_traj[2, 0] = sp[ind]
        ref_traj[3, 0] = cyaw[ind]

        # based on current velocity, distance traveled on the ref line between time steps
        travel = abs(state.v) * self.config.DTK
        dind = travel / self.config.dlk
        ind_list = int(ind) + np.insert(
            np.cumsum(np.repeat(dind, self.config.TK)), 0, 0
        ).astype(int)
        ind_list[ind_list >= ncourse] -= ncourse
        ref_traj[0, :] = cx[ind_list]
        ref_traj[1, :] = cy[ind_list]
        ref_traj[2, :] = sp[ind_list]
        cyaw[cyaw - state.yaw > 4.5] = np.abs(
            cyaw[cyaw - state.yaw > 4.5] - (2 * np.pi)
        )
        cyaw[cyaw - state.yaw < -4.5] = np.abs(
            cyaw[cyaw - state.yaw < -4.5] + (2 * np.pi)
        )
        ref_traj[3, :] = cyaw[ind_list]

        return ref_traj

    def predict_motion(self, x0, oa, od, xref):
        path_predict = xref * 0.0
        for i, _ in enumerate(x0):
            path_predict[i, 0] = x0[i]

        state = State(x=x0[0], y=x0[1], yaw=x0[3], v=x0[2])
        for (ai, di, i) in zip(oa, od, range(1, self.config.TK + 1)):
            state = self.update_state(state, ai, di)
            path_predict[0, i] = state.x
            path_predict[1, i] = state.y
            path_predict[2, i] = state.v
            path_predict[3, i] = state.yaw

        return path_predict

    def update_state(self, state, a, delta):

        # input check
        if delta >= self.config.MAX_STEER:
            delta = self.config.MAX_STEER
        elif delta <= -self.config.MAX_STEER:
            delta = -self.config.MAX_STEER

        state.x = state.x + state.v * math.cos(state.yaw) * self.config.DTK
        state.y = state.y + state.v * math.sin(state.yaw) * self.config.DTK
        state.yaw = (
            state.yaw + (state.v / self.config.WB) * math.tan(delta) * self.config.DTK
        )
        state.v = state.v + a * self.config.DTK

        if state.v > self.config.MAX_SPEED:
            state.v = self.config.MAX_SPEED
        elif state.v < self.config.MIN_SPEED:
            state.v = self.config.MIN_SPEED

        return state

    def get_model_matrix(self, v, phi, delta):
        """
        Calc linear and discrete time dynamic model-> Explicit discrete time-invariant
        Linear System: Xdot = Ax +Bu + C
        State vector: x=[x, y, v, yaw]
        :param v: speed
        :param phi: heading angle of the vehicle
        :param delta: steering angle: delta_bar
        :return: A, B, C
        """
        
        # Note that some of these model matrices (like A) require other values
        # from the state (like v and yaw). These values come from the reference
        # trajectory or interpolated, predicted trajectory that starts at the
        # nearest point on the reference trajectory.

        # Otherwise, these matrices are constructed just like expected. I.e.,
        # there's a row for every state variable's coefficients corresponding to
        # each of the other state variables.

        # This funciton doesn't do any special operations on those shaping
        # operations of anything like that on these matrices directly. Rather,
        # it returns them in their original, intuitive form.

        # State (or system) matrix A, 4x4
        A = np.zeros((self.config.NXK, self.config.NXK))
        A[0, 0] = 1.0
        A[1, 1] = 1.0
        A[2, 2] = 1.0
        A[3, 3] = 1.0
        A[0, 2] = self.config.DTK * math.cos(phi)
        A[0, 3] = -self.config.DTK * v * math.sin(phi)
        A[1, 2] = self.config.DTK * math.sin(phi)
        A[1, 3] = self.config.DTK * v * math.cos(phi)
        A[3, 2] = self.config.DTK * math.tan(delta) / self.config.WB

        # Input Matrix B; 4x2
        B = np.zeros((self.config.NXK, self.config.NU))
        B[2, 0] = self.config.DTK
        B[3, 1] = self.config.DTK * v / (self.config.WB * math.cos(delta) ** 2)

        C = np.zeros(self.config.NXK)
        C[0] = self.config.DTK * v * math.sin(phi) * phi
        C[1] = -self.config.DTK * v * math.cos(phi) * phi
        C[3] = -self.config.DTK * v * delta / (self.config.WB * math.cos(delta) ** 2)

        return A, B, C

    def mpc_prob_solve(self, ref_traj, path_predict, x0):
        self.x0k.value = x0

        # A_block = []
        # B_block = []
        # C_block = []
        # for t in range(self.config.TK):
        #     A, B, C = self.get_model_matrix(
        #         path_predict[2, t], path_predict[3, t], 0.0
        #     )
        #     A_block.append(A)
        #     B_block.append(B)
        #     C_block.extend(C)

        # A_block = block_diag(tuple(A_block))
        # B_block = block_diag(tuple(B_block))
        # # C_block = np.array(C_block)
        # C_block = np.reshape(np.array(C_block), (self.config.NXK, -1), order='F')

        # self.Annz_k.value = A_block.data
        # self.Bnnz_k.value = B_block.data
        # self.Ck_.value = C_block

        self.ref_traj_k.value = ref_traj

        # Solve the optimization problem in CVXPY
        # Solver selections: cvxpy.OSQP; cvxpy.GUROBI
        self.MPC_prob.solve(solver=cvxpy.OSQP, verbose=False, warm_start=True)

        if (self.MPC_prob.status == cvxpy.OPTIMAL or self.MPC_prob.status == cvxpy.OPTIMAL_INACCURATE):
            self.get_logger().info(f"CVXPY finished with status: {str(self.MPC_prob.status)}")
            ox = np.array(self.xk.value[0, :]).flatten()
            oy = np.array(self.xk.value[1, :]).flatten()
            ov = np.array(self.xk.value[2, :]).flatten()
            oyaw = np.array(self.xk.value[3, :]).flatten()
            oa = np.array(self.uk.value[0, :]).flatten()
            odelta = np.array(self.uk.value[1, :]).flatten()

        else:
            self.get_logger().warning(f"Error: Cannot solve mpc..status: {str(self.MPC_prob.status)}")
            oa, odelta, ox, oy, oyaw, ov = None, None, None, None, None, None

        return oa, odelta, ox, oy, oyaw, ov

    def linear_mpc_control(self, ref_path, x0, oa, od):
        """
        MPC contorl with updating operational point iteraitvely
        :param ref_path: reference trajectory in T steps
        :param x0: initial state vector
        :param oa: acceleration of T steps of last time
        :param od: delta of T steps of last time
        """

        # oa and od are supposed to be what the acceleration and steering angle
        # (delta) should be through the next TK timesteps. In this case, we're
        # just going to predict the motion of the vehicle if those control
        # values were to stay the same. I.e., constant velocity (because
        # acceleration 0) and zero steering angle. Wouldn't this always produce
        # the predicted path as a straight line? Well, maybe--but remember--this
        # IS NOT the trajectory that the solver compares xk to. I.e., the cost
        # function looks at the reference trajectory, not this predicted
        # trajectory. This predicted trajectory is ONLY USED to linearize the
        # state model and generate matrices A, B, and C. This is okay, as these
        # can be approximately correct.
        # ACTUALLY, though, I'm now realizing that these are only zero
        # INITIALLY. In subsequent calls to solve the MPC problem, the next call
        # will incorporate the previous call's computed optimal control value
        # sequence --as those are a part of calculating the A and B matrix over
        # TK timesteps!!
        # Therefore, by using the previously optimal values, hopefully as time
        # goes on, our linearized state model becomes a better and better
        # approximation of the original, nonlinearized state model.
        if oa is None or od is None:
            oa = [0.0] * self.config.TK
            od = [0.0] * self.config.TK

        # Call the Motion Prediction function: Predict the vehicle motion for x-steps
        path_predict = self.predict_motion(x0, oa, od, ref_path)
        poa, pod = oa[:], od[:]

        # Run the MPC optimization: Create and solve the optimization problem
        mpc_a, mpc_delta, mpc_x, mpc_y, mpc_yaw, mpc_v = self.mpc_prob_solve(
            ref_path, path_predict, x0
        )

        return mpc_a, mpc_delta, mpc_x, mpc_y, mpc_yaw, mpc_v, path_predict

def main(args=None):
    rclpy.init(args=args)
    mpc_node = MPC()
    rclpy.spin(mpc_node)
    mpc_node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()