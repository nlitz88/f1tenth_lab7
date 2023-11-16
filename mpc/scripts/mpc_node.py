#!/usr/bin/env python3
import math
from dataclasses import dataclass, field

import cvxpy
import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDrive, AckermannDriveStamped
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from scipy.linalg import block_diag
from scipy.sparse import block_diag, csc_matrix, diags
from sensor_msgs.msg import LaserScan
from utils import nearest_point

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
        default_factory=lambda: np.diag([0.01, 100.0])
    )  # input cost matrix, penalty for inputs - [accel, steering]
    # This is the vector of control input DIFFERENCE weights. I.e., how much we
    # penalize large changes for each of our control input values. 
    Rdk: list = field(
        default_factory=lambda: np.diag([0.01, 100.0])
    )  # input difference cost matrix, penalty for change of inputs - [accel, steering]
    # This is the vector of weights that defines how much we want to weight each
    # of the state vector differences. I.e., how much cost do we add for large
    # differences in each our state variables? If we care more about one state
    # variable than another, then we might place high weight here, as we want
    # the difference to result in a high cost--therefore hopefully solving for a
    # control value that yields a smaller difference for that variable.
    Qk: list = field(
        default_factory=lambda: np.diag([13.5, 13.5, 13.0, 5.5])
    )  # state error cost matrix, for the the next (T) prediction time steps [x, y, v, yaw]
    # This is the vector of weights that defines how important for our state
    # variables are at the end of the time horizon. I.e., if we want the control
    # values we pick to get us as close as possible to the desired position T
    # timesteps away, then more weight should go to those values.
    Qfk: list = field(
        default_factory=lambda: np.diag([13.5, 13.5, 13.0, 5.5])
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
        super().__init__('mpc_node')
        # TODO: create ROS subscribers and publishers
        #       use the MPC as a tracker (similar to pure pursuit)

        # TODO: May have to change this to PoseWithCovarianceStamped.
        self.__pose_subscriber = self.create_subscription(msg_type=PoseStamped)

        # TODO: get waypoints here
        self.waypoints = None

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

    def pose_callback(self, pose_msg):
        pass
        # TODO: extract pose from ROS msg
        vehicle_state = None

        # TODO: Calculate the next reference trajectory for the next T steps
        #       with current vehicle pose.
        #       ref_x, ref_y, ref_yaw, ref_v are columns of self.waypoints
        # ref_x, ref_y, ref_yaw, ref_v = (0, 0, 0, 0,)
        ref_path = self.calc_ref_trajectory(self, vehicle_state, ref_x, ref_y, ref_yaw, ref_v)
        x0 = [vehicle_state.x, vehicle_state.y, vehicle_state.v, vehicle_state.yaw]

        # TODO: solve the MPC control problem
        (
            self.oa,
            self.odelta,
            ox,
            oy,
            oyaw,
            ov,
            state_predict,
        ) = self.linear_mpc_control(ref_path, x0, self.oa, self.odelta)

        # TODO: publish drive message.
        steer_output = self.odelta[0]
        speed_output = vehicle_state.v + self.oa[0] * self.config.DTK

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
        objective = 0.0  # Objective value of the optimization problem
        constraints = []  # Create constraints array

        # Initialize reference vectors
        self.x0k = cvxpy.Parameter((self.config.NXK,))
        self.x0k.value = np.zeros((self.config.NXK,))

        # Initialize reference trajectory parameter
        self.ref_traj_k = cvxpy.Parameter((self.config.NXK, self.config.TK + 1))
        self.ref_traj_k.value = np.zeros((self.config.NXK, self.config.TK + 1))

        # Initializes block diagonal form of R = [R, R, ..., R] (NU*T, NU*T)
        R_block = block_diag(tuple([self.config.Rk] * self.config.TK))

        # Initializes block diagonal form of Rd = [Rd, ..., Rd] (NU*(T-1), NU*(T-1))
        Rd_block = block_diag(tuple([self.config.Rdk] * (self.config.TK - 1)))

        # Initializes block diagonal form of Q = [Q, Q, ..., Qf] (NX*T, NX*T)
        Q_block = [self.config.Qk] * (self.config.TK)
        Q_block.append(self.config.Qfk)
        Q_block = block_diag(tuple(Q_block))

        # Formulate and create the finite-horizon optimal control problem (objective function)
        # The FTOCP has the horizon of T timesteps

        # IN THIS STEP, we are putting the objective function together. 
        # QUESTION: Are we computing the cost here? Or are we just writing the
        # general functionality for what the cost function will be? I feel like
        # this is where we'd be computing the VALUE of the cost function.
        # Rather, we're telling CVXPY how to set up the optimization problem.
        # What I think this really means is using CVXPYs API to configure all
        # those weight matrices and difference matrices to actually start the
        # optimization process. I.e., the solver needs the objective function,
        # all the optimization variables, and all the constraints (whether
        # represented as scalars, 1D vectors, or matrices) to set everything up,
        # as that is the format that it maintains everything in when it performs
        # the optimization process/algorithm.
        

        # --------------------------------------------------------
        # TODO: fill in the objectives here, you should be using
        # cvxpy.quad_form() somehwhere.

        # TODO: Objective part 1: Influence of the control inputs: Inputs u multiplied by the penalty R


        # TODO: Objective part 2: Deviation of the vehicle from the reference trajectory weighted by Q, including final Timestep T weighted by Qf

        # TODO: Objective part 3: Difference from one control input to the next control input weighted by Rd

        # --------------------------------------------------------

        # Constraints 1: Calculate the future vehicle behavior/states based on the vehicle dynamics model matrices
        # Evaluate vehicle Dynamics for next T timesteps
        A_block = []
        B_block = []
        C_block = []
        # init path to zeros
        # Creates 4x9 matrix. 9 columns for 9 timesteps, where each row in each
        # column is a different state variable.
        path_predict = np.zeros((self.config.NXK, self.config.TK + 1))
        # For each timestep, compute a new linearized model matrix set.
        for t in range(self.config.TK):
            # Pass the velocity (v at index 2) and heading (phi at index 3) and
            # the steering angle as 0 into the get_model_matrix function.
            A, B, C = self.get_model_matrix(
                path_predict[2, t], path_predict[3, t], 0.0
            )
            A_block.append(A)
            B_block.append(B)
            C_block.extend(C)

        A_block = block_diag(tuple(A_block))
        B_block = block_diag(tuple(B_block))
        C_block = np.array(C_block)

        # [AA] Sparse matrix to CVX parameter for proper stuffing
        # Reference: https://github.com/cvxpy/cvxpy/issues/1159#issuecomment-718925710
        m, n = A_block.shape
        # NOTE: we create these model coefficient matrices as CVXPY parameters,
        # as they're not Variables whose value we're trying to optimize, but
        # instead values that we're simply using in the optimization process.
        self.Annz_k = cvxpy.Parameter(A_block.nnz)
        data = np.ones(self.Annz_k.size)
        rows = A_block.row * n + A_block.col
        cols = np.arange(self.Annz_k.size)
        Indexer = csc_matrix((data, (rows, cols)), shape=(m * n, self.Annz_k.size))

        # Setting sparse matrix data
        self.Annz_k.value = A_block.data

        # Now we use this sparse version instead of the old A_ block matrix
        self.Ak_ = cvxpy.reshape(Indexer @ self.Annz_k, (m, n), order="C")

        # Same as A
        m, n = B_block.shape
        self.Bnnz_k = cvxpy.Parameter(B_block.nnz)
        data = np.ones(self.Bnnz_k.size)
        rows = B_block.row * n + B_block.col
        cols = np.arange(self.Bnnz_k.size)
        Indexer = csc_matrix((data, (rows, cols)), shape=(m * n, self.Bnnz_k.size))
        self.Bk_ = cvxpy.reshape(Indexer @ self.Bnnz_k, (m, n), order="C")
        self.Bnnz_k.value = B_block.data

        # No need for sparse matrices for C as most values are parameters
        self.Ck_ = cvxpy.Parameter(C_block.shape)
        self.Ck_.value = C_block

        # -------------------------------------------------------------
        # TODO: Constraint part 1:
        #       Add dynamics constraints to the optimization problem
        #       This constraint should be based on a few variables:
        #       self.xk, self.Ak_, self.Bk_, self.uk, and self.Ck_

        # Above, the sequence of A matrices are generated (as the vehicle model,
        # while unchanging, its coefficients that are computed with other state
        # variables (that do change) change, and therefore the A matrix needs to
        # be recomputed for each of the time steps)). That is done above and the
        # resulting A matrices are put together and diagonalized.

        # NOTE: So, one idea is that I have to construct that big matrix
        # mentioned in the lecture by combining each of the A, B, and C blocks.
        # Ak_, Bk_, Ck_ are all already in that blocked form individually (I
        # think, according to that block_diag function). Therefore, we'd just
        # have to make ONE BIGGER MATRIX out of all those.

        
        # constraints.append(self.Ak_ @ self.xk <= x_max)
        # constraints.append(self.Ak_ @ self.xk >= x_min)

        # constraints.append(self.Bk_ @ self.uk <= u_max)
        # constraints.append(self.Bk_ @ self.uk >= u_min)

        # constraints.append(self.Ck_ @ ????)

        # OR

        # constraints.append(self.Ak_ + self.Bk_ + self.Ck_ @ self.xk) # ????

        # OR:

        for i in range(self.config.TK):
            constraints.append(self.xk[:, i+1] == self.Ak_ @ self.xk[:, i] + self.Bk_ @ self.uk[:, i] + self.Ck_)
        
        # NOTE: Potential problem, though: For a given predicted state xk[i],
        # don't we only want to constrain that to the ith A matrix?? Hard to do
        # though, as the ith A matrix is wrapped up in a block_diag structure,
        # right?? If this is true, then maybe we'd have something like this:
        for i in range(self.config.TK):
            # Get the ith "A" matrix out of the block representation of A.
            a_i = self.Ak_[self.config.NXK*i:self.config.NXK*i + self.config.NXK - 1]
            new_constraint = self.xk[:i+1] == a_i @ self.xk[:, i]
            constraints.append()

        # NOTE Only problem with this, however, is that after getting "blocked,"
        # the A-matrix gets put into a sparse representation from
        # scipy--therefore, I don't think indexing in this way is going to work.

        # Is there a more compact of making this work?
        
        # NOTE: I still don't think the original way I did it was correct,
        # either. That way says "take this singular state vector and multiply it
        # across all N A matrices"--which doesn't make sense, unless there's
        # some way of just isolating that particular state vector and
        # encapsulating it in a vector of the original length.

        # NOTE: If the above case isn't true, then maybe we just have to
        # constrain each of the blocked matrices individually? I.e., 

        # Actually, the model's A matrices aren't just produced above, but they
        # actually produced the A,B,C matrices evalauted over multiple
        # timesteps. How does that make any sense? Wouldn't the state model not
        # change? 

        # I.e., A_block is initially created as a huge matrix, where there is a
        # smaller A matrix at a particular timestep, and the next timestep's
        # followed down and to the right from this one. 

        # The linearized (vehicle) state model is produced/provided above via
        # the get_model_matrix function. Each is provided as a CVXPY parameter.

        # Think elementwise how these constraints work. Each row represents the
        # constraint on each of the variables within our vehicle's state. I.e.,
        # one row is x-position, one row is y-position, etc.

        # AND, the relationship between the next state variable value and its
        # current value and control input are encoded in the coefficients of the
        # A, B, and C matrices. 

        # This relationship or equation or function for EACH variable of our
        # state is essentially a constraint. I.e., you're telling the optimizer
        # that "hey, whatever value you compute for x-position--you must be able
        # to obtain it from this equation here!"

        # Each variable's "constraint equation" is, therefore, encoded by one of
        # the rows across the linearized state model's A, B, and C coefficent
        # matrices.

        # Also worth explaining is that each state variable may be present on
        # more than one *other* state variables (from z) and multiple *control
        # variables* (from u). Therefore, this is why there is a column in A for
        # each state variable in z--because there's a possibility that a state
        # variable is a function of multiple other state variables (and
        # therefore has some coefficient multiplied by each of them, whether 0
        # or some other real number).

        # Likewise, for each control variable (in u), there's a possibility that
        # a state variable is computed as a function of that control variable,
        # times some weihgt/coefficient. Similiarly, then, there's a column
        # in the B matrix for every control variable in our vector u, so as to
        # allow each state variable's row in B to contain the coefficients that
        # each state variable's equation uses to scale each control variable by
        # in the computation of that state variable (just like in A but this
        # time with control variables).

        # THEN, one step further: Rather than leaving A, B, and C as just normal
        # matrices--they take them and create diagonal versions of them. WHAT IS
        # THE LOGIC IN DOING THIS?
        
        # I think it has to do with how constraints are evaluated elementwise
        # when provided as matrices.
        # https://www.cvxpy.org/examples/basic/quadratic_program.html (see "The
        # inequality constraint is elementwise.")

        # Therefore, would we get something like:
        # A @ z = z ?????? 
        # AND
        # B @ u = z ???? That doesn't make sense though.
        # OR
        # A @ z + B @u + C = z????? Not even sure if that makes sense.

        # NEED TO LOOK UP AND UNDERSTAND WHAT CVXPY'S "@" OPERATOR DOES!!

        # So, we actually get these linearized, descritized vehicle model
        # equations (in the form of A, B, and C matrices) from the
        # "get_model_matrix" helper function above.

        # Dvij explained that to "linearize" these models, it's nothing
        # fancy-- x' = cos(theta), but to linearize this, we can just make a
        # linear approximation of the function at some point (theta), which will
        # just give us the tangent line at that theta == a linear function.

        # To descritize the function--that I'm still not sure of. So, if it's
        # just a matter of taking that tangent line and turning it into a
        # "discrete" function, then I think that's just a matter of "sampling
        # points" along that tangent line--I.e., computing values at each of the
        # discrete timesteps along the tangent line?? Or doing this using the
        # slope? Something like that. Look at their function to figure out how
        # this is done.
        
        # TODO: Constraint part 2:
        #       Add constraints on steering, change in steering angle
        #       cannot exceed steering angle speed limit. Should be based on:
        #       self.uk, self.config.MAX_DSTEER, self.config.DTK
        min_steering_angle_constraint = self.uk[0:] <= self.config.MIN_STEER
        constraints.append(min_steering_angle_constraint)
        max_steering_angle_constraint = self.uk[0:] <= self.config.MAX_STEER
        constraints.append(max_steering_angle_constraint)
        # The change per unit of time must not exceed the maximum rate of change
        # of the steering angle from one computed steering angle to the next.
        for i in range(self.config.TK):
            new_max_steering_rate_constraint = (self.uk[0:i+1] - self.uk[0:i])/self.config.TK <= self.config.MAX_DSTEER
            constraints.append(new_max_steering_rate_constraint)

        # First dimension / index is the row we want to select. In this case,
        # I'm saying that I want to get ALL of the values in the first row.

        # Is there a better way of doing this in a more elementwise way? I COULD
        # make an "upper bound" and "lower bound" array like shown in the MPC
        # lecture. I.e., for the bounds on my control values (u) I'd have a 2D
        # array, where the number of rows == number of control values, and then
        # number of columns would similarly be the number of timesteps in the
        # horizon, or the number of solved-for optimal control values (N+1).
        # I.e., it would have the same dimensions as uk. However, because this
        # would be a lot of repeated values, I'm not sure that representation is
        # as useful for this particular use case, right?

        # This is where we specify the constraints. I.e., steering limits,
        # acceleration limits.

        # TODO: Constraint part 3:
        #       Add constraints on upper and lower bounds of states and inputs
        #       and initial state constraint, should be based on:
        #       self.xk, self.x0k, self.config.MAX_SPEED, self.config.MIN_SPEED,
        #       self.uk, self.config.MAX_ACCEL, self.config.MAX_STEER
        
        # -------------------------------------------------------------

        # Create the optimization problem in CVXPY and setup the workspace
        # Optimization goal: minimize the objective function
        self.MPC_prob = cvxpy.Problem(cvxpy.Minimize(objective), constraints)

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

        A_block = []
        B_block = []
        C_block = []
        for t in range(self.config.TK):
            A, B, C = self.get_model_matrix(
                path_predict[2, t], path_predict[3, t], 0.0
            )
            A_block.append(A)
            B_block.append(B)
            C_block.extend(C)

        A_block = block_diag(tuple(A_block))
        B_block = block_diag(tuple(B_block))
        C_block = np.array(C_block)

        self.Annz_k.value = A_block.data
        self.Bnnz_k.value = B_block.data
        self.Ck_.value = C_block

        self.ref_traj_k.value = ref_traj

        # Solve the optimization problem in CVXPY
        # Solver selections: cvxpy.OSQP; cvxpy.GUROBI
        self.MPC_prob.solve(solver=cvxpy.OSQP, verbose=False, warm_start=True)

        if (
            self.MPC_prob.status == cvxpy.OPTIMAL
            or self.MPC_prob.status == cvxpy.OPTIMAL_INACCURATE
        ):
            ox = np.array(self.xk.value[0, :]).flatten()
            oy = np.array(self.xk.value[1, :]).flatten()
            ov = np.array(self.xk.value[2, :]).flatten()
            oyaw = np.array(self.xk.value[3, :]).flatten()
            oa = np.array(self.uk.value[0, :]).flatten()
            odelta = np.array(self.uk.value[1, :]).flatten()

        else:
            print("Error: Cannot solve mpc..")
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
    print("MPC Initialized")
    mpc_node = MPC()
    rclpy.spin(mpc_node)

    mpc_node.destroy_node()
    rclpy.shutdown()
