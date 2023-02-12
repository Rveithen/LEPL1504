#
# LEPL1504 -  MISSION 1  -  Modeling a cart-pendulum system
#
# @date 2022
# @author Robotran team
# 
# Universite catholique de Louvain


# import useful modules and functions
from math import sin, cos, pi
import numpy as np
from scipy.integrate import solve_ivp


# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
class MBSData:
    """ Class to define the parameters of the double mass-spring model
    and of the problem to solve.
     
    It contains the following fields (in SI units): 
    
    general parameters:
    -------------------
    g:     Gravity
    
    masses:
    -------
    m1:    Cart mass
    m2:    Pendulum mass
    
    parameters of the pendulum:
    ---------------------------
    Lp:    Pendulum length
    
    parameter of the external force:
    --------------------------------
    Fmax:  Semi-amplitude of the force
    f0:    Frequency at start
    t0:    Time for specifying frequency at start
    f1:    Frequency at end
    t1:    Time for specifying frequency at end
    
    parameter of the controller:
    ----------------------------
    Kp:    Proportional factor
    Kd:    Differential factor
    
    initial positions and velocities:
    ---------------------------------
    q1:    Initial position coordinate of the cart
    q2:    Initial position coordinate of the pendulum
    qd1:   Initial velocity coordinate of the cart
    qd2:   Initial velocity coordinate of the pendulum
    """
    
    def __init__(self):
        self.g = 9.81
        
        self.m1 = 1
        self.m2 = 0.5
        
        self.Lp = 1
        
        self.Fmax = 0.5  		   
        self.f0 = 1
        self.t0 = 0
        self.f1 = 10
        self.t1 = 10
          	   
        self.Kp = 1
        self.Kd = 1
          	   
        self.q1 = 0
        self.q2 = pi/6
        self.qd1 = 0
        self.qd2 = 0
    
        
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
def sweep(t, t0, f0, t1, f1, Fmax):
    """ Compute the value of a force sweep function at the given time.
    The sweep function has a sinusoidal shape with constant amplitude 
    and a varying frequency. This function enables to consider linear
    variation of the frequency between f0 and f1 defined at time t0 and t1.

	:param t: the time instant when to compute the function.
	:param t0: the time instant when to specify f0.
	:param f0: the frequency at time t0.
	:param t1: the time instant when to specify f1.
	:param f1: the frequency at time t1.
	:param Fmax: the semi-amplitude of the function.
		
	:return Fext: the value of the sweep function.
    """
    return Fmax * sin(2 * pi * t * (f0 + t * (f1-f0) / (2 * (t1-t0))))
    

# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *    
def compute_derivatives(t, y, data):
    """ Compute the derivatives yd for a given state y of the system.
        The derivatives are computed at the given time t with
        the parameters values in the given data structure.
        
        It is assumed that the state vector y contains the following states:
          y = [q1, q2, qd1, qd2] with:
             - q1: the cart position
             - q2: the pendulum position 
             - qd1: the cart velocity
             - qd2: the pendulum velocity 

        :param  t: the time instant when to compute the derivatives.
        :param  y: the numpy array containing the states 
        :return: yd a numpy array containing the states derivatives  yd = [qd1, qd2, qdd1, qdd2]
        :param data: the MBSData object containing the parameters of the model
    """                 
    F = sweep(t, data.t0, data.f0, data.t1, data.f1, data.Fmax)
    cost = cos(y[1])
    sint = sin(y[1])
    q2ds = y[3] * y[3]
    Ls = data.Lp * data.Lp
    
    M = np.array([[data.m1 + data.m2 , data.m2 * data.Lp * cost /2], [cost, 2 * data.Lp /3]])
    Q_c = np.array([F + (data.m2 * Ls * q2ds * sint/2), data.g *sint])
    
    qdd = np.linalg.solve(M,Q_c)
    
    yd = np.array([y[2], y[3], qdd[0], qdd[1]])
    
    return yd


# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
def compute_dynamic_response(data):
    """  Compute the time evolution of the dynamic response of the cart-pendulum system
         for the given data. Initial and final time are determined
         by the t0 and t1 parameter of the parameter data structure.
         Results are saved to three text files named dirdyn_q.res, dirdyn_qd.res and dirdyn_qdd.res
 
        Time evolution is computed using an time integrator (typically Runge-Kutta).
 
       :param data: the MBSData object containing the parameters of the model
     """
    # Write your code here
    # ### Runge Kutta ###   should be called via solve_ivp()
    # to pass the MBSData object to compute_derivative function in solve_ivp, you may use lambda mechanism:
    #
    #    fprime = lambda t,y: compute_derivatives(t, y, data)
    #
    # fprime can be viewed as a function that takes two arguments: t, y
    # this fprime function can be provided to solve_ivp
    # Note that you can change the tolerances with rtol and atol options (see online solve_iv doc)
    #
    # Write some code here
    fprime = lambda t,y: compute_derivatives(t, y, data)
    
    t = np.linspace(data.t0, data.t1, 101)
    
    y = np.array([data.q1, data.q2, data.qd1, data.qd2])
    
    h = solve_ivp(fprime, [data.t0, data.t1], y, t_eval=t, method='RK23')
    
    q1 = h.y[0]
    q2 = h.y[1]
    qd1 = h.y[2]
    qd2 = h.y[3]
    
    qdd1 = np.zeros(len(q1))
    qdd2 = np.zeros(len(q1))
    
    for i in range(len(q1)):
        qdd = compute_derivatives(h.t[i], [q1[i],q2[i],qd1[i],qd2[i]], data)
        qdd1[i] = qdd[2]
        qdd2[i] = qdd[3]

    np.savetxt("dirdyn_q.res", np.array([h.t, q1, q2]).transpose())
    np.savetxt("dirdyn_qd.res", np.array([h.t, qd1, qd2]).transpose())
    np.savetxt("dirdyn_qdd.res", np.array([h.t, qdd1, qdd2]).transpose())
    
    return h


# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# Main function

if __name__ == '__main__':
    mbs_data = MBSData(1, 0.5, 1)
    mbs_data.Fmax = 0.5
    mbs_data.f0 = 1
    mbs_data.f1 = 10
    mbs_data.t0 = 0
    mbs_data.t1 = 10
    mbs_data.q1 = 0
    mbs_data.q2 = 30*pi/180
    mbs_data.qd1 = 0
    mbs_data.qd2 = 0

    compute_dynamic_response(mbs_data)

    sol = np.loadtxt("dirdyn_q.res")
    ref = np.loadtxt("dirdyn_positions_ref.res")
    plt.figure(figsize=(7, 4.5))
    plt.subplot(2,1,1)
    plt.plot(sol[:, 0], sol[:, 1])
    plt.plot(ref[:, 0], ref[:, 1], "r.")
    plt.subplot(2,1,2)
    plt.plot(ref[:, 0], ref[:, 2], "r. ")
    plt.plot(sol[:, 0], sol[:, 2])

    plt.xlabel("Temps[s]")
    plt.ylabel("Position[m]")
    plt.legend(["Solution du programme","Reference"])
    plt.show()
