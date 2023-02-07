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
    
	
   def __init__(self,m1,m2,Lp):
      self.g = 9.81
      self.m1 = m1
      self.m2v= m2
      self.Lp = Lp
    		   
      self.f0
      self.t0
      self.f1
      self.t1
    	   
      self.Kp
      self.Kd
    	   
      self.q1
      self.q2
      self.qd1
      self.qd2
      
        
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
    # Write your code here
    ............


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
    # Write your code here
    ............    
    # sweep function should be called here: sweep(t, data.t0, data.f0, data.t1, data.f1, data.Fmax)


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
    ............
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
    ............
  


# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# Main function

if __name__ == '__main__':
    mbs_data = MBSData()
    
    compute_dynamic_response(mbs_data)  
