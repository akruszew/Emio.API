#!/usr/bin/env -S uv run --script

import time
import logging
import os
import sys
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
sys.path.append(os.path.dirname(os.path.realpath(__file__))+'/..')
from emioapi import EmioMotors
from emioapi._logging_config import logger
import argparse


'''
This example demonstrates how to use the EMIO API to control DYNAMIXEL motors in PWM mode.
'''

def main(emio: EmioMotors, u1 = 200, u2 = 400, loops=500, motor_id=1):
    '''
        Main function to run the PWM test.
        Parameters:
        - emio: An instance of the EmioMotors class.
        - u1: First PWM value to set.
        - u2: Second PWM value to set.
        - loops: Number of iterations to run the test.
        - motor_id: ID of the motor to test (0-3).
    '''
    # Enable PWM mode
    emio._mg.enablePWMMode()
    time.sleep(1)
    # emio.printStatus()

    # Intialize lists to store data for plotting
    start = time.perf_counter()
    last = start
    dt = []
    speed = []
    pwm = []
    t = []
    pwm_value = [0]*4

    # Loop to set PWM values and read velocity
    for i in range(loops):
        # Record loop time and elapsed time
        new = time.perf_counter()
        dt.append(new-last)
        t.append(new-start)
        last = new

        # Alternate PWM values for the first half and second half of the loops
        if i<loops/2:
            pwm_value[motor_id] = u1
        else:
            pwm_value[motor_id] = u2

        emio.goal_pwm = pwm_value
        pwm.append(pwm_to_volt(pwm_value[motor_id]))
        
        velocity = emio.velocity
        speed.append(raw_velocity_to_rpm(velocity[motor_id]))
                
            
    emio.goal_pwm=[0]*4
    logger.info("PWM test completed. Plotting results...")
    logger.info(f"Average loop time: {np.mean(dt):.4f} seconds")
    logger.info("Plotting PWM and velocity over time...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.plot(t, pwm, '-+')
    ax1.legend(["voltage (V)"])
    ax2.plot(t, speed, '-+')
    ax2.legend(["velocity (rpm)"])
    plt.show()



def volt_to_pwm(volt):
    '''
        Convert voltage to PWM value.
        Parameters:
        - volt: Voltage value to convert.
        Returns:
        - PWM value corresponding to the input voltage.
    '''
    return int(volt/12*885)

def pwm_to_volt(pwm):
    '''
        Convert PWM value to voltage.
        Parameters:
        - pwm: PWM value to convert.
        Returns:
        - Voltage corresponding to the input PWM value.
    '''
    return pwm/885*12

def raw_velocity_to_rpm(raw_velocity):
    '''
        Convert raw velocity value from the motor to RPM.
        Parameters:
        - raw_velocity: Raw velocity value from the motor.
        Returns:
        - RPM value corresponding to the input raw velocity.
    '''
    return raw_velocity * 0.229

  

if __name__ == "__main__":
    
    emio_motors = None
    # retreive command line arguments for PWM values and loops

    parser = argparse.ArgumentParser(description="Control DYNAMIXEL motors in PWM mode.")
    parser.add_argument("u1", type=float, help="First voltage value to set.")
    parser.add_argument("u2", type=float, help="Second voltage value to set.")
    parser.add_argument("--loops", type=int, help="Number of iterations to run the test.", default=500)
    
    args = parser.parse_args()

    u1 = volt_to_pwm(args.u1)
    u2 = volt_to_pwm(args.u2)
    loops = args.loops
    

    try:
        logger.info("Starting EMIO API test...")
        logger.info("Opening and configuring EMIO API...")      
        emio_motors = EmioMotors()
        
        if emio_motors.open(): 
            
            #emio_motors.printStatus()

            logger.info("Emio motors opened and configured.")
            logger.info("Running main function...")
            main(emio_motors, u1=u1, u2=u2, loops=loops, motor_id=1)

            logger.info("Main function completed.")
            logger.info("Closing Emio motor connection...")

            emio_motors.close()
            logger.info("Emio connection closed.")
    except KeyboardInterrupt:
        if emio_motors:
            emio_motors.goal_pwm=[0]*4
            plt.close("all")
            emio_motors.close()
    except Exception as e:
        logger.exception(f"An error occurred: {e}")
        if emio_motors:
            emio_motors.close()