import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__))+'/..')
from emioapi import EmioMotors
import argparse
import csv
from emioapi._logging_config import logger
import logging

logger.setLevel(logging.WARNING) 

SAVE_FILENAME = 'motors_pid_position_data.csv'

RAW_PER_PUSLE_TO_VOLT_PER_RAD = (12/885)/(2*3.14/4096)
VOLT_PER_RAD_TO_RAW_PER_PULSE = 1/RAW_PER_PUSLE_TO_VOLT_PER_RAD
DT = 2/1000 # control loop time step in seconds (2 ms)

MAX_GAIN_TABLE = 16383
MAX_KP = np.round(MAX_GAIN_TABLE/(128*VOLT_PER_RAD_TO_RAW_PER_PULSE),decimals=2)
MAX_KI = np.round(MAX_GAIN_TABLE/(65536*VOLT_PER_RAD_TO_RAW_PER_PULSE*DT),decimals=2)
MAX_KD = np.round(MAX_GAIN_TABLE/(16*VOLT_PER_RAD_TO_RAW_PER_PULSE/DT),decimals=2)


def Kp_to_Kptable(Kp):
    return Kp*128*VOLT_PER_RAD_TO_RAW_PER_PULSE



def Ki_to_Kitable(Ki):
    return Ki*65536*VOLT_PER_RAD_TO_RAW_PER_PULSE*DT

def Kd_to_Kdtable(Kd):
    return Kd*16*VOLT_PER_RAD_TO_RAW_PER_PULSE/DT

def main(Kp_user:float=0, Ki_user:float=0, Kd_user:float=0,duration:float=0.3, reference:float=0.1):

    # convert to table values
    Kp = Kp_to_Kptable(Kp_user)
    Ki = Ki_to_Kitable(Ki_user)
    Kd = Kd_to_Kdtable(Kd_user)
    

    # open motors
    motors = EmioMotors()
    while not motors.open():
        logger.info("Waiting for motors to open...")
        time.sleep(1)
    logger.info("Motors opened successfully.")

    # initial and target angles
    init_angles = np.array([0.0, 0, 0.0, 0])
    target_angles = init_angles + np.array([reference, 0, 0, 0])
    motors.angles = init_angles
    time.sleep(1)

    # set first set of gains
    motors.position_p_gain = [200, 200, 200, 200]
    motors.position_i_gain = [0, 0, 0, 0]
    motors.position_d_gain = [0, 0, 0, 0]
    print("Set default PID gains")
    time.sleep(1)

    p_gains = motors.position_p_gain
    i_gains = motors.position_i_gain
    d_gains = motors.position_d_gain
    logger.info(f"Current position P gains: {p_gains}")
    logger.info(f"current position i gains: {i_gains}")
    logger.info(f"Current position D gains: {d_gains}")

    # move to target and record response
    measures = [motors.angles]
    motors.angles = target_angles
    times = [time.time()]
    t0 = time.time()
    while time.time() - t0 < duration:
        measures.append(motors.angles)
        times.append(time.time())
    time.sleep(1)
    nb_steps1 = len(measures)


    motors.angles = init_angles
    time.sleep(1)
    motors.position_p_gain = [int(Kp), 800, 800, 800]
    motors.position_i_gain = [int(Ki), 0, 0, 0]
    motors.position_d_gain = [int(Kd), 0, 0, 0]
    print(f"Set user PID gains Kp = {Kp_user},Kd = {Kd_user},Ki = {Ki_user}")
    time.sleep(0.1)

    p_gains = motors.position_p_gain
    i_gains = motors.position_i_gain
    d_gains = motors.position_d_gain
    logger.info(f"Updated position P gains: {p_gains}")
    logger.info(f"Updated position I gains: {i_gains}")
    logger.info(f"Updated position D gains: {d_gains}")

    # move to target and record response
    measures.append(motors.angles)
    motors.angles = target_angles
    times.append(time.time())
    t0 = time.time()
    while time.time() - t0 < duration:
        measures.append(motors.angles)
        times.append(time.time())
    time.sleep(1)

    # setting fisrt PID gains back to default
    motors.position_p_gain = [800, 800, 800, 800]
    motors.position_i_gain = [0, 0, 0, 0]
    motors.position_d_gain = [0, 0, 0, 0]
    logger.info("Reset PID gains to default.")
    # putting motors to initial position
    motors.angles = init_angles
    time.sleep(1)

    motors.close()
    logger.info("Motors closed.")

    # process data
    measures = np.array(measures)
    measures1 = measures[:nb_steps1] - init_angles
    measures2 = measures[nb_steps1:] - init_angles
    times = np.array(times)
    times1 = times[:nb_steps1] - times[0]
    times2 = times[nb_steps1:] - times[nb_steps1]
    timesRef = times1 if len(times1) > len(times2) else times2
    measuresRef = [target_angles[0] - init_angles[0]] * len(timesRef)

    # Plot to compare the two responses
    print("\n"*2)
    print("-"*20)
    print(f"Plotting results and saving to {SAVE_FILENAME}")
    print("Close the plot window to finish.")
    plt.figure()
    plt.plot(timesRef, measuresRef, "-r", label="ref")
    plt.plot(times1, measures1[:, 0], "--", label="default PID")
    plt.plot(times2, measures2[:, 0], "--", label="user PID")
    plt.xlabel("Time [s]")
    plt.ylabel("Angle [rad]")
    plt.title("Motor Position Control with Different PID Gains")
    plt.legend()
    

    with open(SAVE_FILENAME, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Time (s)', 'Reference (rad)', 'default PID (rad)', 'user PID (rad)'])
        for t, ref, pid1, pid2 in zip(timesRef, measuresRef, measures1[:, 0], measures2[:, 0]):
            writer.writerow([t, ref, pid1, pid2])

    plt.show()

def Kp_float(x):
    try:
        x = float(x)
        logger.info(f"Received Kp value: {x}  = {Kp_to_Kptable(x)} in table value")
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))

    if x <= -MAX_KP or x >= MAX_KP:
        raise argparse.ArgumentTypeError("%r not in range [-%r, %r]"%(x, MAX_KP, MAX_KP))
    return x

def Ki_float(x):
    try:
        x = float(x)
        logger.info(f"Received Ki value: {x}  = {Ki_to_Kitable(x)} in table value")
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))

    if x <= -MAX_KI or x >= MAX_KI:
        raise argparse.ArgumentTypeError("%r not in range [-%r, %r]"%(x, MAX_KI, MAX_KI))
    return x

def Kd_float(x):
    try:
        x = float(x)
        logger.info(f"Received Kd value: {x}  = {Kd_to_Kdtable(x)} in table value")
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))

    if x <= -MAX_KD or x >= MAX_KD:
        raise argparse.ArgumentTypeError("%r not in range [-%r, %r]"%(x, MAX_KD, MAX_KD))
    return x

def Ki_float(x):
    try:        
        x = float(x)
        logger.info(f"Received Ki value: {x}  = {Ki_to_Kitable(x)} in table value")
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))  


    if x <= -MAX_KI or x >= MAX_KI:
        raise argparse.ArgumentTypeError("%r not in range [-%r, %r]"%(x, MAX_KI, MAX_KI))
    return x




if __name__ == "__main__":
    # retrieve command line arguments for PID gains, duration, and reference position
    parser = argparse.ArgumentParser(description="Control DYNAMIXEL motors in position mode with PID control.")
    parser.add_argument("Kp", type=Kp_float, help="Proportional gain for PID control. Range: [-%r, %r]" % (MAX_KP, MAX_KP), default=1.0)
    parser.add_argument("Ki", type=Ki_float, help="Integral gain for PID control. Range: [-%r, %r]" % (MAX_KI, MAX_KI), default=0)
    parser.add_argument("Kd", type=Kd_float, help="Derivative gain for PID control. Range: [-%r, %r]" % (MAX_KD, MAX_KD)    , default=0)
    parser.add_argument("--duration", type=float, help="Duration of the second PID test in seconds.", default=0.3)
    parser.add_argument("--reference", type=float, help="Reference position in radians.", default=0.1)
   
    if(len(sys.argv) == 1):
        print("\n"*2)
        print("No arguments provided. Using default values: Kp=10.0 Ki = 0.0 Kd = 0.0\nUse --help to see available options.\n")
        args = parser.parse_args(["10.0", "0" ,"0"])
    else:
        args = parser.parse_args()
    

    input("Press Enter to start the test with the specified PID gains...")

    main(Kd_user=args.Kd, Ki_user=args.Ki, Kp_user=args.Kp, duration=args.duration, reference=args.reference)
