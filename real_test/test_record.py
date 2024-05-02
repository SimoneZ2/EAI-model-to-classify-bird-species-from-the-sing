import serial
import re
import time


power_consumption_active_mode = 62
energy_battery_pack = 2 * 3900


start_collecting_time = 0
collecting_data_time = 0
status_prediction = False

ser = serial.Serial('COM7', 115200, timeout=1)

while not status_prediction:
    line = ser.readline().decode('utf-8').strip()
    if line == "Collecting Data...":
        print("Collecting Data...")
        if start_collecting_time == 0:
            start_collecting_time = time.time()
        collecting_data_time = time.time() - start_collecting_time

    elif not re.search('[a-zA-Z]', line):
        print("Prediction result:", line)
        status_prediction = True

active_mode_duration = collecting_data_time
total_time = collecting_data_time
low_power_mode_duration = total_time - active_mode_duration
average_power_consumption = (active_mode_duration * power_consumption_active_mode + low_power_mode_duration * 0)
predicted_autonomy = energy_battery_pack / average_power_consumption

print("Energy consumption:", average_power_consumption, "mW")
print("Predicted autonomy with batteries:", predicted_autonomy, "hours")