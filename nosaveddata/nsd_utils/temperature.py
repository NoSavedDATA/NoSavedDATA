import psutil
import subprocess


def get_gpu_cpu_temps():
    
    output = subprocess.check_output(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"]).decode("utf-8")
    temperatures=[str(output.strip())]
    
    for core, temp in enumerate(psutil.sensors_temperatures()['coretemp']):
        if 'Package' not in temp[0]:
            print(temp[1])
        temperatures.append(str(temp[1]))
    temperatures = '|'.join(temperatures)


    return temperatures