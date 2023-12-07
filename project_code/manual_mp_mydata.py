import time
import os 
import subprocess
num_mp = os.cpu_count()
for i in range(num_mp):
    if i > 1:
        break
    print(f"==============================start {i=}=========================")
    cmd = f"/z/home/yayuanli/Research/darpa_ptg/darpa_ptg_yayuan/ptg_research/exp4/env_exp4/bin/python EpicKitchens50.py 001 xx {num_mp} {i}"
    # os.system(cmd)
    subprocess.Popen(cmd, shell=True, stdout=open("../outputs/tmp/manual_mp_mydata.out.", "w"))
    time.sleep(1)
    
   
print(f"all started w/o handle. pending by sleep")
time.sleep(3600*48)
print(f"EOF")
