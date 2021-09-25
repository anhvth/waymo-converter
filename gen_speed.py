from glob import glob
import time
import mmcv



def get_num():
    return len(glob('./data/images/*'))

start_num = get_num()
timer = mmcv.Timer()
while True:
    delta_n = get_num()- start_num
    delta_t = timer.since_start()
    speed = delta_n/delta_t
    print(f"\rSpeed: {speed:0.2f} imgs/s", end='')
    
    time.sleep(1)