from glob import glob
import time
# import mmcv



def get_num():
    return len(glob('./data/image/*/*'))

start_num = get_num()
# timer = mmcv.Timer()
start = time.time()
while True:
    n = get_num()
    delta_n = n- start_num
    delta_t = time.time()-start
    speed = delta_n/delta_t
    print(f"\rSpeed: {speed:0.2f} imgs/s | Total: {n}", end='')
    
    time.sleep(10)