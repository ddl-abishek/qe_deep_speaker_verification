import os 

for i in range(100):
    cmd = f'cp ./checkpoint /mnt/artifacts/results/checkpoint{i}'
    os.system(cmd)