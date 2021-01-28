import os

for i in range(10):
    cmd = f'cp ./checkpoints/checkpoint_vivek ./checkpoints/checkpoint_vivek{i}'
    os.system(cmd)
