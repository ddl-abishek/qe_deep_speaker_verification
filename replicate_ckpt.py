import os

for i in range(50):
    cmd = f'cp ./checkpoints/checkpoint_vivek ./checkpoints/checkpoint_vivek{4899+i}'
    os.system(cmd)
