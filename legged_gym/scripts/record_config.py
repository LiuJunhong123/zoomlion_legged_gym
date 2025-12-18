import os
from legged_gym import LEGGED_GYM_ENVS_DIR

def record_config(log_root, name="a1_amp"):
    log_dir=log_root
    os.makedirs(log_dir, exist_ok=True)

    str_config = name + '_config.txt'
    file_path1=os.path.join(log_dir, str_config)
    file_path2=os.path.join(log_dir, 'legged_robot_config.txt')
    file_path3=os.path.join(log_dir, name + '.txt')
    file_path4=os.path.join(log_dir, 'legged_robot_walk.txt')
    file_path5=os.path.join(log_dir, 'legged_robot_stand.txt')
    
    root1 = name.split('_')[0]

    root_path1 = os.path.join(LEGGED_GYM_ENVS_DIR, root1, name + '_config.py')
    root_path2 = os.path.join(LEGGED_GYM_ENVS_DIR, 'base', 'legged_robot_config.py')
    root_path3 = os.path.join(LEGGED_GYM_ENVS_DIR, root1, name + '.py')
    root_path4 = os.path.join(LEGGED_GYM_ENVS_DIR, 'base', 'legged_robot_walk.py')
    root_path5 = os.path.join(LEGGED_GYM_ENVS_DIR, 'base', 'legged_robot_stand.py')

    with open(root_path1, 'r', encoding='utf-8') as file:
        content = file.read()

    with open(file_path1, 'w', encoding='utf-8') as file:
        file.write(content)

    with open(root_path2, 'r',encoding='utf-8') as file:
        content = file.read()

    with open(file_path2, 'w', encoding='utf-8') as file:
        file.write(content)

    with open(root_path3, 'r',encoding='utf-8') as file:
        content = file.read()

    with open(file_path3, 'w', encoding='utf-8') as file:
        file.write(content)

    with open(root_path4, 'r',encoding='utf-8') as file:
        content = file.read()

    with open(file_path4, 'w', encoding='utf-8') as file:
        file.write(content)

    with open(root_path5, 'r',encoding='utf-8') as file:
        content = file.read()

    with open(file_path5, 'w', encoding='utf-8') as file:
        file.write(content)