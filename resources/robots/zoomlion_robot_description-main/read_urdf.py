import xml.etree.ElementTree as ET
import pandas as pd
import argparse
import os

def parse_urdf_to_csv(urdf_file, output_dir="."):
    """
    解析URDF文件并将关节和连杆参数导出为CSV文件
    
    Args:
        urdf_file (str): URDF文件路径
        output_dir (str): 输出目录
    """
    
    # 解析URDF文件
    try:
        tree = ET.parse(urdf_file)
        root = tree.getroot()
    except Exception as e:
        print(f"解析URDF文件时出错: {e}")
        return
    
    # 存储关节和连杆数据
    joints_data = []
    links_data = []
    
    # 解析所有连杆
    for link in root.findall('link'):
        link_data = parse_link(link)
        if link_data:
            links_data.append(link_data)
    
    # 解析所有关节
    for joint in root.findall('joint'):
        joint_data = parse_joint(joint)
        if joint_data:
            joints_data.append(joint_data)
    
    # 创建DataFrame
    joints_df = pd.DataFrame(joints_data)
    links_df = pd.DataFrame(links_data)
    
    # 生成输出文件名
    base_name = os.path.splitext(os.path.basename(urdf_file))[0]
    joints_file = os.path.join(output_dir, f"{base_name}_joints.csv")
    links_file = os.path.join(output_dir, f"{base_name}_links.csv")
    
    # 保存为CSV文件
    joints_df.to_csv(joints_file, index=False, encoding='utf-8')
    links_df.to_csv(links_file, index=False, encoding='utf-8')
    
    print(f"关节参数已保存到: {joints_file}")
    print(f"连杆参数已保存到: {links_file}")
    print(f"共解析 {len(joints_data)} 个关节和 {len(links_data)} 个连杆")

def parse_link(link_element):
    """解析连杆元素"""
    link_name = link_element.get('name', '')
    
    # 解析惯性参数
    inertial = link_element.find('inertial')
    mass = 0.0
    inertia = {
        'ixx': 0.0, 'ixy': 0.0, 'ixz': 0.0,
        'iyy': 0.0, 'iyz': 0.0, 'izz': 0.0
    }
    com_xyz = "0 0 0"  # 质心位置
    com_rpy = "0 0 0"  # 质心姿态
    
    if inertial is not None:
        # 质心位置和姿态
        origin = inertial.find('origin')
        if origin is not None:
            com_xyz = origin.get('xyz', '0 0 0')
            com_rpy = origin.get('rpy', '0 0 0')
        
        # 质量
        mass_elem = inertial.find('mass')
        if mass_elem is not None:
            mass = float(mass_elem.get('value', 0.0))
        
        # 惯性矩阵
        inertia_elem = inertial.find('inertia')
        if inertia_elem is not None:
            inertia = {
                'ixx': float(inertia_elem.get('ixx', 0.0)),
                'ixy': float(inertia_elem.get('ixy', 0.0)),
                'ixz': float(inertia_elem.get('ixz', 0.0)),
                'iyy': float(inertia_elem.get('iyy', 0.0)),
                'iyz': float(inertia_elem.get('iyz', 0.0)),
                'izz': float(inertia_elem.get('izz', 0.0))
            }
    
    return {
        'name': link_name,
        'mass': mass,
        'com_xyz': com_xyz,      # 质心位置
        'com_rpy': com_rpy,      # 质心姿态
        'inertia_ixx': inertia['ixx'],
        'inertia_ixy': inertia['ixy'],
        'inertia_ixz': inertia['ixz'],
        'inertia_iyy': inertia['iyy'],
        'inertia_iyz': inertia['iyz'],
        'inertia_izz': inertia['izz']
    }

def parse_joint(joint_element):
    """解析关节元素"""
    joint_name = joint_element.get('name', '')
    joint_type = joint_element.get('type', '')
    
    # 解析父子关系
    parent = joint_element.find('parent')
    child = joint_element.find('child')
    parent_link = parent.get('link', '') if parent is not None else ''
    child_link = child.get('link', '') if child is not None else ''
    
    # 解析原点位置和姿态
    origin = joint_element.find('origin')
    origin_xyz = "0 0 0"
    origin_rpy = "0 0 0"
    
    if origin is not None:
        origin_xyz = origin.get('xyz', '0 0 0')
        origin_rpy = origin.get('rpy', '0 0 0')
    
    # 解析轴
    axis_elem = joint_element.find('axis')
    axis = "1 0 0"  # 默认X轴
    if axis_elem is not None:
        axis = axis_elem.get('xyz', '1 0 0')
    
    # 解析限制参数
    limit_elem = joint_element.find('limit')
    effort = 0.0
    velocity = 0.0
    lower = 0.0
    upper = 0.0
    
    if limit_elem is not None:
        effort = float(limit_elem.get('effort', 0.0))
        velocity = float(limit_elem.get('velocity', 0.0))
        lower = float(limit_elem.get('lower', 0.0))
        upper = float(limit_elem.get('upper', 0.0))
    
    # 解析动力学参数
    dynamics_elem = joint_element.find('dynamics')
    damping = 0.0
    friction = 0.0
    
    if dynamics_elem is not None:
        damping = float(dynamics_elem.get('damping', 0.0))
        friction = float(dynamics_elem.get('friction', 0.0))
    
    return {
        'name': joint_name,
        'type': joint_type,
        'parent_link': parent_link,
        'child_link': child_link,
        'origin_xyz': origin_xyz,
        'origin_rpy': origin_rpy,
        'axis': axis,
        'effort_limit': effort,
        'velocity_limit': velocity,
        'lower_limit': lower,
        'upper_limit': upper,
        'damping': damping,
        'friction': friction
    }

def main():
    parser = argparse.ArgumentParser(description='将URDF文件转换为关节和连杆的CSV表格')
    parser.add_argument('urdf_file', help='URDF文件路径')
    parser.add_argument('-o', '--output', default='.', help='输出目录 (默认: 当前目录)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.urdf_file):
        print(f"错误: 文件 '{args.urdf_file}' 不存在")
        return
    
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    parse_urdf_to_csv(args.urdf_file, args.output)

if __name__ == "__main__":
    main()