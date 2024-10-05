import mujoco_py
import numpy as np

# 加载模型
model = mujoco_py.load_model_from_path('safety-gymnasium/safety_gymnasium/assets/xmls/quadrotor.xml')
sim = mujoco_py.MjSim(model)

# 设置渲染器
viewer = mujoco_py.MjViewer(sim)

# 仿真参数
step_count = 1000
control_force = 0.1  # 设置一个控制输入

# 仿真和渲染循环
for step in range(step_count):
    # 为关节应用控制力
    sim.data.ctrl[0] = control_force

    # 进行一步仿真
    sim.step()

    # 渲染仿真结果
    viewer.render()

    # 打印位置信息以观察变化
    print("Step:", step, "Position:", sim.data.qpos[:])

# 关闭渲染器
viewer.close()
