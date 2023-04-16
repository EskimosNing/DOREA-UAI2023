#### 多议题协商环境

- 基础协商策略包括：基于时间让步的智能体和基于对手行为让步的智能体，可以通过设置`beta`和`kind_rate`参数来调整智能体的让步幅度。

##### 使用方法
- 该环境为双边协商的多议题环境，使用时参照`test.py`

- 首先创建一个协商的对象，该对象表示协商的环境:
`negotiation = Negotiation(max_round=30, issue_num=3, render=True)`
- 可以自己继承`Agent`类定义自己想要的智能体或者直接修改`Agent`类
```python
    agent1 = TimeAgentBoulware(max_round=30, name="time boulware agent")
    agent2 = BehaviorAgentAverage(max_round=30, name="behavior average agent")

```
- 将需要的两个智能体加入协商环境
```python
    negotiation.add(agent1)
    negotiation.add(agent4)
```
- 运行环境。这里的对立度表示了两者偏好向量之间的冲突程度，对立度越高则冲突程度越大，两者之间越难达成协商。
```python
    negotiation.reset(opposition="low")
    negotiation.run()
```

- 强化学习算法为SAC，训练方法为：
`python run_sac.py --gpu_no=0 --mode=train --oppo_type=mix`

使用模型评估方法为：
- 对打behavior 对手
`python run_sac.py --gpu_no=0 --mode=test --oppo_type=behavior --offer_model=SAC`
- 对打time 对手
`python run_sac.py --gpu_no=0 --mode=test --oppo_type=time --offer_model=SAC`

- 可以根据`run_sac.py`中的参数进行相应的调整

