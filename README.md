# DOREA-UAI2023

#### Multi issue negotiation environment

##### user guide
- This environment is a multi-issue bilateral negotiation environment. Refer to the following `test.py` when using it.

- First create a negotiation object, which represents the negotiation environment:
`negotiation = Negotiation(max_round=30, issue_num=3, render=True)`
- You can inherit the `Agent` class to define your own agent or directly modify the `Agent` class.
```python
    agent1 = randomagent(max_round=30, name="randomagent agent")
    agent2 = randomagent(max_round=30, name="randomagent agent")

```
- Add the two required agents to the negotiation environment
```python
    negotiation.add(agent1)
    negotiation.add(agent4)
```
-  Run the environment. The degree of opposition here indicates the degree of conflict between the two preference vectors. The higher the degree of opposition, the greater the degree of conflict, and the harder it is for them to reach an agreement.
```python
    negotiation.reset(opposition="low")
    negotiation.run()
```
