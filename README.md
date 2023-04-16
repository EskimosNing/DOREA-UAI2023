# An Effective Negotiating Agent Framework based on Deep Offline Reinforcement Learning

## Installation
This codebase is based on rlkit (https://github.com/vitchyr/rlkit).<br/>
To set up the environment and run an example experiment:

1. Create conda virtual environment. 
```
$ cd rlkit
$ conda env create -f environment/linux-gpu-env.yml
```

2. Add this repo directory to your `PYTHONPATH` environment variable or simply run:
```
$ conda activate rlkit
(rlkit) $ pip install -e .
```

3. Update pip and torch
```
(rlkit) $ pip install -U pip
(rlkit) $ pip install torch==1.4.0
```

## Run a demo experiment
offline train a negotiation agent
``` 
(rlkit) $ cd multi_issue_negotiation 
(rlkit) $ python algorithm/offline_train.py --buffer_filename <buffer_filename> --save_model_dir <save_model_dir> --domain <domain>
```

Fine-tune an offline agent
```
(rlkit) $ cd multi_issue_negotiation
(rlkit) $ python algorithm/finetune.py --buffer_filename <buffer_filename> --save_model_dir <save_model_dir> --domain <domain>
```


## Multi issue negotiation environment

### user guide
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
