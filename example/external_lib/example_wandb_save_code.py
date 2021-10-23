import wandb
from inspect import getfile
from rlrl.agents import SacAgent
import os

if __name__ == "__main__":
    run = wandb.init(project="rlrl_example", name="log_code")
    # art = run.log_artifact(__file__, type="code")
    print(os.path.join(wandb.run.dir, "code/"))
    print(run.save(os.path.join(wandb.run.dir, "code/").join(getfile(SacAgent))))
    # run.save(getfile(SacAgent))
    # run.log_code()
    # print(getfile(SacAgent))
