from runner import Runner

from utils import load_json

config=load_json("./config/config.json")

runner=Runner(config)
runner.train()
runner.output()

