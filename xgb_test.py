import xgboost as xgb
import numpy as np
import json

X = np.random.randn(100, 5)
y = np.random.randn(100)        # NOTE 1D

m = xgb.XGBRegressor()
m.fit(X, y)

cfg = json.loads(m.get_booster().save_config())

print(cfg["learner"]["learner_model_param"])

print(m.get_booster().save_config())