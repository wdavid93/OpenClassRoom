import pickle as pkl

with (open('xgboost.pickle', 'rb')) as xfile:
   clf = pkl.load(xfile)
#   clf._booster.save_model('xgb.model')