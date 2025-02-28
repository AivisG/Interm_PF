from lstm_model import LSTM_Model
from xgb_model import XGBoostModel
from gp_model import GaussianProcessModel
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logs


class Models:
    class LSTM_Model(LSTM_Model):  
        pass

    class XGB_Model(XGBoostModel):  
        pass

    class GP_Model(GaussianProcessModel):
        pass
