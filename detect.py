"""
File: detect.py
Author: Benson Kimani - https://www.linkedin.com/in/benson-kimani-infotech/
Date: 2024-06-15

Description: Script to classify a link/URL as phish or benign based on a saved MLP ml model state.
"""


# from torchvision import models
import torch
# Import model class
import Dnn

import pandas as pd
import numpy as np

# # Phish Sample
# link_features = {'numSubDomains': 0.25, 'hasHttps': 1, 'numImages': 0.25, 'numLinks': 0.0, 'specialChars': 0,
#                  'sbr': 0.25, 'numberOfIncludedElements': 0, 'urlAge': 0.25}

# Benign Sample
link_features = {'numSubDomains': 1.0, 'hasHttps': 1, 'numImages': 0.75, 'numLinks': 0.2, 'specialChars': 0.25,
                 'sbr': 0.25, 'numberOfIncludedElements': 1.0, 'urlAge': 0.75}


class Detect(object):
    def __init__(self, features):
        self.features = features

    def get_detection(self):

        # Configs
        path = 'state_dict_model.pth'
        classification_id = None

        # Normalize Data
        link_data = pd.DataFrame(self.features, index = [0])
        link_data.replace(True, 1, inplace = True)
        link_data.replace(False, 0, inplace = True)

        # Load Model
        model = Dnn.Dnn()
        model.load_state_dict(torch.load(path))
        # Since we are using our model only for inference, switch to `eval` mode:
        model.eval()

        # print ("Model state_dict:")
        # for param_tensor in model.state_dict():
        #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())

        # print (features)

        with torch.no_grad():
            link_data_values = link_data.values.astype(np.float32)
            # Apply MinMax Scalar ( Why Against what? )
            torch_tensor = torch.from_numpy(link_data_values)
            outputs = model.forward(torch_tensor)
            y_pred_tag = torch.round(outputs)
            predicted = (y_pred_tag.item())
            # print (outputs)

            if predicted >= 0.5:
                classification_id = "LGT"
                print("Link Is Benign")
                return classification_id
                
            else:
                classification_id = "SSP"
                print("Link is Suspicious")
                return classification_id


p1 = Detect(link_features)
p1.get_detection()
