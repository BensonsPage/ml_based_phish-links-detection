"""
File: dnn_model.py
Author: Benson Kimani - https://www.linkedin.com/in/benson-kimani-infotech/
Date: 2024-06-15

Description: Script to analyze phishing and being links data and used it to train a machine learning model.
"""

# Import pandas and numpy for data manipulation

import pandas as pd
import numpy as np

# Import matplot lib and seaborn for data visualization and startistcial representation

import matplotlib.pyplot as plt
import seaborn as sns

# Import pytorch ML framework
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

# Import sklearn for ML model tuning and fitting

from sklearn import metrics
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import itertools
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

from sklearn.datasets import load_digits
from sklearn.feature_selection import SelectKBest, chi2, f_classif

# Import Deep Neural Network Model implemented in another class Dnn.py
import Dnn


# Path to Training Data
df_data_path = "tt-data/benign_phish_links_features.csv"

# read Data
df_data = pd.read_csv(df_data_path)
# Encoding 'File' as label benign(1) & phish(0)
# df_data['File'].replace({'tt-data/phish_links': 0, 'tt-data/benign_links_tiny': 1, 'tt-data/benign_links': 1,
#                          'tt-data/phish_links_tiny': 0}, inplace=True)
df_data.replace({df_data['File']: {'tt-data/phish_links': 0, 'tt-data/benign_links_tiny': 1, 'tt-data/benign_links': 1,
                                   'tt-data/phish_links_tiny': 0}}, inplace=True)

# Rename colum "File" to Target (Dependent Variable)
df_data.rename(columns={'File': 'target'}, inplace=True)
# Normalize Data Types
df_data.replace({True: 1, False: 0}, inplace=True)

# get rid of missing values
df_data = df_data.dropna()

# Normalize data
df_data.astype({'numDigits': 'float64'}).dtypes

# Split data into intervals using percentiles
# Columns to be split into intervals
interval_contenders = ['urlAge', 'urlLength', 'numTitles', 'bodyLength', 'scriptLength', 'specialChars',
                       'numSubDomains', 'numberOfPeriods', 'numberOfIncludedElements', 'entropy', 'numImages',
                       'numLinks', 'sbr']


def intervals_by_percentiles(contenders):
    for ind in df_data.index:
        for contender in contenders:
            first_quantile = df_data[contender].quantile(q=0.25)
            second_quantile = df_data[contender].quantile(q=0.50)
            third_quantile = df_data[contender].quantile(q=0.75)
            if df_data[contender][ind] <= first_quantile:
                df_data.at[ind, contender] = 0.25
            elif df_data[contender][ind] <= second_quantile:
                df_data.at[ind, contender] = 0.50
            elif df_data[contender][ind] <= third_quantile:
                df_data.at[ind, contender] = 0.75
            else:
                df_data.at[ind, contender] = 1.00


intervals_by_percentiles(interval_contenders)

print(df_data.head(30))
print(df_data.dtypes)


def data_plot():
    # Pearson Correlation Heatmap
    plt.rcParams['figure.figsize'] == [36, 32]
    sns.set(font_scale=0.5)
    sns.heatmap(df_data.corr(), annot=True, cmap="YlGnBu");
    # plt.show()

    # dropping columns with no correlation
    df_data.drop(columns = {'urlHasPortInString'}, inplace = True)

    # Setting the bar graph and pie chart parameters
    plt.rcParams['figure.figsize'] = [18, 8]
    fig, ax = plt.subplots(1, 2)

    # Bar graph
    bar = sns.countplot(x=df_data["target"], ax = ax[0], palette = ['mediumorchid', 'coral',])
    bar.set(xlabel='Link Type', ylabel = 'Count')
    bar.set_title("Distribution of Phish(0) and Benign(1) Link", bbox={'facecolor':'0.8', 'pad':5})

    # Benign Vs Phish links pie-chart
    types = df_data['target'].value_counts()
    labels = list(types.index)
    aggregate = list(types.values)
    percentage = [(x*100)/sum(aggregate) for x in aggregate]
    print("The percentages of Benign and Phish Links are : ", percentage)

    # Plotting the Pie-chart to see the percentage distribution of the Links

    plt.rcParams.update({'font.size': 12})
    explode = (0, 0.1)
    ax[1].pie(aggregate, labels=labels, autopct='%1.2f%%', shadow=True, explode = explode, colors = ['coral', 'mediumorchid'])
    plt.title("Pie Chart for Phish(0) and Benign(1) Link", bbox={'facecolor': '0.8', 'pad': 5})
    plt.legend(labels, loc='best')
    plt.tight_layout()
    plt.show()

    # Violin Plot showing the relation HTTP, HTTPS protocol relations.
    plt.rcParams['figure.figsize'] = [18, 8]
    fig, ax = plt.subplots(1, 2)

    PAG = sns.violinplot(x=df_data.target, y=df_data.hasHttp, data=df_data, ax=ax[0])
    PAG.set(title='Violin Plot for hasHttp Benign(1) and Phish(0)', xlabel='Link Type')
    PAG_ = sns.violinplot(x=df_data.target, y=df_data.hasHttps, data=df_data, ax=ax[1])
    PAG_.set(title='Violin Plot for hasHttps Benign(1) and Phish(0)', xlabel='Link Type')
    plt.show()


# data_plot()


def train_test_dnn_model():
    # Drop non-significant features
    df_data.drop(
        columns={'scriptLength', 'urlLength', 'urlIntendedLifeSpan', 'bodyLength', 'numTitles', 'hasHttp', 'entropy',
                 'numberOfPeriods', 'urlLifeRemaining', 'daysSinceExpiration', 'has_ip', 'numParameters', 'isEncoded',
                 'urlIsLive', 'numEncodedChar', 'num_%20', 'sscr', 'bscr', 'numberOfDoubleDocuments', 'numDigits',
                 'numberOfIframes', 'numFragments', 'num_@', 'numberOfWhitespace'},
        inplace=True)
    # All Who's Data/ Third Party Data Removed as they are not always available for all tld during real time inference

    # Train & Test Set
    X= df_data.iloc[:, :-1]
    y= df_data['target']

    train_x,test_x,train_y,test_y = train_test_split(X, y, random_state=42, test_size=0.2)
    print("\n--Training data samples--")
    print(train_x.shape)

    print("\n--Test data samples--")
    print(test_x.shape)

    # Features Selection, Select best features
    selector = SelectKBest(f_classif, k=8)
    selector.fit(train_x, train_y)
    scores = -np.log10(selector.pvalues_)
    scores /= scores.max()

    # Plot uni-variate features selection.
    x_indices = np.arange(X.shape[-1])
    plt.figure(1)
    plt.clf()
    plt.bar(x_indices - 0.5, scores, width=1)
    plt.title("Feature uni-variate score")
    plt.xlabel("Feature number")
    plt.ylabel(r"Uni-variate score ($-Log(p_{value})$)")
    plt.show()

    x_train = train_x.values
    x_test = test_x.values

    print("Scaled values of Train set \n")
    print(x_train)
    print("\nScaled values of Test set \n")
    print(x_test)

    # Convert Train and Test sets into Tensors
    x_tensor = torch.from_numpy(x_train).float()
    y_tensor = torch.from_numpy(train_y.values.ravel()).float()
    xtest_tensor = torch.from_numpy(x_test).float()
    ytest_tensor = torch.from_numpy(test_y.values.ravel()).float()

    print("\nTrain set Tensors \n")
    print(x_tensor)
    print(y_tensor)
    print("\nTest set Tensors \n")
    print(xtest_tensor)
    print(ytest_tensor)

    # Batch Size
    bs = 48
    # Both x_train and y_train can be combined in a single TensorDataset, which will be easier to iterate over and slice
    y_tensor = y_tensor.unsqueeze(1)
    train_ds = TensorDataset(x_tensor, y_tensor)
    # Pytorch’s DataLoader is responsible for managing batches.
    # You can create a DataLoader from any Dataset. DataLoader makes it easier to iterate over batches
    train_dl = DataLoader(train_ds, batch_size=bs)

    # For the validation/test dataset
    ytest_tensor = ytest_tensor.unsqueeze(1)
    test_ds = TensorDataset(xtest_tensor, ytest_tensor)
    test_loader = DataLoader(test_ds, batch_size=24)

    model = Dnn.Dnn()
    # Loss Computation
    loss_func = nn.BCELoss()  # nn.BCEWithLogitsLoss()

    # Optimizer
    # MOMENTUM= 0.99
    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    epochs = 50

    model.train()
    train_loss = []
    for epoch in range(epochs):
        # Within each epoch run the subsets of data = batch sizes.
        for xb, yb in train_dl:
            y_pred = model(xb)            # Forward Propagation
            loss = loss_func(y_pred, yb)  # Loss Computation
            optimizer.zero_grad()         # Clearing all previous gradients, setting to zero
            loss.backward()               # Back Propagation
            optimizer.step()              # Updating the parameters
        print("Loss in iteration :"+str(epoch)+" is: "+str(loss.item()))
        train_loss.append(loss.item())
    print('Last iteration loss value: '+str(loss.item()))

    # Plot the Model
    plt.plot(train_loss)
    plt.show()

    y_pred_list = []
    model.eval()
    # No back propagate the gradients in test set we use torch.no_grad()
    with torch.no_grad():
        for xb_test,yb_test  in test_loader:
            y_test_pred = model(xb_test)
            y_pred_tag = torch.round(y_test_pred)
            y_pred_list.append(y_pred_tag.detach().numpy())

    # Takes arrays and makes them list of list for each batch
    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    # flattens the lists in sequence
    ytest_pred = list(itertools.chain.from_iterable(y_pred_list))
    y_true_test = test_y.values.ravel()

    conf_matrix = confusion_matrix(y_true_test, ytest_pred)
    print("Confusion Matrix of the Test Set")
    print("-----------")
    print(conf_matrix)
    print("Precision of the DNN :\t"+str(precision_score(y_true_test,ytest_pred)))
    print("Recall of the DNN    :\t"+str(recall_score(y_true_test,ytest_pred)))
    print("F1 Score of the Model :\t"+str(f1_score(y_true_test,ytest_pred)))

    # Saving Model
    # Specify a path
    path = "state_dict_model.pth"
    # Save
    torch.save(model.state_dict(), path)

    # Classification Report
    cls_report = metrics.classification_report(y_true_test,ytest_pred)

    print("")
    print(f"Accuracy : {metrics.accuracy_score(y_true_test,ytest_pred)*100 : .3f} %")
    print("")
    print("Classification Report : ")
    print(cls_report)
    # Setting the params for the plot
    plt.rcParams['figure.figsize'] = [10, 7]
    sns.set(font_scale=1.2)

    # Confusion Matrix
    cm = metrics.confusion_matrix(y_true_test, ytest_pred)

    # Plotting the Confusion Matrix
    ax = sns.heatmap(cm, annot=True, cmap='YlGnBu')
    ax.set(title="Confusion Matrix", xlabel='Predicted Labels', ylabel='True Labels')


# train_test_dnn_model()
