import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
from selection import *
from classifier import *
warnings.filterwarnings('ignore')
from sklearn.metrics import mean_squared_error

data = pd.read_csv('data/mart/train_data.csv')
data.Item_Weight = data.Item_Weight.fillna(data.Item_Weight.mean())
data.Outlet_Size = data.Outlet_Size.fillna('Medium')

#Item type combine:
data['Item_Identifier'].value_counts()
data['Item_Type_Combined'] = data['Item_Identifier'].apply(lambda x: x[0:2])
data['Item_Type_Combined'] = data['Item_Type_Combined'].map({'FD':'Food',
                                                             'NC':'Non-Consumable',
                                                             'DR':'Drinks'})
data['Item_Type_Combined'].value_counts()

le = LabelEncoder()
#New variable for outlet
data['Outlet'] = le.fit_transform(data['Outlet_Identifier'])
var_mod = ['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Item_Type_Combined','Outlet_Type','Outlet']
le = LabelEncoder()
for i in var_mod:
    data[i] = le.fit_transform(data[i])

#One Hot Coding:
data = pd.get_dummies(data, columns=['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Outlet_Type','Item_Type_Combined','Outlet'])


#Drop the columns which have been converted to different types:
data.drop(['Item_Type','Outlet_Establishment_Year'],axis=1,inplace=True)

# Divide into test and train:
train_ = data.loc[:5999]
test_ = data.loc[5999:]

X_train = train_.drop(['Item_Outlet_Sales', 'Outlet_Identifier','Item_Identifier'], axis=1)
y_train = train_.Item_Outlet_Sales

X_test = test_.drop(['Item_Outlet_Sales', 'Outlet_Identifier','Item_Identifier'], axis=1)
y_test = test_.Item_Outlet_Sales

# X = data.drop(['Item_Outlet_Sales', 'Outlet_Identifier','Item_Identifier'], axis=1)
# y = data.Item_Outlet_Sales

# std = StandardScaler()
# X = std.fit_transform(X)
# X_train, X_test, y_train, y_test = train_test_split(X, np.array(y))


##################################################
# Linear Regression Model
# Fitting Multiple Linear Regression to the training set
from sklearn.linear_model import  LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pre = regressor.predict(X_test)
print(regressor.score(X_test, y_test))
print('MSE: {:.3f}.'.format(mean_squared_error(y_test,y_pre)))

##################################################
# Decision Tree Model
# Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(max_depth=15,min_samples_leaf=300)
regressor.fit(X_train, y_train)
y_pre = regressor.predict(X_test)
print(regressor.score(X_test, y_test))
print('MSE: {:.3f}.'.format(mean_squared_error(y_test,y_pre)))

##################################################
# Random Forest Model]
# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=100,max_depth=6, min_samples_leaf=50,n_jobs=4)
regressor.fit(X_train, y_train)
y_pre = regressor.predict(X_test)
print(regressor.score(X_test, y_test))
print('MSE: {:.3f}.'.format(mean_squared_error(y_test,y_pre)))

##################################################
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
# 加载训练集和测试集
train_dataset = TensorDataset(
    torch.from_numpy(X_train.values).type(torch.float),
    torch.from_numpy(y_train.values).type(torch.float))
train_dataloader = DataLoader(dataset=train_dataset,
                              batch_size=32,
                              shuffle=True,
                              num_workers=2)

test_dataset = TensorDataset(
    torch.from_numpy(X_test.values).type(torch.float),
    torch.from_numpy(y_test.values).type(torch.float))
test_dataloader = DataLoader(test_dataset,
                             batch_size=2524,
                             shuffle=True,
                             num_workers=2)


class MLPmodel(nn.Module):
    def __init__(self):
        super(MLPmodel, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(31, 16),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(16, 10),
            nn.ReLU(),
        )
        self.regression = nn.Linear(10, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        output = self.regression(x)
        return output

# 参数设置
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
learning_rate = 0.001
epochs = 50

def train(train_dataloader):
    model = MLPmodel()
    model.train()
    model.to(device)

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 损失函数
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        loss_epoch = 0
        for batch_idx, (batch_x, batch_y) in enumerate(train_dataloader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            output = model(batch_x).flatten()
            loss = criterion(output, batch_y)
            loss_epoch += loss*batch_x.shape[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Train Epoch: {}\t MSE:{:.3f}'.format(
            epoch + 1, loss_epoch / X_train.shape[0]))
        # 保存模型
        # torch.save(model.state_dict(), 'model/model{}.pth'.format(epoch + 1))
    return model


def test(test_dataloader):
    model = MLPmodel()
    model.load_state_dict(torch.load('model/model30.pth'))
    model.eval()
    model.to(device)
    criterion = nn.MSELoss()

    batch_idx, (batch_x, batch_y) = list(enumerate(test_dataloader))[0]
    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
    output = model(batch_x).flatten()
    MSE = criterion(output, batch_y)

    # 计算准确率
    print('MSE: {:.3f}.'.format(MSE))


if __name__ == '__main__':
    models = []
    features = []
    types = []
    label_map = dict()
    algorithm = ['mv', 'rs', 'ta', 'efmv']
    for i in range(30):
        m = train(train_dataloader)
        models.append(Regressor(regressor_type='mlp', rgs=m))
        features.append(list(range(data.shape[1]-1)))
        label_map[i] = i
        types.append('mlp')
    full_ensemble = Ensemble(30, types, features, label_map, task=Regression, base_learners=models)
    noTree = 4
    ddl = 10
    print('ddl', ddl)
    coef = 1
    trees = []
    mean_MSE = np.zeros((len(algorithm)))
    mean_MAE = np.zeros((len(algorithm)))
    term=1
    for t in range(noTree):
        tree1 = EnsembleTree(full_ensemble, train_, rank=t + 1, coefficient=coef, one_result=False, )
        trees.append(tree1)
    forest = EnsembleForest(trees, full_ensemble.label_map)
    for a in range(len(algorithm)):
        alg = algorithm[a]
        start_time = time.time()
        MSE = 0
        if alg == 'mv':
            MSE, MAE = full_ensemble.MSE(test_)
        elif alg == 'rs':
            # confidence_matrix = full_ensemble.randomSelectMajorityVote(test, ddl)
            MSE, MAE = full_ensemble.rsMSE(train_, _, ddl)
        elif alg == 'ta':
            MSE, MAE = full_ensemble.topKMSE(train_, test_, ddl)
        elif alg == 'efmv':
             MSE, MAE = forest.MSE(test_, ddl)
        print(alg, 'MSE=', MSE)
        mean_MSE[a] += MSE / term
        mean_MAE[a] += MAE / term

    print('algorithm   MSE')
    for i in range(len(algorithm)):
        print(algorithm[i], mean_MSE[i], mean_MAE[i])
