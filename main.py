
import pandas as pd
import numpy as np
from sklearn import tree


# 欠損データを返すテーブルを表示
def deficit_table(df):
    null_sum = df.isnull().sum()
    percent = 100 * null_sum / len(df)
    deficit_table = pd.concat([null_sum, percent], axis=1)
    deficit_table_len_columns = deficit_table.rename(
        columns={0: '欠損数', 1: '%'}
    )
    return deficit_table_len_columns


# 前処理
train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')

train['Age'] = train['Age'].fillna(train['Age'].median())
train['Embarked'] = train['Embarked'].fillna('S')
train.loc[train['Sex'] == 'male', 'Sex'] = 0
train.loc[train['Sex'] == 'female', 'Sex'] = 1
train.loc[train['Embarked'] == 'S', 'Embarked'] = 0
train.loc[train['Embarked'] == 'C', 'Embarked'] = 1
train.loc[train['Embarked'] == 'Q', 'Embarked'] = 2

test['Age'] = test['Age'].fillna(test['Age'].median())
test.loc[test['Sex'] == 'male', 'Sex'] = 0
test.loc[test['Sex'] == 'female', 'Sex'] = 1
test.loc[test['Embarked'] == 'S', 'Embarked'] = 0
test.loc[test['Embarked'] == 'C', 'Embarked'] = 1
test.loc[test['Embarked'] == 'Q', 'Embarked'] = 2
test['Fare'] = test['Fare'].fillna(test["Fare"].median())


# 決定木1
target = train['Survived'].values
features_one_list = ['Pclass', 'Sex', 'Age', 'Fare']
features_one = train[features_one_list].values
my_tree_one = tree.DecisionTreeClassifier().fit(features_one, target)

test_features_one = test[features_one_list].values
my_prediction_one = my_tree_one.predict(test_features_one)

PassengerId = np.array(test['PassengerId']).astype(int)
my_solution_one = pd.DataFrame(my_prediction_one, PassengerId, columns=['Survived'])
my_solution_one.to_csv('my_tree_one.csv', index_label=['PassengerId'])

# 決定木2
target = train['Survived'].values
features_two_list = ['Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Parch', 'Embarked']
max_depth = 10
min_samples_split = 5

features_two = train[features_two_list].values
my_tree_two = tree.DecisionTreeClassifier(max_depth=max_depth,
                                          min_samples_split=min_samples_split,
                                          random_state=1)
my_tree_two = my_tree_two.fit(features_two, target)

test_features_two = test[features_two_list].values
my_prediction_two = my_tree_two.predict(test_features_two)

PassengerId = np.array(test['PassengerId']).astype(int)
my_solution_two = pd.DataFrame(my_prediction_two, PassengerId, columns=['Survived'])
my_solution_two.to_csv('my_tree_two.csv', index_label=['PassengerId'])
