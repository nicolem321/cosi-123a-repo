import os
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV, Lasso, LassoLarsCV, OrthogonalMatchingPursuit, ElasticNetCV, ARDRegression
from sklearn.preprocessing import StandardScaler

def main(dir_path):
    # load models to be tested for each dataset, see finding_best_model_per_dataset.py for how we got these models
    model_per_dataset = {'TrainX_3b.csv': ARDRegression(), 'Train_1a.csv': RidgeCV(), 'TrainX_2b.csv': ElasticNetCV(), 'TrainX_5b.csv': ARDRegression(),
                         'TrainX_4b.csv': OrthogonalMatchingPursuit(), 'Train_5a.csv': Lasso(), 'Train_4a.csv': OrthogonalMatchingPursuit(),
                         'Train_3a.csv': RidgeCV(), 'TrainX_1b.csv': LassoLarsCV(), 'Train_2a.csv': LassoLarsCV()}

    # looping through each dataset
    for subdir in os.listdir(dir_path):
        subdir_path = os.path.join(dir_path, subdir)
        if os.path.isdir(subdir_path):
            files = [filename for filename in os.listdir(subdir_path)]
            dataset_name = (((files[0].split('_'))[1]).split('.'))[0]

            # train model
            for filename in files:
                if 'Train' in filename:
                    f = os.path.join(subdir_path, filename)

                    # read in the training data, splitting into X and y
                    train_data = pd.read_csv(f)
                    train_X = StandardScaler().fit_transform(train_data.iloc[:,:-1])
                    train_y = train_data.iloc[:,-1]

                    # train model
                    model = model_per_dataset[filename]
                    model.fit(train_X, train_y)
            
            # predict model
            for filename in files:
                if 'Test' in filename:
                    f = os.path.join(subdir_path, filename)

                    # read in the training data, splitting into X and y
                    test_data = pd.read_csv(f)
                    test_X = StandardScaler().fit_transform(test_data)

                    # pred model
                    test_y = model.predict(test_X)
            
            # save model
            results = pd.read_csv('hw5_result.csv')
            results[dataset_name] = test_y
            results.to_csv('hw5_result.csv', index=False)


if __name__ == "__main__":
    dir_path = 'HW5data/a'
    main(dir_path)

    dir_path = 'HW5data/b'
    main(dir_path)
