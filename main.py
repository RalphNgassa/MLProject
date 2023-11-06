import datetime
import hashlib
import matplotlib.pyplot as plt
import numpy
import pandas as pd
import sys
import seaborn as sns
from sklearn import linear_model
from sklearn import metrics
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.model_selection import cross_val_score, KFold, LeaveOneOut, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, precision_score, classification_report
import operator


numpy.set_printoptions(threshold=sys.maxsize)
DATA_PATH = "trots_2013-2022.parquet"
scale = StandardScaler()
horseCount = {} 

def plotValues(df):
    # plt.figure(figsize=(8, 6))

    # Adding jitter to better visualize overlapping points
    sns.stripplot(x='FrontHindShoes', y='BeatenMargin', data=df, jitter=True, palette='viridis')

    # Optionally, you can use a violin plot for better distribution visualization
    # sns.violinplot(x='Category', y='Numeric_Output', data=df, inner='quartile', palette='viridis')

    plt.title('Scatter Plot of Categorical Input vs. Numeric Output')
    plt.xlabel('FrontHindShoes')
    plt.ylabel('BeatenMargin')

    plt.show()

def mapFinish(x):
    val = x.strip()
    if x.strip().isnumeric(): return int(x)
    elif val=='BS': return -1
    elif val=='UN': return -2
    elif val=='DQ': return -3
    elif val=='PU': return -4
    elif val=='NP': return -5
    elif val=='FL': return -6
    elif val=='UR': return -7
    elif val=='WC': return -8

def mapFinishBinary(x):
    if x.strip() == '1' :return 1
    else: return 0

def strToDate(str):
    return str.date()

def logit2prob(logr, X):
  log_odds = logr.coef_ * X + logr.intercept_
  odds = numpy.exp(log_odds)
  probability = odds / (1 + odds)
  return(probability)

def evaluateWithConfusionMatrix(actual, predicted):
    
    confusion_matrix = metrics.confusion_matrix(actual, predicted, labels=actual.unique())

    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = actual.unique())

    cm_display.plot()
    plt.show()

def scaleDf(orgDf, columnsToScale, otherAttr):
    scaled_data = scale.fit_transform(orgDf[columnsToScale])

    # # Create a new DataFrame with scaled values
    scaled_df = pd.DataFrame(scaled_data, columns=columnsToScale)
    # print(trainingSet)
    # # Combine the scaled numeric columns with the non-numeric columns
    df_scaled = pd.concat([scaled_df, orgDf[otherAttr].reset_index(drop=True)],
                           ignore_index=True, sort=False, axis=1)
    return df_scaled

def crossValidateModel(model, X, y, cv_technique):
    scores = cross_val_score(model, X, y, cv = cv_technique)
    print("Cross Validation Scores: ", scores)
    print("Average CV Score: ", scores.mean())
    print("Number of CV Scores used in Average: ", len(scores))

def countNumberOfHorsesPerRace(df):

    for index, row in df.iterrows():
        id = row['RaceID']
        horseCount[id] = horseCount.get(id, 0) + 1

def addHorseCount(raceId):
    return horseCount.get(raceId)

def main():
    
    FILTER_DATE = datetime.date(2021, 11, 1)
    k_folds = KFold(n_splits=5)

    k_best = SelectKBest(score_func=f_classif, k='all')

    outPutDf = pd.DataFrame(columns=["Winprob"])

    df = pd.read_parquet(DATA_PATH)

    df = pd.get_dummies(df, columns=["AgeRestriction", "CourseIndicator",  "GoingAbbrev", "HandicapType", "RaceGroup",
                                      "RacingSubType","SexRestriction",
                                      "StartType", "Surface"],
                         dtype=int, drop_first=True)
    countNumberOfHorsesPerRace(df)
    df["FinishPosition"] = df["FinishPosition"].apply(mapFinishBinary)
    df["NumberOfHorses"] = df["RaceID"].apply(addHorseCount)
    df["FrontHindShoes"] = df["FrontShoes"].astype(str) + df["HindShoes"].astype(str)
    # df["JockeyID"] = df["JockeyID"].apply(hashId)

    startTimeSeries = df["RaceStartTime"].apply(strToDate)
    trainingSet = df.loc[startTimeSeries < FILTER_DATE]
    # trainingSet, validateSet = train_test_split(df.loc[startTimeSeries < FILTER_DATE], test_size=0.2)
    testSet = df.loc[startTimeSeries >= FILTER_DATE]
    
    # plotValues(trainingSet.iloc[:, :500][["HorseID", "RaceOverallTime"]])
    # print(df.columns)

    columns_to_scale = ["Distance", 'HandicapDistance', 'WeightCarried']

    trainingAttr = ['Barrier', 'GoingAbbrev_H  ', 'GoingAbbrev_SO ', 'GoingAbbrev_U  ',
                    'GoingAbbrev_VF ', "FrontShoes", "HandicapType_Cwt", "HandicapType_Hcp", 
                     "HorseAge", "HindShoes",  "RaceGroup_G1", 
                    "RaceGroup_G2", "RaceGroup_G3",'SexRestriction_C&G', 'SexRestriction_F',
                    'SexRestriction_M',  "StartType_V", "StartingLine", 
                    "Surface_S", "Surface_T", "WetnessScale", 'AgeRestriction_2yo',
       'AgeRestriction_3-10yo', 'AgeRestriction_3-5yo', 'AgeRestriction_3yo',
       'AgeRestriction_3yo+', 'AgeRestriction_4&5yo', 'AgeRestriction_4-10yo',
       'AgeRestriction_4-6yo', 'AgeRestriction_4-7yo', 'AgeRestriction_4-8yo',
       'AgeRestriction_4-9yo', 'AgeRestriction_4yo', 'AgeRestriction_4yo+',
       'AgeRestriction_5&6yo', 'AgeRestriction_5-10yo', 'AgeRestriction_5-7yo',
       'AgeRestriction_5-8yo', 'AgeRestriction_5-9yo', 'AgeRestriction_5yo',
       'AgeRestriction_5yo+', 'AgeRestriction_6&7yo', 'AgeRestriction_6-10yo',
       'AgeRestriction_6-8yo', 'AgeRestriction_6-9yo', 'AgeRestriction_6yo',
       'AgeRestriction_6yo+', 'AgeRestriction_7&8yo', 'AgeRestriction_7-10yo',
       'AgeRestriction_7-9yo', 'AgeRestriction_7yo', 'AgeRestriction_7yo+',
       'AgeRestriction_8&9yo', 'AgeRestriction_8-10yo', 'AgeRestriction_8yo',
       'AgeRestriction_8yo+', 'AgeRestriction_9&10yo', 'AgeRestriction_Pour 9', 'NumberOfHorses', 'RaceID',
       'HorseID', 'TrainerID', 'JockeyID', 'TrackID', 'CourseIndicator_&', 'CourseIndicator_G', 'CourseIndicator_P',
       'RacePrizemoney']

    targetAttr = ["FinishPosition"]

    # X_train_best = k_best.fit_transform(trainingSet[trainingAttr], trainingSet[targetAttr])

    # selected_features_indices = k_best.get_support(indices=True)

    # # Get feature names
    # selected_feature_names = trainingSet[trainingAttr].columns[selected_features_indices]

    # for feature_name, score, p_value in zip(selected_feature_names, k_best.scores_, k_best.pvalues_):
    #     print(f"Feature: {feature_name}, Score: {score}, P-value: {p_value}")

    # sampleTestData = testSet[testSet["RaceID"]==1682989]
    

    # print(df[trainingAttr].corr(method ='pearson')['RaceOverallTime'].to_string())
    # print(df['NumberOfHorses'].unique())
    X_train = scaleDf(trainingSet, columns_to_scale, trainingAttr)
    y_train = trainingSet[targetAttr]
    
    # scaled_data = scale.fit_transform(trainingSet[columns_to_scale])

    # print(trainingSet)
    logr = linear_model.LogisticRegression(max_iter=4000)
    logr.fit(X_train, y_train.values.ravel())
    X_test = scaleDf(testSet, columns_to_scale, trainingAttr)
    y_true = testSet[targetAttr]
    y_pred = logr.predict(X_test.to_numpy())


    
    probaFromTrain = logr.predict_proba(X_train)
    probaFromTest = logr.predict_proba(X_test)
    combined = numpy.vstack((probaFromTrain, probaFromTest))
    
    outPutDf = pd.DataFrame(combined, columns=['NotWin', 'Win'])[['Win']]


    # print(testSet[['RaceID', 'FinishPosition']][:20])
    # print('--------------------------------')
    # print(y_true[:20])
    # print('--------------------------------')
    # print(y_pred[:20])
    # print(logr.coef_)
    # for index, row in X_train.iterrows():
        # print(row.to_numpy())
    # logit2prob(logr, X_train)

    # print(logr.score(X_train, y_train))

    # crossValidateModel(logr, X_train, y_train.values.ravel(), k_folds)
    # evaluateWithConfusionMatrix(y_true, y_pred)
    # print(classification_report(y_true, y_pred))


    # knn = KNeighborsClassifier(n_neighbors=5)
    # knn.fit(X_train, y_train.values.ravel())
    # y_pred = knn.predict(X_test)
    # print(y_true.to_numpy()[:80])
    # print("-------------------------------------------------------------")
    # print(y_pred[:80])
    # # precision ratio: tp / (tp + fp), aiming at minimize fp (predict: win, actual: lose)
    # print(precision_score(y_true, y_pred, average='weighted'))
    # print(classification_report(y_true, y_pred))
    # scores[13] = ps
    # scores_list.append(ps)
    
    # print(scores)
    # print(max(scores_list))

    # y_pred = logr.predict(testSet[trainingAttr].to_numpy())

    # regressor = RandomForestRegressor(n_estimators=100, random_state=42)

    # # Train the regressor on the training data
    # regressor.fit(trainingSet[trainingAttr], trainingSet[targetAttr])

    # # Make predictions on the test set
    # y_pred = regressor.predict(testSet[trainingAttr])

    # # Evaluate the model using mean squared error
    # mse = mean_squared_error(testSet[targetAttr].to_numpy(), y_pred)
    # print("Mean Squared Error:", mse)
    outPutDf.to_parquet('output.parquet')


main()
