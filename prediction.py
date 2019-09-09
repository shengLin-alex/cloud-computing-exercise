from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics


class Prediction(object):
    def __init__(self, x, y, prediction_strategy=None):
        self.x = x
        self.y = y
        self.prediction_strategy = prediction_strategy

        # 70% training 30% test
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(x, y, test_size=0.3, random_state=0)

    def get_accuracy(self):
        if self.prediction_strategy:
            accuracy = self.prediction_strategy(self)
        else:
            accuracy = 0

        return accuracy


def dtree_prediction(prediction: Prediction):
    # build tree
    clf = DecisionTreeClassifier()
    clf.fit(prediction.X_train, prediction.Y_train)
    y_test_predicted = clf.predict(prediction.X_test)
    accuracy_test = metrics.accuracy_score(prediction.Y_test, y_test_predicted)

    return accuracy_test


def random_forest_prediction(prediction: Prediction):
    sc = StandardScaler()
    x_train = sc.fit_transform(prediction.X_train)
    x_test = sc.transform(prediction.X_test)

    rf = RandomForestClassifier()
    rf.fit(x_train, prediction.Y_train.values.ravel())
    y_test_predicted = rf.predict(x_test)
    accuracy_test = metrics.accuracy_score(prediction.Y_test, y_test_predicted)

    return accuracy_test


def gaussian_nb_prediction(prediction: Prediction):
    gnb = GaussianNB()

    # ravel 展開為 1d array
    gnb.fit(prediction.X_train, prediction.Y_train.values.ravel())
    y_test_predicted = gnb.predict(prediction.X_test)
    accuracy_test = metrics.accuracy_score(prediction.Y_test, y_test_predicted)

    return accuracy_test
