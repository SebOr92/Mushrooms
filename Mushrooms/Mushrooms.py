import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
matplotlib.style.use('ggplot')
from tpot import TPOTClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.metrics import confusion_matrix,accuracy_score

def load_and_check_data(path):

    data = pd.read_csv(path,
                       sep=',')
    print(data.shape)
    print(data.head())
    print(data.dtypes)
    print(data.columns)
    print(data.isnull().sum())
    print("Successfully loaded data from CSV")
    return data

def exploratory_data_analysis(data, plot_with_target = False, save = False):

    for col in data.columns:
        sns.countplot(x = col, data = data)
        plt.title("Distribution of " + str(col))
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.xlabel(None)
        if save:
            plt.savefig(fname=str(col)+"_count.png")
        plt.show()

def prepocess_data(data, test_size, seed, save = False):
    X = data[[x for x in list(data.columns) if x not in ["class"]]]
    y = data["class"]

    enc = OneHotEncoder(handle_unknown = 'error', drop='if_binary')
    le = LabelEncoder()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)

    X_train = enc.fit(X_train).transform(X_train).toarray()
    print("Shape of training data: " + str(X_train.shape))
    X_test = enc.transform(X_test).toarray()
    print("Shape of test data: " + str(X_test.shape))

    y_train = le.fit(y_train).transform(y_train)
    print("Shape of training target: " + str(y_train.shape))
    y_test = le.transform(y_test)
    print("Shape of test target: " + str(y_test.shape))

    return X_train, X_test, y_train, y_test

def TPOT(generations, population_size, cv, seed, verbosity):

    pipeline_optimizer = TPOTClassifier(generations=generations,
                                        population_size=population_size,
                                        cv=cv,
                                        random_state=seed, 
                                        verbosity=verbosity)

    pipeline_optimizer.fit(X_train, y_train)
    y_true, y_pred= y_test, model.predict(X_test)
    
    acc = accuracy_score(y_true, y_pred)
    print("Accuracy on test set: " + str(bal_acc))

    cm = confusion_matrix(y_true, y_pred)
    ax = sns.heatmap(cm,
                 annot=True,
                 fmt="d",
                 cbar=False,
                 cmap="Blues",
                 xticklabels=['Edible', 'Poisonous'],
                 yticklabels=['Edible', 'Poisonous'])
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix test set')
    if save:
        plt.savefig(fname="ConfusionMatrix.png")
    plt.show()

    pipeline_optimizer.export('tpot_exported_pipeline.py')

random_seed = 191
data = load_and_check_data('mushrooms.csv')
#exploratory_data_analysis(data, True, True)
X_train, X_test, y_train, y_test = prepocess_data(data, 0.25, random_seed)
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=random_seed)
TPOT(3, 10, cv, random_seed, 3, False)