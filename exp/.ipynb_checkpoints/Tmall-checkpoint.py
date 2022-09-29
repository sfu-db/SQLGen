import pandas as pd
import os
import sys


from sqlgen.sqlgen import QueryTemplate, SQLGen
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

sys.path.append("../")
sys.path.insert(1, "../exp")


def load_data():
    path = "../exp_data/Tmall/"
    train_data = pd.read_csv(os.path.join(path, "train_data.csv"))
    test_data = pd.read_csv(os.path.join(path, "test_data.csv"))
    user_log = pd.read_csv(os.path.join(path, "user_log.csv"))
    print(len(user_log))

    train_data = train_data.drop(train_data.columns[0], axis=1)
    train_labels = train_data["label"]
    train_data.drop(columns=["label"])

    test_data = test_data.drop(test_data.columns[0], axis=1)
    test_labels = test_data["label"]
    test_data.drop(columns=["label"])

    user_log = user_log.drop(user_log.columns[0], axis=1)
    # data = pd.concat([train_data, test_data])
    print(len(user_log))
    print(
        "Train true label:",
        len(train_data[train_data["label"] == 1]),
        "Train false label:",
        len(train_data[train_data["label"] == 0]),
    )
    print(
        "Test true label:",
        len(test_data[test_data["label"] == 1]),
        "Test false label:",
        len(test_data[test_data["label"] == 0]),
    )
    return train_data, train_labels, test_data, test_labels, user_log


def evaluate_test_data(
    train_data, train_labels, test_data, test_labels, optimal_query_list
):
    for query in optimal_query_list:
        new_feature = sqlgen_task.generate_new_feature(query["param"])
        train_data.merge(new_feature, how="left")
        test_data.merge(new_feature, how="left")
    clf = XGBClassifier(random_state=0)
    clf.fit(train_data, train_labels)
    predictions = clf.predict(test_data)
    score = roc_auc_score(test_labels, predictions)

    return score


if __name__ == "__main__":
    train_data, train_labels, test_data, test_labels, user_log = load_data()
    seed_list = [0, 89, 143, 572, 1024]
    test_score_list = []

    print(user_log["gender"].unique())
    print(user_log["age_range"].unique())
    print(user_log["action_type"].unique())
    print(user_log["time_stamp"].unique())
    fkeys = ["user_id"]
    agg_funcs = ["SUM", "MIN", "MAX", "COUNT", "AVG"]
    agg_attrs = ["merchant_id", "item_id", "brand_id", "cat_id"]
    predicate_attrs = ["gender", "age_range", "action_type", "time_stamp"]
    groupby_keys = fkeys
    predicate_attr_types = {
        "gender": {"type": "categorical", "choices": user_log["gender"].unique()},
        "age_range": {"type": "categorical", "choices": user_log["age_range"].unique()},
        "action_type": {
            "type": "categorical",
            "choices": user_log["action_type"].unique(),
        },
        "time_stamp": {"type": "datetime", "choices": user_log["time_stamp"].unique()},
    }
    query_template = QueryTemplate(
        fkeys=fkeys,
        agg_funcs=agg_funcs,
        agg_attrs=agg_attrs,
        predicate_attrs=predicate_attrs,
        groupby_keys=groupby_keys,
        predicate_attrs_type=predicate_attr_types,
    )

    sqlgen_task = SQLGen()
    sqlgen_task.build_task(
        query_template=query_template,
        base_table=train_data,
        labels=train_labels,
        relevant_table=user_log,
    )
    for seed in seed_list[:1]:
        optimal_query_list = sqlgen_task.optimize()
        test_score = evaluate_test_data(
            train_data, train_labels, test_data, test_labels, optimal_query_list
        )
        test_score_list.append(test_score)

    print(f"The average test score is: {sum(test_score_list) / len(test_score_list)}")
