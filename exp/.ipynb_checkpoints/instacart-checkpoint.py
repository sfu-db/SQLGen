import pandas as pd
import os
import sys
import warnings
import copy
import numpy as np
import featuretools as ft


from sqlgen.sqlgen import QueryTemplate, SQLGen
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

sys.path.append("../")
sys.path.insert(1, "../exp")


def load_data():
    path = "../exp_data/Instacart/"
    order_products = pd.read_csv(os.path.join(path, "order_products__prior.csv"))
    orders = pd.read_csv(os.path.join(path, "orders.csv"))
    departments = pd.read_csv(os.path.join(path, "departments.csv"))
    products = pd.read_csv(os.path.join(path, "products.csv"))
    # user = pd.DataFrame()
    # user['user_id'] = orders["user_id"]
    # user = user.drop_duplicates(keep='first', inplace=False)

    logs = (
        order_products.merge(
            orders, how="left", left_on="order_id", right_on="order_id"
        )
        .merge(products, how="left", left_on="product_id", right_on="product_id")
        .merge(
            departments, how="left", left_on="department_id", right_on="department_id"
        )
    )

    new_product_name = logs["product_name"].str.contains(r"Banana").astype(int)
    logs = logs.drop(columns=["product_name"])
    logs["product_name"] = new_product_name

    train_logs = logs[logs["eval_set"] == "train"]
    users_buy_banana = train_logs[train_logs["product_name"] == 1]["user_id"].unique()
    print("# of users buy banana", len(users_buy_banana))
    users_donot_buy_banana = train_logs[train_logs["product_name"] == 0][
        "user_id"
    ].unique()
    print("# of users do not buy banana", len(users_donot_buy_banana))

    user_train_logs = train_logs[["user_id", "product_name"]].drop_duplicates()
    X_train, X_test, y_train, y_test = train_test_split(
        user_train_logs.drop(["product_name"], axis=1),
        user_train_logs["product_name"],
        test_size=0.2,
        random_state=42,
    )
    print(X_train.shape, X_test.shape)
    train_data = X_train
    train_labels = y_train
    test_data = X_test
    test_labels = y_test

    train_logs["index"] = (
        train_logs["user_id"].astype(str)
        + "_"
        + train_logs["order_id"].astype(str)
        + "_"
        + train_logs["product_id"].astype(str)
    )
    train_logs = train_logs[
        [
            "index",
            "user_id",
            "order_id",
            "product_id",
            "department_id",
            "order_dow",
            "days_since_prior_order",
            "product_name",
        ]
    ]

    train_logs_copy = copy.deepcopy(train_logs)
    train_logs_copy = train_logs_copy.rename({"index": "index_copy"}, axis=1)
    train_logs_copy["unique_id"] = range(0, len(train_logs_copy))

    from woodwork.logical_types import Categorical, Integer, Datetime

    log_train_vtypes = {
        "index": Categorical,
        "user_id": Categorical,
        "order_id": Categorical,
        "product_id": Categorical,
        "order_dow": Categorical,
        "days_since_prior_order": Categorical,
        "department_id": Categorical,
        "product_name": Categorical,
    }

    es = ft.EntitySet("instacart")
    es.add_dataframe(
        dataframe_name="train_logs",
        dataframe=train_logs,
        index="index",
        logical_types=log_train_vtypes,
    )

    es.normalize_dataframe(
        base_dataframe_name="train_logs", new_dataframe_name="users", index="user_id"
    )

    feature_matrix, features = ft.dfs(
        target_dataframe_name="users",
        agg_primitives=["sum", "min", "max", "count", "mean"],
        trans_primitives=[],
        ignore_columns={"train_logs": ["order_id", "product_id", "department_id"]},
        entityset=es,
        verbose=True,
    )

    train_data = train_data.merge(
        feature_matrix, how="left", left_on="user_id", right_on="user_id"
    )
    test_data = test_data.merge(
        feature_matrix, how="left", left_on="user_id", right_on="user_id"
    )
    print(train_logs["days_since_prior_order"])
    print(train_labels.shape)

    return train_data, train_labels, test_data, test_labels, train_logs


def evaluate_test_data(
    train_data, train_labels, test_data, test_labels, optimal_query_list
):
    for query in optimal_query_list:
        arg_list = []
        for key in query["param"]:
            arg_list.append(query["param"][key])
        new_feature, join_keys = sqlgen_task.generate_new_feature(arg_list=arg_list)
        train_data = train_data.merge(
            new_feature, how="left", left_on=join_keys, right_on=join_keys
        )
        test_data = test_data.merge(
            new_feature, how="left", left_on=join_keys, right_on=join_keys
        )
    train_data = train_data.fillna(0)
    test_data = test_data.fillna(0)
    clf = RandomForestClassifier(random_state=0)
    clf.fit(train_data, train_labels)
    predictions = clf.predict(test_data)
    score = accuracy_score(test_labels, predictions)

    return score


if __name__ == "__main__":
    train_data, train_labels, test_data, test_labels, user_log = load_data()
    seed_list = [0, 89, 143, 572, 1024]
    test_score_list = []

    fkeys = ["user_id"]
    agg_funcs = ["SUM", "MIN", "MAX", "COUNT", "AVG"]
    agg_attrs = ["order_dow", "product_name"]
    predicate_attrs = [
        "product_name",
        "department_id",
        "order_dow",
        "days_since_prior_order",
    ]

    groupby_keys = fkeys
    predicate_attr_types = {
        "product_name": {
            "type": "categorical",
            "choices": [str(x) for x in user_log["product_name"].unique()] + ["None"],
        },
        "department_id": {
            "type": "categorical",
            "choices": [str(x) for x in user_log["department_id"].unique()] + ["None"],
        },
        "order_dow": {
            "type": "categorical",
            "choices": [str(x) for x in user_log["order_dow"].unique()] + ["None"],
        },
        "days_since_prior_order": {
            "type": "datetime",
            "choices": [str(x) for x in user_log["days_since_prior_order"].unique()]
            + ["None"],
        },
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
        optimal_query_list = sqlgen_task.optimize(
            ml_model="rf",
            metric="accuracy",
            base_tpe_budget=50,
            turn_on_mi=False,
            turn_on_mapping_func=False,
        )
        test_score = evaluate_test_data(
            train_data, train_labels, test_data, test_labels, optimal_query_list
        )
        test_score_list.append(test_score)

    print(f"The average test score is: {sum(test_score_list) / len(test_score_list)}")
