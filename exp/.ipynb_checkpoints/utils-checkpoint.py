import pandas as pd
import numpy as np
import itertools
import os
import random as rand
import math

# PostgreSQL
import psycopg2
from sqlalchemy import create_engine

# sklearn
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.utils import compute_class_weight

import xgboost as xgb

########## Instacart exp #########
def store_instacart(log_train):
    try:
        connection = psycopg2.connect(
            database="auto_fg",
            user="postgres",
            password="postgres",
            host="127.0.0.1",
            port="5432",
        )

        cursor = connection.cursor()
        # Print PostgreSQL Connection properties
        print(connection.get_dsn_parameters(), "\n")

        # Print PostgreSQL version
        cursor.execute("SELECT version();")
        record = cursor.fetchone()
        print("You are connected to - ", record, "\n")

        # Delete user_info table if exist
        cursor.execute("DROP TABLE IF EXISTS log_train;")
        # Create user_info
        cursor.execute(
            "CREATE TABLE log_train (user_id integer NOT NULL, \
                                                order_id integer NOT NULL, \
                                                product_id integer NOT NULL, \
                                                eval_set char(10), \
                                                order_numer integer, \
                                                order_dow integer, \
                                                order_hour_of_dat integer, \
                                                days_since_prior_order float, \
                                                add_to_cart_order integer, \
                                                reordered integer, \
                                                aisle_id integer, \
                                                department_id integer, \
                                                product_name integer, \
                                                PRIMARY KEY (user_id, order_id, product_id));"
        )

        connection.commit()
        print("log_train Table created successfully in PostgreSQL. \n\n")

        engine = create_engine(
            "postgresql://postgres:postgres@127.0.0.1:5432/auto_fg", echo=False
        )
        log_train.to_sql("log_train", con=engine, if_exists="replace", index=False)
    except (Exception, psycopg2.Error) as error:
        print("Error while connecting to PostgreSQL", error)
    finally:
        #         closing database connection.
        if connection:
            cursor.close()
            connection.close()
            print("PostgreSQL connection is closed")
    return engine


# generate features for pre-defined attributes
def generate_feature_in_small_space(engine, agg, m, lb_day, ub_day, w_cat, w_cat_value):
    w_cat_str = ""
    if len(w_cat) != 0:
        for i in range(len(w_cat)):
            w_cat_str = w_cat_str + str(w_cat[i]) + "=" + str(w_cat_value[i]) + " AND "

    return engine.execute(
        "SELECT user_id, "
        + str(agg)
        + "("
        + str(m)
        + ") \
                   FROM log_train \
                   WHERE "
        + w_cat_str
        + "\
                       days_since_prior_order BETWEEN "
        + str(lb_day)
        + " AND "
        + str(ub_day)
        + "\
                   GROUP BY user_id"
    ).fetchall()


def model(user_train):
    y = user_train["label"]
    y = y.to_frame()
    X = user_train.drop(["label"], axis=1)
    X = X.fillna(0)

    clf = RandomForestClassifier(random_state=0)
    accuracy = cross_validate(
        clf,
        X,
        y,
        cv=5,
        scoring=("accuracy"),
        return_train_score=True,
        n_jobs=-1,
        return_estimator=True,
    )
    valid_accuracy = accuracy["test_score"].mean()
    train_accuracy = accuracy["train_score"].mean()

    return valid_accuracy, accuracy["estimator"][0]


########## Tmall exp #########
# generate features for pre-defined attributes
def generate_feature_in_small_space_tmall(
    engine, agg, m, lb_day, ub_day, w_cat, w_cat_value
):
    w_cat_str = ""
    if len(w_cat) != 0:
        for i in range(len(w_cat)):
            w_cat_str = w_cat_str + str(w_cat[i]) + "=" + str(w_cat_value[i]) + " AND "

    return engine.execute(
        "SELECT user_id, "
        + str(agg)
        + "("
        + str(m)
        + ") \
                   FROM tmall_log \
                   WHERE "
        + w_cat_str
        + "\
                       time_stamp BETWEEN "
        + str(lb_day)
        + " AND "
        + str(ub_day)
        + "\
                   GROUP BY user_id"
    ).fetchall()


def model_tmall(user_train):
    y = user_train["label"]
    y = y.to_frame()
    X = user_train.drop(["label"], axis=1)
    X = X.fillna(0)

    clf = xgb.XGBRegressor(random_state=0)
    auc = cross_validate(
        clf,
        X,
        y,
        cv=5,
        scoring=("roc_auc"),
        return_train_score=True,
        n_jobs=-1,
        return_estimator=True,
    )
    valid_auc = auc["test_score"].mean()
    train_auc = auc["train_score"].mean()

    return valid_auc, auc["estimator"]


def store_tmall(tmall_log):
    engine = None
    try:

        connection = psycopg2.connect(
            database="auto_fg",
            user="postgres",
            password="postgres",
            host="127.0.0.1",
            port="5432",
        )
        cursor = connection.cursor()
        # Print PostgreSQL Connection properties
        print(connection.get_dsn_parameters(), "\n")

        # Print PostgreSQL version
        cursor.execute("SELECT version();")
        record = cursor.fetchone()
        print("You are connected to - ", record, "\n")

        # Delete user_info table if exist
        cursor.execute("DROP TABLE IF EXISTS tmall_log;")
        # Create user_info
        cursor.execute(
            "CREATE TABLE tmall_log (user_id integer NOT NULL, \
                                                item_id integer NOT NULL, \
                                                cat_id integer NOT NULL, \
                                                merchant_id integer NOT NULL, \
                                                brand_id float, \
                                                time_stamp integer, \
                                                action_type integer, \
                                                gender float, \
                                                age_range float, \
                                                PRIMARY KEY (user_id));"
        )

        connection.commit()
        print("tmall_log Table created successfully in PostgreSQL. \n\n")

        engine = create_engine(
            "postgresql://postgres:postgres@localhost:5432/auto_fg", echo=False
        )
        tmall_log.to_sql("tmall_log", con=engine, if_exists="replace", index=False)
    except (Exception, psycopg2.Error) as error:
        print("Error while connecting to PostgreSQL", error)
    finally:
        # closing database connection.
        if connection:
            cursor.close()
            connection.close()
            print("PostgreSQL connection is closed")
    return engine
