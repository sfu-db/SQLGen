import optuna
import time
import duckdb
import numpy as np
from typing import Any, List, Optional, Union
import warnings

from optuna.samplers import TPESampler, RandomSampler
from optuna.trial import FrozenTrial
from optuna.trial import create_trial
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBClassifier
from sklearn.model_selection import cross_validate
from sklearn.feature_selection import mutual_info_classif

warnings.filterwarnings("ignore")


class QueryTemplate(object):
    """Query Template T(F, A, P, K)"""

    def __init__(
        self,
        fkeys,
        agg_funcs,
        agg_attrs,
        predicate_attrs,
        groupby_keys,
        predicate_attrs_type,
    ) -> None:
        self.fkeys = fkeys
        self.agg_funcs = agg_funcs
        self.agg_attrs = agg_attrs
        self.predicate_attrs = predicate_attrs
        self.groupby_keys = groupby_keys
        self.predicate_attrs_type = predicate_attrs_type


class SQLGen(object):
    def __init__(
        self,
    ) -> None:
        self.metric = None
        self.ml_model = None
        self.relevant_table = None
        self.base_table = None
        self.labels = None
        self.query_template = None
        self.optimal_query_list = None

    def reinit(
        self,
    ) -> None:
        self.__init__()

    def build_task(
        self, query_template: QueryTemplate, base_table, labels, relevant_table
    ) -> Any:
        self.query_template = query_template
        self.base_table = base_table
        self.labels = labels
        self.relevant_table = relevant_table
        return self

    def optimize(
        self,
        base_sampler: str = "tpe",
        ml_model: str = "xgb",
        metric: str = "roc_auc",
        outer_budget_type: str = "trial",
        outer_budget: Union[float, int] = 5,
        mi_budget: Union[float, int] = 1000,
        mi_topk: int = 100,
        base_tpe_budget: Union[float, int] = 500,
        turn_on_mi: bool = True,
        turn_on_mapping_func: bool = True,
        mapping_func: str = "RandomForest",
        seed: int = 0,
        mi_seeds: List = [572, 1567, 2711, 25, 5737],
    ) -> Optional[List]:

        self.ml_model = ml_model
        self.metric = metric
        self.optimal_query_list = []

        # mi_seeds = [572, 1567, 2711, 25, 5737]
        # mi_seeds = [89, 572, 1024, 25, 3709]
        mi_seeds_pos = 0

        while outer_budget > 0:
            start = time.time()
            observed_query_list = []
            if turn_on_mi:
                mi_study = optuna.create_study(
                    direction="maximize",
                    sampler=TPESampler(
                        n_startup_trials=20, seed=mi_seeds[mi_seeds_pos]
                    ),
                )
                mi_study.optimize(self._mi_objective_func, n_trials=mi_budget)
                mi_seeds_pos += 1
                # Change for loop according to frozen trials
                mi_trials = mi_study.get_trials()
                topk_mi_trials = self._rank_trials(mi_trials)[:mi_topk]
                # Real evaluate topk_mi_trials
                for trial in topk_mi_trials:
                    real_evaluation = self._get_real_evaluation(trial["param"])
                    observed_query_list.append(
                        {
                            "param": trial["param"],
                            "mi_value": trial["value"],
                            "real_value": real_evaluation,
                        }
                    )

                if turn_on_mapping_func:
                    mapping_func = self._learn_mapping_func(observed_query_list)
                    for trial in mi_trials:
                        evaluated = False
                        predicted_evaluation = mapping_func.predict(
                            np.array([trial.value]).reshape(-1, 1)
                        )
                        for topk_mi_trial in topk_mi_trials:
                            if trial.params == topk_mi_trial["param"]:
                                evaluated = True
                        if not evaluated:
                            observed_query_list.append(
                                {
                                    "param": trial.params,
                                    "mi_value": trial.value,
                                    "real_value": predicted_evaluation[0],
                                }
                            )
                    # how to warm start with learned mapping function? 需不需要改变inner code？
            # Warm start with mi_study (observed_query_list)
            temp_study = optuna.create_study(study_name="temp", sampler=TPESampler())
            temp_study.optimize(self._mi_objective_func, n_trials=1)
            distributions = temp_study.best_trial.distributions

            if base_sampler == "tpe":
                base_study = optuna.create_study(
                    direction="maximize",
                    sampler=TPESampler(n_startup_trials=20, seed=seed),
                )
            elif base_sampler == "random":
                base_study = optuna.create_study(
                    direction="maximize", sampler=RandomSampler(seed=seed)
                )

            for observed_query in observed_query_list:
                trial = create_trial(
                    params=observed_query["param"],
                    distributions=distributions,
                    value=observed_query["real_value"],
                )
                base_study.add_trial(trial)
            base_study.optimize(self._objective_func, n_trials=base_tpe_budget)
            best_trial = base_study.best_trial
            self.optimal_query_list.append(
                {"param": best_trial.params, "value": best_trial.value}
            )
            end = time.time()
            if outer_budget_type == "trial":
                outer_budget -= 1
            elif outer_budget_type == "time":
                outer_budget -= end - start
            # add new feature to base table
            arg_list = []
            for key in best_trial.params:
                arg_list.append(best_trial.params[key])
            new_feature, join_keys = self._generate_new_feature(arg_list=arg_list)
            self.base_table = self.base_table.merge(
                new_feature, how="left", left_on=join_keys, right_on=join_keys
            )

            print(f"Top {outer_budget} is completed!")
        return self.optimal_query_list

    def generate_new_feature(self, arg_list) -> Any:
        new_feature, join_keys = self._generate_new_feature(arg_list)
        return new_feature, join_keys

    def _objective_func(self, trial) -> Any:
        next_trial_param = self._suggest_next_trial(trial)
        new_feature, join_keys = self._generate_new_feature(arg_list=next_trial_param)
        new_train_data = self.base_table.merge(
            new_feature, how="left", left_on=join_keys, right_on=join_keys, copy=False
        )
        new_train_data = new_train_data.fillna(0)

        # cross-validation score
        if self.ml_model == "xgb":
            clf = XGBClassifier(random_state=0)
            scores = cross_validate(
                clf,
                new_train_data,
                self.labels.to_frame(),
                cv=5,
                scoring=self.metric,
                return_train_score=True,
                n_jobs=-1,
                return_estimator=True,
            )
        elif self.ml_model == "rf":
            clf = RandomForestClassifier(random_state=0)
            scores = cross_validate(
                clf,
                new_train_data,
                self.labels.values.ravel(),
                cv=5,
                scoring=self.metric,
                return_train_score=True,
                n_jobs=-1,
                return_estimator=True,
            )
        valid_score = scores["test_score"].mean()

        return valid_score

    def _mi_objective_func(self, trial):
        next_trial_param = self._suggest_next_trial(trial)
        new_feature, join_keys = self._generate_new_feature(arg_list=next_trial_param)
        df_with_fkeys = self.base_table[self.query_template.fkeys]
        new_feature_after_join = df_with_fkeys.merge(
            new_feature, how="left", left_on=join_keys, right_on=join_keys
        )
        new_feature_after_join = new_feature_after_join.drop(
            columns=self.query_template.fkeys
        )
        new_feature_after_join = new_feature_after_join.fillna(0)

        mi_score = mutual_info_classif(
            new_feature_after_join, self.labels, random_state=0
        )

        return mi_score

    def _get_real_evaluation(self, param):
        arg_list = []
        for key in param:
            arg_list.append(param[key])
        new_feature, join_keys = self._generate_new_feature(arg_list=arg_list)
        new_train_data = self.base_table.merge(
            new_feature, how="left", left_on=join_keys, right_on=join_keys, copy=False
        )
        new_train_data = new_train_data.fillna(0)

        # cross-validation score
        if self.ml_model == "xgb":
            clf = XGBClassifier(random_state=0)
            scores = cross_validate(
                clf,
                new_train_data,
                self.labels.to_frame(),
                cv=5,
                scoring=self.metric,
                return_train_score=True,
                n_jobs=-1,
                return_estimator=True,
            )
        elif self.ml_model == "rf":
            clf = RandomForestClassifier(random_state=0)
            scores = cross_validate(
                clf,
                new_train_data,
                self.labels.values.ravel(),
                cv=5,
                scoring=self.metric,
                return_train_score=True,
                n_jobs=-1,
                return_estimator=True,
            )
        valid_score = scores["test_score"].mean()

        return valid_score

    def _suggest_next_trial(self, trial) -> Optional[List]:
        agg_func_suggestion = [
            trial.suggest_categorical(
                "agg_func",
                np.array([i for i in range(len(self.query_template.agg_funcs))]),
            )
        ]

        agg_attr_suggestion = [
            trial.suggest_categorical(
                "agg_attr",
                np.array([i for i in range(len(self.query_template.agg_attrs))]),
            )
        ]

        predicate_attrs_suggestion = []
        for predicate_attr in self.query_template.predicate_attrs:
            predicate_type = self.query_template.predicate_attrs_type[predicate_attr][
                "type"
            ]
            if predicate_type == "categorical":
                predicate_choices = self.query_template.predicate_attrs_type[
                    predicate_attr
                ]["choices"]
                predicate_attrs_suggestion.append(
                    trial.suggest_categorical(
                        name=f"predicate_attr_{predicate_attr}",
                        choices=predicate_choices,
                    )
                )
            elif predicate_type == "datetime":
                predicate_choices = self.query_template.predicate_attrs_type[
                    predicate_attr
                ]["choices"]
                predicate_attrs_suggestion.append(
                    trial.suggest_categorical(
                        name=f"predicate_attr_{predicate_attr}_bound1",
                        choices=predicate_choices,
                    )
                )
                predicate_attrs_suggestion.append(
                    trial.suggest_categorical(
                        name=f"predicate_attr_{predicate_attr}_bound2",
                        choices=predicate_choices,
                    )
                )
            elif predicate_type == "float":
                predicate_low = self.query_template.predicate_attrs_type[
                    predicate_attr
                ]["low"]
                predicate_high = self.query_template.predicate_attrs_type[
                    predicate_attr
                ]["high"]
                predicate_attrs_suggestion.append(
                    trial.suggest_float(
                        name=f"predicate_attr_{predicate_attr}_bound1",
                        low=predicate_low,
                        high=predicate_high,
                    )
                )
                predicate_attrs_suggestion.append(
                    trial.suggest_float(
                        name=f"predicate_attr_{predicate_attr}_bound2",
                        low=predicate_low,
                        high=predicate_high,
                    )
                )
            elif predicate_type == "int":
                predicate_low = self.query_template.predicate_attrs_type[
                    predicate_attr
                ]["low"]
                predicate_high = self.query_template.predicate_attrs_type[
                    predicate_attr
                ]["high"]
                predicate_attrs_suggestion.append(
                    trial.suggest_int(
                        name=f"predicate_attr_{predicate_attr}_bound1",
                        low=predicate_low,
                        high=predicate_high,
                    )
                )
                predicate_attrs_suggestion.append(
                    trial.suggest_int(
                        name=f"predicate_attr_{predicate_attr}_bound2",
                        low=predicate_low,
                        high=predicate_high,
                    )
                )
            elif predicate_type == "loguniform":
                predicate_low = self.query_template.predicate_attrs_type[
                    predicate_attr
                ]["low"]
                predicate_high = self.query_template.predicate_attrs_type[
                    predicate_attr
                ]["high"]
                predicate_attrs_suggestion.append(
                    trial.suggest_loguniform(
                        name=f"predicate_attr_{predicate_attr}_bound1",
                        low=predicate_low,
                        high=predicate_high,
                    )
                )
                predicate_attrs_suggestion.append(
                    trial.suggest_loguniform(
                        name=f"predicate_attr_{predicate_attr}_bound2",
                        low=predicate_low,
                        high=predicate_high,
                    )
                )
            elif predicate_type == "uniform":
                predicate_low = self.query_template.predicate_attrs_type[
                    predicate_attr
                ]["low"]
                predicate_high = self.query_template.predicate_attrs_type[
                    predicate_attr
                ]["high"]
                predicate_attrs_suggestion.append(
                    trial.suggest_uniform(
                        name=f"predicate_attr_{predicate_attr}_bound1",
                        low=predicate_low,
                        high=predicate_high,
                    )
                )
                predicate_attrs_suggestion.append(
                    trial.suggest_uniform(
                        name=f"predicate_attr_{predicate_attr}_bound2",
                        low=predicate_low,
                        high=predicate_high,
                    )
                )
            elif predicate_type == "discrete_uniform":
                predicate_low = self.query_template.predicate_attrs_type[
                    predicate_attr
                ]["low"]
                predicate_high = self.query_template.predicate_attrs_type[
                    predicate_attr
                ]["high"]
                predicate_q = self.query_template.predicate_attrs_type[predicate_attr][
                    "q"
                ]
                predicate_attrs_suggestion.append(
                    trial.suggest_discrete_uniform(
                        name=f"predicate_attr_{predicate_attr}_bound1",
                        low=predicate_low,
                        high=predicate_high,
                        q=predicate_q,
                    )
                )
                predicate_attrs_suggestion.append(
                    trial.suggest_discrete_uniform(
                        name=f"predicate_attr_{predicate_attr}_bound2",
                        low=predicate_low,
                        high=predicate_high,
                        q=predicate_q,
                    )
                )

        groupby_keys_suggestion = []
        if len(self.query_template.groupby_keys) == 1:
            groupby_keys_suggestion.append(
                trial.suggest_categorical(name=f"groupby_keys", choices=np.array([1]))
            )
        else:
            groupby_keys_suggestion.append(
                trial.suggest_categorical(
                    name=f"groupby_keys_{self.query_template.groupby_keys[0]}",
                    choices=np.array([1]),
                )
            )
            for groupby_key in self.query_template.groupby_keys[1:]:
                groupby_keys_suggestion.append(
                    trial.suggest_categorical(
                        name=f"groupby_keys_{groupby_key}", choices=np.array([0, 1])
                    )
                )

        arg_list = (
            agg_func_suggestion
            + agg_attr_suggestion
            + predicate_attrs_suggestion
            + groupby_keys_suggestion
        )

        return arg_list

    def _generate_new_feature(self, arg_list: List = []):
        fkeys_in_sql = ""
        for fkey in self.query_template.fkeys:
            fkeys_in_sql += f"{fkey}, "
        fkeys_in_sql = fkeys_in_sql[: (len(fkeys_in_sql) - 2)]

        agg_func_in_sql = self.query_template.agg_funcs[arg_list[0]]
        agg_attr_in_sql = self.query_template.agg_attrs[arg_list[1]]
        predicate_attrs_label = arg_list[
            2 : (len(arg_list) - len(self.query_template.groupby_keys))
        ]
        groupby_keys_label = arg_list[
            (len(arg_list) - len(self.query_template.groupby_keys)) :
        ]

        where_clause_in_sql = ""
        predicate_attrs_label_pos = 0
        for i in range(len(self.query_template.predicate_attrs)):
            predicate_attr = self.query_template.predicate_attrs[i]
            predicate_type = self.query_template.predicate_attrs_type[predicate_attr][
                "type"
            ]
            if predicate_type == "categorical":
                chosen_value = predicate_attrs_label[predicate_attrs_label_pos]
                if chosen_value != "None":
                    where_clause_in_sql += f"{predicate_attr} = {chosen_value} AND "
                predicate_attrs_label_pos += 1
            elif predicate_type in (
                "float",
                "int",
                "loguniform",
                "uniform",
                "discrete_uniform",
                "datetime",
            ):
                chosen_value1 = predicate_attrs_label[predicate_attrs_label_pos]
                chosen_value2 = predicate_attrs_label[predicate_attrs_label_pos + 1]
                if chosen_value1 == "None" and chosen_value2 != "None":
                    where_clause_in_sql += f"{predicate_attr} >= {chosen_value2} AND "
                elif chosen_value2 == "None" and chosen_value1 != "None":
                    where_clause_in_sql += f"{predicate_attr} >= {chosen_value1} AND "
                elif chosen_value1 == "None" and chosen_value2 == "None":
                    continue
                elif chosen_value1 <= chosen_value2:
                    where_clause_in_sql += f"{predicate_attr} BETWEEN {chosen_value1} AND {chosen_value2} AND "
                elif chosen_value2 <= chosen_value1:
                    where_clause_in_sql += f"{predicate_attr} BETWEEN {chosen_value2} AND {chosen_value1} AND "
                predicate_attrs_label_pos += 2
            elif predicate_type == "datetime":
                chosen_value1 = predicate_attrs_label[predicate_attrs_label_pos]
                chosen_value2 = predicate_attrs_label[predicate_attrs_label_pos + 1]
                if chosen_value1 == "None" and chosen_value2 != "None":
                    where_clause_in_sql += f"{predicate_attr} >= {chosen_value2} AND "
                elif chosen_value2 == "None" and chosen_value1 != "None":
                    where_clause_in_sql += f"{predicate_attr} >= {chosen_value1} AND "
                elif chosen_value1 == "None" and chosen_value2 == "None":
                    continue
                elif int(chosen_value1) <= int(chosen_value2):
                    where_clause_in_sql += f"{predicate_attr} BETWEEN {chosen_value1} AND {chosen_value2} AND "
                elif int(chosen_value2) <= int(chosen_value1):
                    where_clause_in_sql += f"{predicate_attr} BETWEEN {chosen_value2} AND {chosen_value1} AND "
                predicate_attrs_label_pos += 2
        where_clause_in_sql = where_clause_in_sql[: (len(where_clause_in_sql) - 5)]

        groupby_clause_in_sql = ""
        join_keys = []
        for i in range(len(groupby_keys_label)):
            if groupby_keys_label[i] == 1:
                groupby_clause_in_sql += self.query_template.groupby_keys[i] + ", "
                join_keys.append(self.query_template.groupby_keys[i])
        groupby_clause_in_sql = groupby_clause_in_sql[
            : (len(groupby_clause_in_sql) - 2)
        ]
        fkeys_in_sql = groupby_clause_in_sql

        relevant_table = self.relevant_table
        if len(where_clause_in_sql) > 0:
            feature_sql = (
                f"SELECT {fkeys_in_sql}, {agg_func_in_sql}({agg_attr_in_sql}) "
                f"FROM relevant_table "
                f"WHERE {where_clause_in_sql} "
                f"GROUP BY {groupby_clause_in_sql} "
            )
        else:
            feature_sql = (
                f"SELECT {fkeys_in_sql}, {agg_func_in_sql}({agg_attr_in_sql}) "
                f"FROM relevant_table "
                f"GROUP BY {groupby_clause_in_sql} "
            )

        new_feature = duckdb.query(feature_sql).df()
        # new_feature = new_feature.astype("float")
        # print(new_feature.columns)

        return new_feature, join_keys

    def _rank_trials(self, trials: Optional[FrozenTrial] = None) -> Optional[List]:
        # extract parameter and values list
        param_value_list = []
        for trial in trials:
            param_value_list.append({"param": trial.params, "value": trial.value})
        param_value_list = sorted(
            param_value_list, key=lambda x: x["value"], reverse=True
        )
        return param_value_list

    def _learn_mapping_func(self, observed_query_list: Optional[List] = None) -> Any:
        X = np.array([x["mi_value"] for x in observed_query_list])
        y = np.array([x["real_value"] for x in observed_query_list])
        # clf = RandomForestRegressor(random_state=0)
        clf = DecisionTreeRegressor(max_depth=2, random_state=0)
        # clf = LinearRegression()
        clf.fit(X.reshape(-1, 1), y)
        return clf
