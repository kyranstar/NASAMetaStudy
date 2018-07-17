import datasampling
import pandas as pd
import utility


def calculate_categorical_portions(categorical_cols, data_file):
    """
    Given a data file, it creates a map of the categorical values to the
    percentage of the time they show up. For example, if a file has one
    categorical value, "Country", with two outcomes, "America" and "Canada"
    which appear .6 and .4 of the time respectively, it returns
    {"Country": [("America", 0.6), ("Canada", 0.4)]}.
    """

    portions = {}
    # For each categorical variable
    for col in categorical_cols:
        # Calculate the portions of variables inside it
        col_portion = {}
        num_entries = len(data_file[col])
        for entry in data_file[col]:
            if not entry in col_portion:
                col_portion[entry] = 0.0
            col_portion[entry] += 1.0/num_entries
        # convert to tuple list
        col_portion_list = []
        for entry in col_portion:
            col_portion_list.append((entry, col_portion.get(entry, 0.0)))
        portions[col] = col_portion_list
    return portions


if __name__ == "__main__":
    print("DATA")
    df = pd.read_csv("../mice_real_data_1cat.csv").drop('Unnamed: 0', axis=1).drop('sub', axis=1)
    print(df)
    categorical_cols = ['x15']

    print('CAT PORTIONS')
    cat_portions = calculate_categorical_portions(categorical_cols, df)
    print(cat_portions)

    print('MEAN')
    mean = df.mean(axis=0)
    mean['x15'] = 0.0
    print(mean)

    print('COV')
    cov = utility.correlations(df, cat_portions)
    print(cov)

    print('VARIABLES')
    variables = df.columns.tolist()
    print(variables)

    print('DUMMY COLS')
    #dummy_cols = utility.calculate_dummy(df, categorical_cols)
    # print(dummy_cols)

    dependent_var = 'rff1'

    model = datasampling.DataModel(mean, cov, variables,
                                   cat_portions, [], dependent_var)

    # print('SAMPLES')
    output = datasampling.get_distribution_samples(model, 1000, 'x1+x2')
    # print(output)
    print("NEW CAT PORTIONS")
    #dummies = output[['x15_GG', 'x15_GT', 'x15_TT']].idxmax(axis=1)
    #output = output.drop(['x15_GG', 'x15_GT', 'x15_TT'], axis=1)
    #output.loc[:, 'x15'] = dummies
    # print(output)
    print(calculate_categorical_portions(categorical_cols, output))
    print("OLD")
    print(cat_portions)
