import pandas as pd

def read_file(file_path,num_columns):
    df = pd.read_csv(file_path, sep=",",names=range(num_columns))
    return df


def concatenate_df(df1,df2):
    out = pd.merge(df1,df2,how='inner',on=0)
    return out


def write_df(merged_table,out_file):
    merged_table.to_csv(out_file, header=False,index=False)














