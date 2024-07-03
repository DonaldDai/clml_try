import pandas as pd
from sklearn.utils import resample
import math
from clearml import Task, Dataset, StorageManager
import argparse
import numpy as np

# 按比例重采样
def resample_data(data, target_ratio, random_seed=42):
    """
    对数据集中的正负样本进行重采样，以达到给定的目标比例。
    
    参数:
    data : DataFrame，包含至少一列名为 'AV_Bit' 的数据，其中 1 表示正样本，0 表示负样本。
    target_ratio : float，目标正样本的比例。
    random_seed : int，随机种子，用于确保重采样的可重复性。
    
    返回:
    DataFrame，重采样后的数据。
    """
    # 分离正负样本
    positive_samples = data[data['AV_Bit'] == 1]
    negative_samples = data[data['AV_Bit'] == 0]

    # 计算新的正样本和负样本数量
    total_samples = min(int(len(positive_samples) // target_ratio), int(len(negative_samples) // (1-target_ratio)))
    target_positive_count = math.floor(total_samples * target_ratio)
    target_negative_count = total_samples - target_positive_count
    print('采集样本', f'all: {total_samples} +:{target_positive_count}|{target_positive_count/total_samples} -:{target_negative_count}|{target_negative_count/total_samples}')
    if (target_positive_count == 0 or target_negative_count == 0):
        raise Error('数据太少')

    # 重新采样
    resampled_positive = resample(positive_samples, replace=False, n_samples=target_positive_count, random_state=random_seed)
    resampled_negative = resample(negative_samples, replace=False, n_samples=target_negative_count, random_state=random_seed)

    # 合并数据集
    resampled_data = pd.concat([resampled_positive, resampled_negative]).reset_index(drop=True)
    
    return resampled_data

# 分层采样
def stratified_sample_with_remainder(df, sample_frac, seed=None):
    """
    对DataFrame进行分层采样，并返回采样的数据和未采样的数据，同时重置索引。
    
    Args:
    - df: 输入的DataFrame。
    - sample_frac: 每个层的采样比例。
    - seed: 随机种子，确保可重复性。
    
    Returns:
    - sample_df: 采样后且索引重置的DataFrame。
    - remainder_df: 未被采样且索引重置的DataFrame。
    """
    np.random.seed(seed)
    
    # 使用 groupby 和 sample 采样数据，然后将未被采样的数据也返回
    sample_df = pd.DataFrame()  # 初始化采样 DataFrame
    remainder_df = pd.DataFrame()  # 初始化未采样 DataFrame

    for name, group in df.groupby('AV_Bit'):
        group_sampled = group.sample(frac=sample_frac)
        group_remainder = group.drop(group_sampled.index)  # 删除已采样的数据得到剩余的数据
        
        sample_df = pd.concat([sample_df, group_sampled])
        remainder_df = pd.concat([remainder_df, group_remainder])
    
    # 重置索引并丢弃旧索引
    sample_df = sample_df.reset_index(drop=True)
    remainder_df = remainder_df.reset_index(drop=True)
    
    return sample_df, remainder_df

def get_data(args):
    data = pd.DataFrame()
    # 通过dataset id获取数据
    if args.data_id:
        data_path = Dataset.get(
                dataset_id=args.data_id,
                only_completed=True,
                only_published=False,
            ).get_local_copy()
        data = pd.read_csv(f'{data_path}/data.csv')
    # 通过dateset project name和dateset name获取数据
    elif args.d_project and args.d_name:
        data_path = Dataset.get(
                dataset_id=None,
                dataset_project=args.d_project,
                dataset_name=args.d_name,
                only_completed=True,
                only_published=False,
            ).get_local_copy()
        data = pd.read_csv(f'{data_path}/data.csv')
    # 通过artifact url获取数据
    elif args.data_url:
        train_path = StorageManager.get_local_copy(args.data_url)
        data = pd.read_csv(train_path, compression='gzip')
    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Chain two Python programs that handle CSV input/output.")
    parser.add_argument("--data_id", required=False, default='', help="cleaml dataset id")
    parser.add_argument("--d_project", required=False, default='', help="dataset project name")
    parser.add_argument("--d_name", required=False, default='', help="dataset name")
    parser.add_argument("--data_url", required=False, default='', help="data url")
    parser.add_argument("--ratio", required=False, type=int, default=1, help="split ratio")
    parser.add_argument("--fraction", required=False, type=float, default=0.9, help="stratification fraction")
    parser.add_argument("--seed", required=False, type=int, default=42, help="random seed")
    args = parser.parse_args()
    task = Task.init(project_name='paper', task_name='gen_data')
    all_data = get_data(args)
    i = args.ratio
    # 使用函数
    resampled_data = resample_data(all_data, 1 / (1+i+1))
    print('resampled_data.shape', resampled_data.shape)
    # 指定采样比例
    sample_fraction = args.fraction
    # 指定随机种子
    seed = args.seed
    train, val = stratified_sample_with_remainder(all_data, sample_fraction, seed=seed)
    task.upload_artifact('train', train, wait_on_upload=True)
    task.upload_artifact('val', val, wait_on_upload=True)
    print('DONE')
    task.close()