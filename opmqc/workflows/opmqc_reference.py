"""
Obtain the range of quality metrics.
"""

def obtain_quality_ranges(dataset_metrics_pd, yaml_fname="bounds.yaml", sigma=1):
    """Calculate the 1,2,3sigma rule and generate the quality reference file(yaml)

    Parameters
    ----------
    dataset_metrics_pd : pandas.DataFrame
        the dataset metrics
    yaml_fname : str
        the name of the yaml
    sigma : int
        the sigma of metric quality distribution.
    Returns
    -------

    """
    avg_mag = dataset_metrics_pd.loc['avg_mag']
    std_mag = dataset_metrics_pd.loc['std_mag']
    upper_bound = avg_mag + sigma * std_mag
    lower_bound = avg_mag - sigma * std_mag

    # handle lower bound < 0
    lower_bound = lower_bound.apply(lambda x: max(x, 0))

    dataset_metrics_pd.loc['upper_bound'] = upper_bound
    dataset_metrics_pd.loc['lower_bound'] = lower_bound

    # convert to dict
    columns = dataset_metrics_pd.columns
    bounds_dict = {}
    df = dataset_metrics_pd.to_dict()
    for col in columns:
        mean = df[col]['avg_mag']
        std_dev = df[col]['std_mag']
        upper_bound = df[col]['upper_bound']
        lower_bound = df[col]['lower_bound']
        bounds_dict[col] = {'range': [lower_bound, upper_bound], 'mean': mean, 'std': std_dev}

    # Save into YAML
    with open(yaml_fname, 'w') as file:
        yaml.dump(bounds_dict, file, default_flow_style=False)

    return dataset_metrics_pd



if __name__ == '__main__':

    # get opm quality references
    # avg_dataset_df = (opm_cog + opm_face) / 2.
    # obtain_quality_ranges(avg_dataset_df, yaml_fname='opm_quality_reference.yaml')
    #
    # # get squid quality references
    # avg_dataset_squid_df = (masc_df + hcp_df + omega_df) / 3
    # obtain_quality_ranges(avg_dataset_squid_df, yaml_fname='squid_quality_reference.yaml')

    import yaml
    # 从YAML文件加载参考指标数据
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # 遍历配置文件中的指标
    for metric, data in config.items():
        mean = data["mean"]
        std_dev = data["std"]
        sigma_3_range = data["range"]
        print(f"{metric} - 3σ range: {sigma_3_range}")
