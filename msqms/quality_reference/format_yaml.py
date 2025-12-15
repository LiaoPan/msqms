import yaml
"""
freq_mapping = {
    "p1": "mean_amplitude",             # 平均振幅
    "p2": "std_amplitude",              # 振幅标准差
    "p3": "skewness_amplitude",         # 振幅偏度
    "p4": "kurtosis_amplitude",         # 振幅峰度
    "p5": "mean_frequency",             # 平均频率
    "p6": "std_frequency",              # 频率标准差
    "p7": "rms_frequency",              # 频率均方根值
    "p8": "fourth_moment_frequency",    # 频率的四阶矩
    "p9": "normalized_second_moment",   # 归一化二阶矩
    "p10": "frequency_dispersion",      # 频率离散度
    "p11": "frequency_skewness",        # 频率偏度
    "p12": "frequency_kurtosis",        # 频率峰度
    "p13": "mean_absolute_deviation",   # 频率绝对偏差均值
}
"""

# 映射字典：将 p1-p13 替换为新的名称
p1_p13_mapping = {
    "p1": "mean_amplitude",
    "p2": "std_amplitude",
    "p3": "skewness_amplitude",
    "p4": "kurtosis_amplitude",
    "p5": "mean_frequency",
    "p6": "std_frequency",
    "p7": "rms_frequency",
    "p8": "fourth_moment_frequency",
    "p9": "normalized_second_moment",
    "p10": "frequency_dispersion",
    "p11": "frequency_skewness",
    "p12": "frequency_kurtosis",
    "p13": "mean_absolute_deviation",
    "S": "form_factor",
    "C": "peak_factor",
    "I": "pulse_factor",
    "L": "margin_factor"
}

format_yaml = "opm_quality_reference.yaml"
with open(format_yaml, "r") as file:
    data = yaml.safe_load(file)

updated_data = {}
for key, value in data.items():
    new_key = p1_p13_mapping.get(key, key)  # 如果键在映射字典中，则替换，否则保持原样
    if new_key != key:
        print(f"old_key:{key}, new_key:{new_key}")
    updated_data[new_key] = value

with open(format_yaml, "w") as file:
    yaml.dump(updated_data, file, default_flow_style=False)
print("YAML文件已更新替换！")
# print("YAML 文件已更新并保存为 output.yaml")
