import numpy as np
import pyBigWig
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from auc import score_record, calculate_auc

# 定义预测文件和真实标签文件的路径
prediction_file = '../../output/pre.npy'#选择预测文件
truth_file = '../../data/chip-seq gem peak/tf-cell.bigwig'#选择label
# 读取预测文件
predictions = np.load(prediction_file)

# 读取真实标签文件
bw = pyBigWig.open(truth_file)


chromosome = 'chr21'
start = 0
end = bw.chroms()[chromosome]  # 获取染色体的长度

# 读取指定染色体区域的真实标签
truth = np.array(bw.values(chromosome, start, end))
bw.close()
truth = np.nan_to_num(truth)
min_length = min(len(predictions), len(truth))
predictions = predictions[:min_length]
truth = truth[:min_length]

# 调用 score_record 函数
pos_values, neg_values = score_record(truth, predictions)

# 调用 calculate_auc 函数
auroc, auprc = calculate_auc(pos_values, neg_values)

print(f"AUC: {auroc}")
print(f"AUPRC: {auprc}")