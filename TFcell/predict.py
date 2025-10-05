#!/home/hyangl/anaconda3/bin/python
import pyBigWig
import argparse
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  
tf.get_logger().setLevel('ERROR')
import unet
from unet import get_unet, FlowAttention
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
from tensorflow.keras.optimizers import Adam
sys.stderr = stderr

K.set_image_data_format('channels_last')  # TF dimension ordering

# 参数配置
size = 2**11 * 5  # 10240
num_channel = 6
write_pred = True
size_edge = 100
batch = 16

# 染色体长度信息
chr_all = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrX']
num_bp = np.array([249250621, 243199373, 198022430, 191154276, 180915260, 171115067, 159138663, 146364022, 141213431, 135534747, 135006516, 133851895, 115169878, 107349540, 102531392, 90354753, 81195210, 78077248, 59128983, 63025520, 48129895, 51304566, 155270560])
chr_len = {chr_all[i]: num_bp[i] for i in range(len(chr_all))}

# 数据路径
path1 = '../../data/dna_bigwig/'  # DNA 数据路径
path2 = '../../data/dnase_bigwig/'  # DNase 数据路径

# 参数解析
def get_args():
    parser = argparse.ArgumentParser(description="Predict")
    parser.add_argument('-m', '--model', default='weights_K562_IMR-90_1.h5', type=str, help='预训练模型路径')
    parser.add_argument('-tf', '--transcription_factor', default='CTCF', type=str, help='转录因子名称')
    parser.add_argument('-te', '--test', default='K562', type=str, help='测试细胞类型')
    parser.add_argument('-chr', '--chromosome', default=['chr21'], nargs='+', type=str, help='测试染色体')
    parser.add_argument('-para', '--parallel', default=1, type=int, help='控制 GPU 内存使用')
    return parser.parse_args()

args = get_args()


ss = 10
# 定义 crossentropy_cut 函数
def crossentropy_cut(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    y_pred_f = tf.clip_by_value(y_pred_f, 1e-7, (1. - 1e-7))
    mask = K.greater_equal(y_true_f, -0.5)
    losses = -(y_true_f * K.log(y_pred_f) + (1.0 - y_true_f) * K.log(1.0 - y_pred_f))
    losses = tf.boolean_mask(losses, mask)
    masked_loss = tf.reduce_mean(losses)
    return masked_loss
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    mask=K.cast(K.greater_equal(y_true_f,-0.5), dtype='float32')
    intersection = K.sum(y_true_f * y_pred_f * mask)
    return (2. * intersection + ss) / (K.sum(y_true_f * mask) + K.sum(y_pred_f * mask) + ss)
# 加载模型

model = tf.keras.models.load_model(
    args.model,
    custom_objects={
        'FlowAttention': unet.FlowAttention,
        'crossentropy_cut': crossentropy_cut,
        'dice_coef': dice_coef
    }
)
# 打开 BigWig 文件
list_dna = ['A', 'C', 'G', 'T']
dict_dna = {the_id: pyBigWig.open(f"{path1}{the_id}.bigwig") for the_id in list_dna}
feature_avg = pyBigWig.open(f"{path2}avg.bigwig")
feature_test = pyBigWig.open(f"{path2}{args.test}.bigwig")

if __name__ == '__main__':
    the_name = os.path.splitext(os.path.basename(args.model))[0]
    
    for the_chr in args.chromosome:
        print(f"Processing chromosome: {the_chr}")
        output_all = np.zeros(chr_len[the_chr])
        count_all = np.zeros(chr_len[the_chr])
        
        # 滑动窗口预测（双相位增强覆盖）
        for phase in [0, 0.5]:
            start_offset = int(size * phase)
            end_offset = chr_len[the_chr] - size + start_offset if phase != 0 else chr_len[the_chr]
            current = start_offset
            
            while current < end_offset:
                chunk_start = current
                chunk_end = current + size * batch
                if chunk_end > end_offset:
                    chunk_end = end_offset
                    chunk_start = end_offset - size * batch
                
                # 构建输入数据
                image = np.zeros((num_channel, size * batch))
                for idx, the_id in enumerate(list_dna):
                    image[idx, :] = dict_dna[the_id].values(the_chr, chunk_start, chunk_end)
                
                # 计算 DNase 特征和差异
                dnase = np.array(feature_test.values(the_chr, chunk_start, chunk_end))
                avg_dnase = np.array(feature_avg.values(the_chr, chunk_start, chunk_end))
                image[4, :] = dnase
                image[5, :] = dnase - avg_dnase
                
                # 模型预测
                input_pred = image.T.reshape(batch, size, num_channel)
                output = model.predict(input_pred)
                output_flat = output.reshape(size * batch)
                
                # 窗口融合
                for i in range(batch):
                    window_start = chunk_start + i * size
                    window_end = window_start + size
                    
                    # 边缘处理
                    valid_start = window_start + size_edge if window_start != 0 else window_start
                    valid_end = window_end - size_edge if window_end != chunk_end else window_end
                    
                    # 数据索引
                    src_start = i * size + size_edge if window_start != 0 else i * size
                    src_end = src_start + (valid_end - valid_start)
                    
                    output_all[valid_start:valid_end] += output_flat[src_start:src_end]
                    count_all[valid_start:valid_end] += 1
                
                current += int(size * batch)
        
        # 平均预测结果
        output_all = np.divide(output_all, count_all, out=np.zeros_like(output_all), where=count_all!=0)
        
        # 保存结果
        output_path = f"./output/testpred_{args.test}_{the_chr}_{the_name}.npy"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, output_all)
        print(f"预测结果已保存至：{os.path.abspath(output_path)}")

# 关闭文件
for the_id in list_dna:
    dict_dna[the_id].close()
feature_avg.close()
feature_test.close()
