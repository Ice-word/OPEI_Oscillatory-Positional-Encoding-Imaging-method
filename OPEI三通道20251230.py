import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os
from tqdm import tqdm
from sklearn.preprocessing import minmax_scale
import scipy.signal as sp


def OPEI_transform(signal, sample_rate, k=30, img_size=224):
    """
    振荡位置编码图像化(OPEI)转换
    :param signal: 输入音频信号
    :param sample_rate: 采样率
    :param k: 振荡器数量
    :param img_size: 输出图像尺寸
    :return: OPEI图像
    """
    # 1. 信号预处理
    signal = signal.astype(np.float32)
    signal /= np.max(np.abs(signal))  # 归一化

    # 2. 创建多频振荡器组 (对数尺度)
    nyquist = sample_rate / 2
    min_freq = 10  # 最低频率(Hz)
    max_freq = min(nyquist * 0.8, 20000)  # 最高频率(Hz)
    frequencies = np.logspace(np.log10(min_freq), np.log10(max_freq), k)

    # 3. 计算时间点
    t = np.arange(len(signal)) / sample_rate

    # 4. 动态相位调制
    alpha = np.pi / np.max(np.abs(signal))  # 相位缩放因子
    V_matrix = np.zeros((len(signal), 2 * k))

    for i, freq in enumerate(frequencies):
        # 幅值调制相位
        phi = alpha * signal
        # 计算正交分量
        V_matrix[:, 2 * i] = np.sin(2 * np.pi * freq * t + phi)  # S分量
        V_matrix[:, 2 * i + 1] = np.cos(2 * np.pi * freq * t + phi)  # C分量

    # 5. 双通道正交投影
    weights = 1 / frequencies  # 低频权重更高
    X = np.zeros(len(signal))
    Y = np.zeros(len(signal))

    for i in range(k):
        X += weights[i] * V_matrix[:, 2 * i]  # 时间演化轴
        Y += weights[i] * V_matrix[:, 2 * i + 1]  # 频率分布轴

    # 6. 归一化并映射到图像空间
    X_norm = minmax_scale(X, feature_range=(0, img_size - 1))
    Y_norm = minmax_scale(Y, feature_range=(0, img_size - 1))

    # 7. 创建图像并赋值像素强度
    opei_image = np.zeros((img_size, img_size))

    # 转换坐标为整数
    x_coords = X_norm.astype(int)
    y_coords = Y_norm.astype(int)

    # 保留最大幅值
    for i in range(len(signal)):
        intensity = np.abs(signal[i])
        if intensity > opei_image[y_coords[i], x_coords[i]]:
            opei_image[y_coords[i], x_coords[i]] = intensity

    # 8. 增强对比度
    opei_image = (opei_image - np.min(opei_image)) / (np.max(opei_image) - np.min(opei_image) + 1e-8)
    return opei_image


def compute_channel_difference(left, right, sample_rate, k=30, img_size=224):
    """
    计算左右声道差异特征（相位差+音量差）
    使用振荡正交编码保留特征信息
    """
    # 计算音量差（幅度差）
    amp_diff = np.abs(left) - np.abs(right)

    # 计算相位差（使用希尔伯特变换）
    analytic_left = sp.hilbert(left)
    analytic_right = sp.hilbert(right)
    phase_diff = np.angle(analytic_left) - np.angle(analytic_right)

    # 组合差异特征（音量差 + 相位差）
    combined_diff = amp_diff * np.exp(1j * phase_diff)

    # 分离实部和虚部作为两个独立特征
    diff_real = np.real(combined_diff)
    diff_imag = np.imag(combined_diff)

    # 创建复合信号：实部和虚部交替
    composite_signal = np.zeros(len(left) * 2)
    composite_signal[::2] = diff_real
    composite_signal[1::2] = diff_imag

    # 使用OPEI编码复合信号
    diff_image = OPEI_transform(composite_signal, sample_rate * 2, k, img_size)
    return diff_image


def process_audio_folder(input_folder, output_folder, img_size=224):
    """
    处理文件夹中的所有WAV文件（支持双声道）
    :param input_folder: 输入文件夹路径 (包含WAV文件)
    :param output_folder: 输出文件夹路径 (保存OPEI图像)
    :param img_size: 图像尺寸
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    audio_files = [f for f in os.listdir(input_folder) if f.endswith('.wav')]

    for audio_file in tqdm(audio_files, desc="Processing audio files"):
        try:
            # 读取音频文件
            file_path = os.path.join(input_folder, audio_file)
            sample_rate, signal = wavfile.read(file_path)

            # 处理单声道/双声道
            if len(signal.shape) == 1:
                # 单声道：复制为双声道
                left_channel = signal
                right_channel = signal
            else:
                # 双声道
                left_channel = signal[:, 0]
                right_channel = signal[:, 1]

            # 分别处理左右声道
            opei_left = OPEI_transform(left_channel, sample_rate, img_size=img_size)  # R通道
            opei_right = OPEI_transform(right_channel, sample_rate, img_size=img_size)  # B通道

            # 计算声道差异特征（G通道）
            opei_diff = compute_channel_difference(
                left_channel, right_channel, sample_rate, img_size=img_size
            )

            # 创建RGB图像
            rgb_image = np.zeros((img_size, img_size, 3))
            rgb_image[:, :, 0] = opei_left  # R: 左声道
            rgb_image[:, :, 1] = opei_diff  # G: 声道差异
            rgb_image[:, :, 2] = opei_right  # B: 右声道

            # 保存图像
            output_path = os.path.join(output_folder, f"{os.path.splitext(audio_file)[0]}.png")
            plt.imsave(output_path, rgb_image, vmin=0, vmax=1)

        except Exception as e:
            print(f"Error processing {audio_file}: {str(e)}")


def visualize_results(input_folder, output_folder, num_samples=3):
    """
    可视化结果 (原始波形 + OPEI图像)
    :param input_folder: 音频文件夹
    :param output_folder: 图像输出文件夹
    :param num_samples: 显示样本数量
    """
    audio_files = [f for f in os.listdir(input_folder) if f.endswith('.wav')][:num_samples]

    plt.figure(figsize=(15, 5 * num_samples))

    for i, audio_file in enumerate(audio_files):
        # 读取音频
        file_path = os.path.join(input_folder, audio_file)
        sample_rate, signal = wavfile.read(file_path)

        # 处理单声道/双声道
        if len(signal.shape) == 1:
            left_channel = signal
            right_channel = signal
            ch_count = 1
        else:
            left_channel = signal[:, 0]
            right_channel = signal[:, 1]
            ch_count = 2

        # 获取对应图像路径
        img_path = os.path.join(output_folder, f"{os.path.splitext(audio_file)[0]}.png")
        opei_img = plt.imread(img_path)

        # 绘制原始波形
        plt.subplot(num_samples, 3, 3 * i + 1)
        plt.plot(np.arange(len(left_channel)) / sample_rate, left_channel, 'r-', alpha=0.7, label='Left')
        if ch_count > 1:
            plt.plot(np.arange(len(right_channel)) / sample_rate, right_channel, 'b-', alpha=0.5, label='Right')
        plt.title(f"Original Signal: {audio_file}")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.legend()

        # 绘制RGB图像
        plt.subplot(num_samples, 3, 3 * i + 2)
        plt.imshow(opei_img, origin='lower')
        plt.title("OPEI RGB Representation")
        plt.xlabel("Time Evolution Axis")
        plt.ylabel("Frequency Distribution Axis")

        # 绘制通道分离视图
        plt.subplot(num_samples, 3, 3 * i + 3)
        plt.subplot(1, 3, 1)
        plt.imshow(opei_img[:, :, 0], cmap='Reds', origin='lower', vmin=0, vmax=1)
        plt.title("R Channel (Left)")
        plt.subplot(1, 3, 2)
        plt.imshow(opei_img[:, :, 1], cmap='Greens', origin='lower', vmin=0, vmax=1)
        plt.title("G Channel (Difference)")
        plt.subplot(1, 3, 3)
        plt.imshow(opei_img[:, :, 2], cmap='Blues', origin='lower', vmin=0, vmax=1)
        plt.title("B Channel (Right)")
        plt.tight_layout()

    plt.savefig(os.path.join(output_folder, "OPEI_visualization_comparison.png"))
    plt.show()


# ====================== 主执行程序 ======================
if __name__ == "__main__":
    # 配置路径
    input_folder = "无人机噪声漏气中口固定切片"  # 包含WAV文件的文件夹
    output_folder = "无人机噪声漏气中口固定切片_OPEI_RGB"  # 输出图像文件夹

    # 处理所有音频文件
    process_audio_folder(input_folder, output_folder, img_size=224)

    # 可视化结果 (显示前3个样本)
    visualize_results(input_folder, output_folder, num_samples=3)
