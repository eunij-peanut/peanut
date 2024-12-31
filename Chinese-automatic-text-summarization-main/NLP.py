import re
import jieba.posseg as posseg
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

###################################################
# TextRank实现
###################################################


# 停用词路径
stopwords_path = 'stopwords.txt'
# 需要排除的词性
stopPOS = []

# 读取停用词
with open(stopwords_path, 'r', encoding='utf-8') as f:
    stopwords = [line.strip() for line in f.readlines()]


def segment_text_to_sentence(text):
    # 将文本分割成句子
    sentences = re.split(r'[。！？!?]', text)
    sentences = [sentence.strip().replace(" ", "").replace('\n', '') for sentence in sentences if sentence.strip()]
    return sentences


def segment_text_to_words(text, use_stopwords):
    # 分词并去除停用词
    global stopPOS, stopwords
    stopPOS = [item.lower() for item in stopPOS]
    words = posseg.cut(text)
    if use_stopwords:
        words = [word for word, flag in words if flag[0].lower() not in stopPOS and word not in stopwords]
    else:
        words = [word for word, flag in words if flag[0].lower() not in stopPOS]
    words = set(words)

    return words


def original_similarity_matrix(sentences, use_stopwords):
    # 计算原始相似性矩阵
    sentence_words = [set(segment_text_to_words(item, use_stopwords)) for item in sentences]
    size = len(sentences)
    similarity_matrix = np.zeros((size, size))
    for i in range(size):
        for j in range(i + 1, size):
            if len(sentence_words[i]) == 0 or len(sentence_words[j]) == 0:
                similarity = 0
            else:
                # 计算相似性
                similarity = len(sentence_words[i] & sentence_words[j]) / (
                            np.log(len(sentence_words[i])) + np.log(len(sentence_words[i])) + 1e-10)
            similarity_matrix[i][j] = similarity_matrix[j][i] = similarity
    return similarity_matrix


def cosine_tfidf_similarity_matrix(sentences, use_stopwords):
    # 计算基于TF-IDF的余弦相似性矩阵
    sentence_words = [' '.join(segment_text_to_words(item, use_stopwords)) for item in sentences]
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(sentence_words)
    similarity_matrix = cosine_similarity(tfidf_matrix)

    # 将对角线元素设置为0，避免自身与自身的相似性干扰
    np.fill_diagonal(similarity_matrix, 0)
    return similarity_matrix


def summarize_text_rank(text, d=0.85, iter_num=200, top=3, method='默认方式', use_stopwords=True):
    sentences = segment_text_to_sentence(text)

    print('---------开始----------------------------------------')
    if method == '默认方式':
        edge_weight = original_similarity_matrix(sentences, use_stopwords)
    elif method == 'TF-IDF':
        edge_weight = cosine_tfidf_similarity_matrix(sentences, use_stopwords)

    node_weight = np.ones((len(sentences)))

    for num in range(iter_num):
        # TextRank迭代公式
        node_weight_new = (1 - d) + d * node_weight @ (edge_weight / (edge_weight.sum(axis=-1) + 1e-10)).T
        if ((node_weight_new - node_weight) ** 2).sum() < 1e-10:
            break
        node_weight = node_weight_new

    if num < iter_num:
        print('迭代{}次，收敛'.format(num))
    else:
        print('迭代{}次，未收敛'.format(num))

    sorted_indices = np.argsort(node_weight)[::-1]

    # 获取最大的几个值及其对应的索引
    top_indices = sorted(sorted_indices[:top])
    top_values = node_weight[top_indices]

    print('最大的{}个值：'.format(top), top_values)
    print('对应的索引：', top_indices)
    print('结果：')
    result = ''
    for idx in top_indices:
        result += sentences[idx] + '。\n'
    print(result)

    return result


# 示例
# text = '在这里输入你的文本'
# summarize_text_rank(text)

[2]
###################################################
# MT5实现
###################################################
# flag = True
#
# # 尝试导入必要的库和模型
# try:
#     import re
#     from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
#     from transformers import MT5ForConditionalGeneration, T5Tokenizer
#
#     # 定义用于处理空格和换行符的函数
#     WHITESPACE_HANDLER = lambda k: re.sub('\s+', ' ', re.sub('\n+', ' ', k.strip()))
#
#     # 定义MT5模型的名称
#     model_name = "./mt5-small"
#
#     # 使用AutoTokenizer加载预训练的MT5分词器
#     # tokenizer = AutoTokenizer.from_pretrained(model_name)
#
#     # 使用AutoModelForSeq2SeqLM加载预训练的MT5模型
#     # model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
# except:
#     # 如果导入失败，将flag设置为False
#     flag = False
#
#
# def summary_mt5(text):
#     try:
#         # 使用T5Tokenizer加载预训练的MT5分词器
#         # tokenizer = T5Tokenizer.from_pretrained(model_name)
#         tokenizer = AutoTokenizer.from_pretrained(model_name)
#
#         # 使用MT5ForConditionalGeneration加载预训练的MT5模型
#         # model = MT5ForConditionalGeneration.from_pretrained(model_name).to(device)
#         model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
#
#         # 加载模型权重
#         model.load_state_dict(torch.load('./mt5-small-mymodel/epoch_1_valid_rouge_25.7529_model_weights.bin'))
#         model.eval()
#
#         # 对输入文本进行编码
#         inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding=True)
#         inputs = {k: v.to(device) for k, v in inputs.items()}
#
#         # 生成摘要
#         max_target_length = 32
#         beam_size = 4
#         no_repeat_ngram_size = 2
#         with torch.no_grad():
#             generated_tokens = model.generate(
#                 inputs["input_ids"],
#                 attention_mask=inputs["attention_mask"],
#                 max_length=max_target_length,
#                 num_beams=beam_size,
#                 no_repeat_ngram_size=no_repeat_ngram_size,
#             )
#
#         # 解码生成的摘要
#         summary = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
#
#         # 去除 <extra_id_0> 标记
#         summary = summary.replace("<extra_id_0>", "").strip()
#
#         return summary
#
#     except Exception as e:
#         # 如果出现异常，返回详细的错误信息
#         return f'生成摘要时出错: {str(e)}'

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import MT5ForConditionalGeneration, T5Tokenizer

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义模型名称
model_name = "./mt5-small"

# 初始化标志
mt5_available = False

try:
    # 尝试加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    # 加载模型权重
    model.load_state_dict(
        torch.load('./mt5-small-mymodel/epoch_1_valid_rouge_25.7529_model_weights.bin', map_location=device))
    model.eval()

    mt5_available = True
except Exception as e:
    print(f"加载 MT5 模型时出错: {str(e)}")
    mt5_available = False


def summary_mt5(text):
    if not mt5_available:
        return "MT5 模型未能成功加载，无法生成摘要。"

    try:
        # 对输入文本进行编码
        inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # 生成摘要
        max_target_length = 32
        beam_size = 4
        no_repeat_ngram_size = 2
        with torch.no_grad():
            generated_tokens = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_target_length,
                num_beams=beam_size,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )

        # 解码生成的摘要
        summary = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

        # 去除 <extra_id_0> 标记
        summary = summary.replace("<extra_id_0>", "").strip()

        return summary

    except Exception as e:
        return f'生成摘要时出错: {str(e)}'



[3]
import tkinter as tk
from tkinter import ttk, scrolledtext
import jieba


###################################################
# UI界面实现
###################################################

def summarize_text():
    input_text = input_text_widget.get("1.0", "end-1c")
    d = float(d_entry.get()) if d_entry.get() else 0.85
    top = int(top_entry.get()) if top_entry.get() else 3
    processing_method = processing_method_var.get()
    use_stopwords = use_stopwords_var.get()
    summary = summarize_text_rank(input_text, d=d, top=top, method=processing_method, use_stopwords=use_stopwords)
    output_text_widget.delete(1.0, tk.END)
    output_text_widget.insert(tk.END, summary)


def summarize_text_mt5():
    input_text = input_text_widget.get("1.0", "end-1c")
    summary_result = summary_mt5(input_text)
    output_text_widget_mt5.delete(1.0, tk.END)
    output_text_widget_mt5.insert(tk.END, summary_result)


# 创建主窗口
root = tk.Tk()
root.title("中文文本自动摘要工具")

# 使用ttk模块中的样式调整
style = ttk.Style()
style.configure('TFrame', padding=10)
style.configure('TButton', padding=(10, 5), font=('Helvetica', 10))
style.configure('TLabel', font=('Helvetica', 10))

# 创建输入文本框
input_label_frame = ttk.LabelFrame(root, text="输入文本")
input_label_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew", columnspan=2)  # 设置columnspan为2，使其横跨两列
input_text_widget = scrolledtext.ScrolledText(input_label_frame, wrap=tk.WORD, width=70, height=10)
input_text_widget.pack(pady=10, fill='both', expand=True)

# 创建摘要长度输入框，设置默认值为100
frame1 = ttk.LabelFrame(root, text="TextRank参数设置")
frame1.grid(row=1, column=0, padx=10, pady=10, sticky="nsew", columnspan=2)  # 设置columnspan为2，使其横跨两列

# 创建停用词复选框
use_stopwords_var = tk.BooleanVar(root)
use_stopwords_var.set(True)  # 默认使用停用词
use_stopwords_checkbutton = ttk.Checkbutton(frame1, text="使用停用词", variable=use_stopwords_var)
use_stopwords_checkbutton.grid(row=0, column=0, pady=5)

default_d = 0.85
d_label = ttk.Label(frame1, text=f"阻尼系数:")
d_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")
d_entry = ttk.Entry(frame1, width=10)
d_entry.insert(0, str(default_d))
d_entry.grid(row=1, column=1, padx=2, pady=5)

default_top = 3
top_label = ttk.Label(frame1, text=f"摘要句数:")
top_label.grid(row=2, column=0, padx=5, pady=5, sticky="w")
top_entry = ttk.Entry(frame1, width=10)
top_entry.insert(0, str(default_top))
top_entry.grid(row=2, column=1, padx=2, pady=5)

processing_method_var = tk.StringVar(root)
processing_method_var.set("默认方式")  # 设置默认选项
processing_method_label = ttk.Label(frame1, text="相似度度量:")
processing_method_label.grid(row=3, column=0, padx=5, pady=5, sticky="w")
processing_method_menu = ttk.Combobox(frame1, textvariable=processing_method_var, values=["默认方式", "TF-IDF"],
                                      width=10)
processing_method_menu.grid(row=3, column=1, padx=2, pady=5)

# 创建按钮，用于触发文本摘要
summarize_button = ttk.Button(root, text="TextRank生成摘要", command=summarize_text, style='TButton')
summarize_button.grid(row=2, column=0, padx=(10, 5), pady=10)  # 添加横向和纵向的内边距

summarize_button_mt5 = ttk.Button(root, text="MT5生成摘要", command=summarize_text_mt5, style='TButton')
summarize_button_mt5.grid(row=2, column=1, padx=(5, 10), pady=10)  # 添加横向和纵向的内边距

# 创建输出文本框
output_label_frame = ttk.LabelFrame(root, text="TextRank输出文本")
output_label_frame.grid(row=3, column=0, padx=10, pady=10, sticky="nsew", columnspan=2)  # 设置columnspan为2，使其横跨两列
output_text_widget = scrolledtext.ScrolledText(output_label_frame, wrap=tk.WORD, width=50, height=10)
output_text_widget.pack(pady=10, fill='both', expand=True)

output_label_frame_mt5 = ttk.LabelFrame(root, text="MT5输出文本")
output_label_frame_mt5.grid(row=4, column=0, padx=10, pady=10, sticky="nsew", columnspan=2)  # 设置columnspan为2，使其横跨两列
output_text_widget_mt5 = scrolledtext.ScrolledText(output_label_frame_mt5, wrap=tk.WORD, width=50, height=10)
output_text_widget_mt5.pack(pady=10, fill='both', expand=True)

# 设置行列权重，使得在窗口变大时，文本框和标签框都能够扩展
for i in range(4):  # 设置所有行的权重为1
    root.grid_rowconfigure(i, weight=1)
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)

# 运行主循环
root.mainloop()