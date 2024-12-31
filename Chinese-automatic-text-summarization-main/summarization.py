# -*- coding: utf-8 -*-
import re
import jieba.posseg as posseg
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tkinter as tk
from tkinter import ttk, scrolledtext

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


# MT5 implementation，使用自己的模型
flag = True

try:
    import sys

    print("Python version:", sys.version)
    print("Python path:", sys.path)

    print("Attempting to import transformers...")
    import transformers

    print("Transformers version:", transformers.__version__)

    from transformers import MT5ForConditionalGeneration, T5Tokenizer
    import os

    WHITESPACE_HANDLER = lambda k: re.sub('\s+', ' ', re.sub('\n+', ' ', k.strip()))

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_name = os.path.join(script_dir, "mt5-base")

    required_files = ['config.json', 'pytorch_model.bin', 'special_tokens_map.json', 'spiece.model',
                      'tokenizer_config.json']

    if all(os.path.exists(os.path.join(model_name, file)) for file in required_files):
        print(f"Loading tokenizer from {model_name}")
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        print(f"Loading model from {model_name}")
        model = MT5ForConditionalGeneration.from_pretrained(model_name)
        print("MT5 model loaded successfully")
    else:
        missing_files = [file for file in required_files if not os.path.exists(os.path.join(model_name, file))]
        print(f"Error: Some required files are missing: {missing_files}")
        raise FileNotFoundError(f"Missing files in {model_name}")

except Exception as e:
    print(f"Error loading MT5 model: {str(e)}")
    import traceback

    print("Full traceback:")
    print(traceback.format_exc())
    flag = False


def summary_mt5(text):
    global flag
    if not flag:
        return 'MT5模型未导入。请检查控制台输出以获取更多信息。'

    try:
        input_ids = tokenizer.encode(WHITESPACE_HANDLER(text), return_tensors="pt", max_length=512, truncation=True)

        output_ids = model.generate(
            input_ids=input_ids,
            max_length=150,
            no_repeat_ngram_size=2,
            num_beams=4
        )[0]

        summary = tokenizer.decode(output_ids, skip_special_tokens=True)
    except Exception as e:
        return f'生成摘要时出错：{str(e)}'

    return summary

# # MT5 implementation  使用google下载的模型
# flag = True
#
# try:
#     import sys
#
#     print("Python version:", sys.version)
#     print("Python path:", sys.path)
#
#     print("Attempting to import transformers...")
#     import transformers
#
#     print("Transformers version:", transformers.__version__)
#
#     from transformers import MT5ForConditionalGeneration, MT5Tokenizer
#     import os
#
#     WHITESPACE_HANDLER = lambda k: re.sub('\s+', ' ', re.sub('\n+', ' ', k.strip()))
#
#     script_dir = os.path.dirname(os.path.abspath(__file__))
#     local_model_name = os.path.join(script_dir, "mt5-base")
#     huggingface_model_name = "google/mt5-base"
#
#     if not os.path.exists(local_model_name) or not os.path.exists(os.path.join(local_model_name, "tokenizer.json")):
#         print(f"Local model files missing or incomplete. Downloading from Hugging Face...")
#         model_name = huggingface_model_name
#     else:
#         model_name = local_model_name
#
#     print(f"Loading tokenizer from {model_name}")
#     tokenizer = MT5Tokenizer.from_pretrained(model_name)
#     print(f"Loading model from {model_name}")
#     model = MT5ForConditionalGeneration.from_pretrained(model_name)
#     print("MT5 model loaded successfully")
#
# except Exception as e:
#     print(f"Error loading MT5 model: {str(e)}")
#     import traceback
#
#     print("Full traceback:")
#     print(traceback.format_exc())
#     flag = False
#
#
# def summary_mt5(text):
#     global flag
#     if not flag:
#         return 'MT5模型未导入。请检查控制台输出以获取更多信息。'
#
#     try:
#         input_ids = tokenizer(
#             [WHITESPACE_HANDLER(text)],
#             return_tensors="pt",
#             padding="max_length",
#             truncation=True,
#             max_length=512
#         )["input_ids"]
#
#         output_ids = model.generate(
#             input_ids=input_ids,
#             max_length=84,
#             no_repeat_ngram_size=2,
#             num_beams=4
#         )[0]
#
#         summary = tokenizer.decode(
#             output_ids,
#             skip_special_tokens=True,
#             clean_up_tokenization_spaces=False
#         )
#     except Exception as e:
#         return f'生成摘要时出错：{str(e)}'
#
#     return summary
#
#
# # ... (rest of the code remains the same)





# UI implementation
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


# Create main window
root = tk.Tk()
root.title("中文文本自动摘要工具")

style = ttk.Style()
style.configure('TFrame', padding=10)
style.configure('TButton', padding=(10, 5), font=('Helvetica', 10))
style.configure('TLabel', font=('Helvetica', 10))

# Input text box
input_label_frame = ttk.LabelFrame(root, text="输入文本")
input_label_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew", columnspan=2)
input_text_widget = scrolledtext.ScrolledText(input_label_frame, wrap=tk.WORD, width=70, height=10)
input_text_widget.pack(pady=10, fill='both', expand=True)

# TextRank parameters
frame1 = ttk.LabelFrame(root, text="TextRank参数设置")
frame1.grid(row=1, column=0, padx=10, pady=10, sticky="nsew", columnspan=2)

use_stopwords_var = tk.BooleanVar(root)
use_stopwords_var.set(True)
use_stopwords_checkbutton = ttk.Checkbutton(frame1, text="使用停用词", variable=use_stopwords_var)
use_stopwords_checkbutton.grid(row=0, column=0, pady=5)

default_d = 0.85
d_label = ttk.Label(frame1, text="阻尼系数:")
d_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")
d_entry = ttk.Entry(frame1, width=10)
d_entry.insert(0, str(default_d))
d_entry.grid(row=1, column=1, padx=2, pady=5)

default_top = 3
top_label = ttk.Label(frame1, text="摘要句数:")
top_label.grid(row=2, column=0, padx=5, pady=5, sticky="w")
top_entry = ttk.Entry(frame1, width=10)
top_entry.insert(0, str(default_top))
top_entry.grid(row=2, column=1, padx=2, pady=5)

processing_method_var = tk.StringVar(root)
processing_method_var.set("默认方式")
processing_method_label = ttk.Label(frame1, text="相似度度量:")
processing_method_label.grid(row=3, column=0, padx=5, pady=5, sticky="w")
processing_method_menu = ttk.Combobox(frame1, textvariable=processing_method_var, values=["默认方式", "TF-IDF"],
                                      width=10)
processing_method_menu.grid(row=3, column=1, padx=2, pady=5)

# Buttons
summarize_button = ttk.Button(root, text="TextRank生成摘要", command=summarize_text, style='TButton')
summarize_button.grid(row=2, column=0, padx=(10, 5), pady=10)

summarize_button_mt5 = ttk.Button(root, text="MT5生成摘要", command=summarize_text_mt5, style='TButton')
summarize_button_mt5.grid(row=2, column=1, padx=(5, 10), pady=10)

# Output text boxes
output_label_frame = ttk.LabelFrame(root, text="TextRank输出文本")
output_label_frame.grid(row=3, column=0, padx=10, pady=10, sticky="nsew", columnspan=2)
output_text_widget = scrolledtext.ScrolledText(output_label_frame, wrap=tk.WORD, width=50, height=10)
output_text_widget.pack(pady=10, fill='both', expand=True)

output_label_frame_mt5 = ttk.LabelFrame(root, text="MT5输出文本")
output_label_frame_mt5.grid(row=4, column=0, padx=10, pady=10, sticky="nsew", columnspan=2)
output_text_widget_mt5 = scrolledtext.ScrolledText(output_label_frame_mt5, wrap=tk.WORD, width=50, height=10)
output_text_widget_mt5.pack(pady=10, fill='both', expand=True)

# Configure grid weights
for i in range(5):
    root.grid_rowconfigure(i, weight=1)
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)

# Run main loop
if __name__ == "__main__":
    root.mainloop()