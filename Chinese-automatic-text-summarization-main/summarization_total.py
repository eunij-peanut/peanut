import re
import jieba.posseg as posseg
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

###################################################
# TextRank实现
###################################################

[2]
###################################################
# MT5文本摘要实现
###################################################

flag = True

# 尝试导入必要的库和模型
try:
    import re
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    from transformers import MT5ForConditionalGeneration, T5Tokenizer

    # 定义用于处理空格和换行符的函数
    WHITESPACE_HANDLER = lambda k: re.sub('\s+', ' ', re.sub('\n+', ' ', k.strip()))

    # 定义MT5模型的名称
    model_name = "./mt5-base"

    # 使用AutoTokenizer加载预训练的MT5分词器
    # tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 使用AutoModelForSeq2SeqLM加载预训练的MT5模型
    # model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
except:
    # 如果导入失败，将flag设置为False
    flag = False


# def summary_mt5(text):
#     global flag
#     # 检查MT5模型是否成功导入
#     if not flag:
#         return 'MT5模型未导入'
#
#     try:
#         # 使用MT5分词器对输入文本进行处理，并生成输入的token ID
#         input_ids = tokenizer(
#             [WHITESPACE_HANDLER(text)],
#             return_tensors="pt",
#             padding="max_length",
#             truncation=True,
#             max_length=512
#         )["input_ids"]
#
#         # 使用MT5模型生成摘要
#         output_ids = model.generate(
#             input_ids=input_ids,
#             max_length=84,
#             no_repeat_ngram_size=2,
#             num_beams=4
#         )[0]
#
#         # 解码生成的token ID，得到摘要
#         summary = tokenizer.decode(
#             output_ids,
#             skip_special_tokens=True,
#             clean_up_tokenization_spaces=False
#         )
#     except:
#         # 如果出现异常，提示检查Transformers的版本号
#         return '请检查Transformers的版本号'
#
#     return summary


def summary_mt5(text):
    try:
        # 使用T5Tokenizer加载预训练的MT5分词器
        tokenizer = T5Tokenizer.from_pretrained(model_name)

        # 使用MT5ForConditionalGeneration加载预训练的MT5模型
        model = MT5ForConditionalGeneration.from_pretrained(model_name).to(device)

        # 使用MT5分词器对输入文本进行处理，并生成输入的token ID
        inputs = tokenizer([WHITESPACE_HANDLER(text)],
                           return_tensors="pt",
                           padding="max_length",
                           truncation=True,
                           max_length=512)

        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        # 使用MT5模型生成摘要
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=150,
            no_repeat_ngram_size=2,
            num_beams=4
        )[0]

        # 解码生成的token ID，得到摘要
        summary = tokenizer.decode(output_ids,
                                   skip_special_tokens=True,
                                   clean_up_tokenization_spaces=False)

        return summary

    except Exception as e:
        # 如果出现异常，返回详细的错误信息
        return f'生成摘要时出错: {str(e)}'

###################################################
# MT5对话摘要实现
###################################################
flag = True

# 尝试导入必要的库和模型
try:
    import torch
    from transformers import MT5ForConditionalGeneration
    import jieba
    from transformers import BertTokenizer
    import argparse

    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    # 定义一个名为 T5PegasusTokenizer 的类，它继承自 BertTokenizer 类。
    # 继承意味着 T5PegasusTokenizer 将拥有 BertTokenizer 的所有方法和属性，并可以添加或修改它们。
    class T5PegasusTokenizer(BertTokenizer):

        # __init__ 是类的构造函数，在创建类的实例时自动调用。
        # *args 和 **kwargs 是用于接收任意数量的位置参数和关键字参数的特殊语法。
        # 这里，它们被传递给父类 BertTokenizer 的构造函数，以确保父类被正确初始化。
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)  # 调用父类 BertTokenizer 的构造函数。

        # pre_tokenizer 是一个自定义方法，用于在正式分词之前对文本进行预处理。
        # 这里，它使用 jieba 库进行中文分词。
        def pre_tokenizer(self, x):
            # jieba.cut 是 jieba 库中的一个函数，用于将中文文本切分成词语。
            # HMM=False 表示不使用隐马尔可夫模型进行新词发现，这通常会加快分词速度。
            return jieba.cut(x, HMM=False)  # 返回分词后的生成器对象。

        # _tokenize 是一个重写的方法，用于定义如何将文本切分成词语或标记。
        # 这个方法会覆盖 BertTokenizer 中的同名方法。
        def _tokenize(self, text, *arg, **kwargs):
            # 初始化一个空列表，用于存储分词结果。
            split_tokens = []

            # 使用 pre_tokenizer 方法对文本进行预处理，jieba.cut 返回一个生成器，逐个产生分词结果。
            for text in self.pre_tokenizer(text):
                # 检查分词后的词语是否在词汇表中。
                if text in self.vocab:
                    # 如果词语在词汇表中，直接将其添加到分词结果列表中。
                    split_tokens.append(text)
                else:
                    # 如果词语不在词汇表中，调用父类的 _tokenize 方法进行进一步分词，
                    # 并将结果扩展到分词结果列表中。
                    split_tokens.extend(super()._tokenize(text))

            # 返回最终的分词结果列表。
            return split_tokens


    def init_argument():
        parser = argparse.ArgumentParser(description='t5-pegasus-chinese')
        parser.add_argument('--pretrain_model', default='./t5_pegasus_pretrain')
        parser.add_argument('--model', default='./saved_model_12.24/summary_model')
        parser.add_argument('--max_len', default=512, help='max length of inputs')
        parser.add_argument('--max_len_generate', default=40, help='max length of generated text')
        args = parser.parse_args()
        return args

    def generate_summary(input_text, model, tokenizer, args):
        model.eval()
        input_ids = tokenizer.encode(input_text, max_length=args.max_len, truncation=True, return_tensors='pt').to(
            device)
        attention_mask = torch.ones_like(input_ids)

        gen = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=args.max_len_generate,
            eos_token_id=tokenizer.sep_token_id,
            decoder_start_token_id=tokenizer.cls_token_id,
        )

        summary = tokenizer.decode(gen[0], skip_special_tokens=True)
        return summary.replace(' ', '')

    # 定义用于处理空格和换行符的函数
    # WHITESPACE_HANDLER = lambda k: re.sub('\s+', ' ', re.sub('\n+', ' ', k.strip()))

    # 定义MT5模型的名称
    # model_name = "./mt5-base"

    # 使用AutoTokenizer加载预训练的MT5分词器
    #tokenizer = AutoTokenizer.from_pretrained(model_name)
    # 使用AutoModelForSeq2SeqLM加载预训练的MT5模型
    #model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    args = init_argument()
    # Load tokenizer and model
    tokenizer = T5PegasusTokenizer.from_pretrained(args.pretrain_model)
    model = torch.load(args.model, map_location=device)
except:
    # 如果导入失败，将flag设置为False
    flag = False


def summary_mt5_dialogue(text):
    global flag
    # 检查MT5模型是否成功导入
    if not flag:
        return 'MT5模型未导入'

    try:
        summary = generate_summary(text, model, tokenizer, args)
    except:
        # 如果出现异常，提示检查Transformers的版本号
        return '请检查Transformers的版本号'

    return summary

[3]
import tkinter as tk
from tkinter import ttk, scrolledtext
import jieba

###################################################
# UI界面实现
###################################################

# 定义一个函数，用于基于MT5模型生成对话摘要
def summarize_text_mt5_dialogue():
    # 从输入文本框中获取文本
    input_text = input_text_widget.get("1.0", "end-1c")
    # 调用summary_mt5函数生成摘要
    summary_result = summary_mt5_dialogue(input_text)
    # 清空MT5输出文本框
    output_text_widget_mt5_dialogue.delete(1.0, tk.END)
    # 在MT5输出文本框中插入生成的摘要
    output_text_widget_mt5_dialogue.insert(tk.END, summary_result)

# 定义一个函数，用于基于MT5模型生成文本摘要
def summarize_text_mt5():
    # 从输入文本框中获取文本
    input_text = input_text_widget.get("1.0", "end-1c")
    # 调用summary_mt5函数生成摘要
    summary_result = summary_mt5(input_text)
    # 清空MT5输出文本框
    output_text_widget_mt5.delete(1.0, tk.END)
    # 在MT5输出文本框中插入生成的摘要
    output_text_widget_mt5.insert(tk.END, summary_result)

# 导入tkinter库，并创建主窗口
root = tk.Tk()
# 设置窗口标题
root.title("中文文本自动摘要工具")

# 使用ttk模块中的样式调整
style = ttk.Style()
# 设置TFrame组件的内边距
style.configure('TFrame', padding=10)
# 设置TButton组件的内边距和字体
style.configure('TButton', padding=(10, 5), font=('Helvetica', 10))
# 设置TLabel组件的字体
style.configure('TLabel', font=('Helvetica', 10))

# 创建一个标签框架，用于包含输入文本框
input_label_frame = ttk.LabelFrame(root, text="输入文本")
# 将标签框架放置在网格布局中，设置列跨度为2
input_label_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew", columnspan=2)
# 创建一个滚动文本框，用于输入文本
input_text_widget = scrolledtext.ScrolledText(input_label_frame, wrap=tk.WORD, width=70, height=10)
# 将滚动文本框放置在标签框架中，并设置其扩展和填充方式
input_text_widget.pack(pady=10, fill='both', expand=True)


# 创建一个按钮，点击时调用summarize_text函数生成摘要
summarize_button = ttk.Button(root, text="MT5生成对话摘要", command=summarize_text_mt5_dialogue, style='TButton')
# 将按钮放置在网格布局中
summarize_button.grid(row=2, column=0, padx=(10, 5), pady=10)

# 创建一个按钮，点击时调用summarize_text_mt5函数生成摘要
summarize_button_mt5 = ttk.Button(root, text="MT5生成文本摘要", command=summarize_text_mt5, style='TButton')
# 将按钮放置在网格布局中
summarize_button_mt5.grid(row=2, column=1, padx=(5, 10), pady=10)

# 创建一个标签框架，用于包含Mt5生成的对话摘要
output_label_frame = ttk.LabelFrame(root, text="MT5输出对话摘要文本")
# 将标签框架放置在网格布局中，设置列跨度为2
output_label_frame.grid(row=3, column=0, padx=10, pady=10, sticky="nsew", columnspan=2)
# 创建一个滚动文本框，用于显示TextRank生成的摘要
output_text_widget_mt5_dialogue = scrolledtext.ScrolledText(output_label_frame, wrap=tk.WORD, width=50, height=10)
# 将滚动文本框放置在标签框架中，并设置其扩展和填充方式
output_text_widget_mt5_dialogue.pack(pady=10, fill='both', expand=True)


# 将滚动文本框放置在标签框架
# 创建一个标签框架，用于包含MT5生成的文本摘要
output_label_frame_mt5 = ttk.LabelFrame(root, text="MT5输出文本摘要文本")
# 将标签框架放置在网格布局中，设置列跨度为2
output_label_frame_mt5.grid(row=4, column=0, padx=10, pady=10, sticky="nsew", columnspan=2)  # 设置columnspan为2，使其横跨两列
# 创建一个滚动文本框，用于显示MT5生成的摘要
output_text_widget_mt5 = scrolledtext.ScrolledText(output_label_frame_mt5, wrap=tk.WORD, width=50, height=10)

output_text_widget_mt5.pack(pady=10, fill='both', expand=True)

# 设置行列权重，使得在窗口变大时，文本框和标签框都能够扩展
for i in range(4):  # 设置所有行的权重为1
    root.grid_rowconfigure(i, weight=1)
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)

# 运行主循环
root.mainloop()