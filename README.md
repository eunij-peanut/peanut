# 项目介绍

# 环境依赖
    summarization_total_mt5smalltrainself.py  缺什么库下什么什么库
    Python 3.8.20 （其他的版本可能也行）
# 目录结构描述
    ├── ReadMe.md           // 帮助文档
    
    ├── Chinese-automatic-text-summarization-main   // 核心文件库，包含各个模型和集成的初步UI界面
    
    │   ├── mt5-small    // 包含 mt5-small 预训练模型 ，用于训练新闻文本摘要功能
    
    │   ├── mt5-small-mymodel    // 包含训练好的 mt5-small 模型，实现新闻文本摘要功能

    │   ├── t5_pegasus_pretain   // 包含 mt5-small 预训练模型 ，用于训练对话摘要功能

    │   └──  saved_model_12.29   // 包含训练好的 mt5-small 模型，实现对话摘要功能 
    
    └── summarization_total_mt5smalltrainself.py          // 集成的代码，初步UI实现，调用各个训练好的模型
 
![目录组织结构](https://i-blog.csdnimg.cn/direct/9e5838afae434df8bf167a06a63cc90a.png#pic_center)

# 使用说明
    通过网盘分享的文件：mt5-small 模型
    链接: https://pan.baidu.com/s/1E_h3RH51k5SwiaHFwKqrVg?pwd=ikkj 提取码: ikkj 
    下载网盘链接的模型文件，按照以上目录结构组织模型
    
    运行 summarization_total_mt5smalltrainself.py，注意各个对应的训练好的模型和预训练模型的代码中的目录是否填写正确
# 版本内容更新
 
 
