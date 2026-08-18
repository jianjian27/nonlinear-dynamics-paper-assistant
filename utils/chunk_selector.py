def select_chunks(
        chunks,
        analysis_type
):
    """
    根据分析任务选择论文片段

    参数:
        chunks:
            文本块列表

        analysis_type:
            分析类型

    返回:
        需要发送给模型的文本
    """


    if analysis_type == "快速阅读":

        selected = chunks[:3]


    elif analysis_type == "数学模型分析":

        selected = chunks[1:4]


    elif analysis_type == "混沌动力学分析":

        selected = chunks[2:5]


    elif analysis_type == "创新点评价":

        selected = chunks[:2] + chunks[-2:]


    else:

        selected = chunks


    return selected