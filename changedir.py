train_root='../data/ocr/ch_test.txt'
new_str='/mnt/disk2/std2021/hejiabang-data/OCR/attention_img/AttentionData'
old_str='../data/ocr/AttentionData'

def alter(file, old_str, new_str):
    """
    替换文件中的字符串
    :param file:文件名
    :param old_str:旧的字符串
    :param new_str:新字符串
    :return:
    """
    file_data = ""
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            if old_str in line:
                line = line.replace(old_str, new_str)
            file_data += line
    with open(file, "w", encoding="utf-8") as f:
        f.write(file_data)

alter(train_root, old_str, new_str)
