import re

def load_and_filter_stopwords(file_path: str) -> set:
    stopwords = set()
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            word = line.strip()
            # 只保留中文词语
            if re.match(r'^[\u4e00-\u9fa5]+$', word):
                stopwords.add(word)
    return stopwords

def write_stopwords(stopwords: set, output_path: str) -> None:
    with open(output_path, 'w', encoding='utf-8') as file:
        for word in sorted(stopwords):
            file.write(word + '\n')

# 示例用法
stop_file_path = './stopwords.txt'
output_file_path = './filtered_stopwords.txt'

stopwords = load_and_filter_stopwords(stop_file_path)
write_stopwords(stopwords, output_file_path)

print(f"Filtered stopwords have been written to {output_file_path}")