# .csv->list[tuple[movie_id:int, tag:list[str]]]
# 允许部分读入，方便完全处理前的结果评估
import csv
csv_file_path = 'selected_book_top_1200_data_tag.csv'

def read_csv(file_path: str, num_lines=1) -> list[tuple[int, list[str]]]:
    """ 
    ID1,"{'a1', 'b1'}"
    ID2,"{'a2', 'b2'}"
    -> 
    [(ID1, ['a1', 'b1'])
    ,(ID2, ['a2', 'b2'])] 
    """
    data = []
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        # skip header
        next(reader)
        # read data
        for i, row in enumerate(reader):
            if i >= num_lines:
                break
            movie_id = int(row[0])
            # Remove curly braces and split by comma
            tags = row[1].strip('{}').split(',')
            # Strip whitespace and quotes from each tag
            tags = [tag.strip(" '\"") for tag in tags]
            data.append((movie_id, tags))
    return data


# 再处理数据tag

# 分词处理
from typing import Protocol
# 定义一个分词模型的接口
class TokenModel(Protocol):
    
    def __init__(self) -> None:
        """
        启动分词模型的方法
        """
        pass
    
    # 采用词性标注的分词
    def tokenizer(self, text: str) -> list[tuple[str, str]]:
        """
        对输入文本进行分词的方法
        :param text: 输入的文本
        :return: 分词后的列表
        """
        pass


class jiebaModel(TokenModel):
    def __init__(self):
        import jieba
        import jieba.posseg 
        self.model=jieba
        self.token=jieba.posseg.cut

    def tokenizer(self, text: str) -> list[tuple[str, str]]:    
        return list(self.token(text))


class pkusegModel(TokenModel):
    
    def __init__(self):
        import pkuseg
        self.model=pkuseg.pkuseg(postag=True)
        self.token=self.model.cut
    
    def tokenizer(self, text: str) -> list[tuple[str, str]]:
        return self.token(text)


# tag:str->分词并且词性标注
def tokenize_tags(tag: str, model: TokenModel) -> list[tuple[str,str]]:
    return [token for token in model.tokenizer(tag)]

stop_file_path = 'stopwords.txt'

# 停用词表过滤器
def load_stopwords(filepath: str) -> set[str]:
    with open(filepath, 'r', encoding='utf-8') as file:
        stopwords = set(line.strip() for line in file)
    return stopwords

stopwords = load_stopwords(stop_file_path)

def is_stopword(token: str, stopwords: set[str]) -> bool:
    return token in stopwords

# 中英文,简体繁体处理
# 由于api转换速度慢放弃对英文的翻译
import re
# from translate import Translator
import opencc
converter = opencc.OpenCC('t2s.json')
# translator = Translator(to_lang="zh")

def process_text(text: str) -> str:
    # 移除非中文字符
    text = re.sub(r'[^\u4e00-\u9fff]', '', text)
    # 翻译成中文
    # text = translator.translate(text)
    # 转换为简体中文
    text = converter.convert(text)
    return text

all_pos=jieba_pos_tags = {
    # 名词（n）
    "n", "nr", "nr1", "nr2", "nrj", "nrf", "ns", "nsf", "nt", "nz", "nl", "ng",
    # 时间词（t）
    "t", "tg",
    # 处所词（s）
    "s",
    # 方位词（f）
    "f",
    # 动词（v）
    "v", "vd", "vn", "vshi", "vyou", "vf", "vx", "vi", "vl", "vg",
    # 形容词（a）
    "a", "ad", "an", "ag", "al",
    # 区别词（b）
    "b",
    # 状态词（z）
    "z",
    # 代词（r）
    "r", "rr", "rz", "rzt", "rzs", "rzv", "ry", "ryt", "rys", "ryv", "rg",
    # 数词（m）
    "m", "mq",
    # 量词（q）
    "q",
    # 副词（d）
    "d",
    # 介词（p）
    "p", "pba", "pbei",
    # 连词（c）
    "c",
    # 助词（u）
    "u", "uzhe", "ule", "uguo", "ude1", "ude2", "ude3", "usuo", "udeng", "uyy", 
    "udh", "uls", "uzhi", "ulian",
    # 叹词（e）
    "e",
    # 语气词（y）
    "y",
    # 拟声词（o）
    "o",
    # 前缀（h）
    "h",
    # 后缀（k）
    "k",
    # 字符串（x）
    "x", "xx",
    # 标点符号（w）
    "wkz", "wky", "wyz", "wyy", "wj", "ww", "wt", "wd", "wf", "wn", "wm", "ws", "wp"
}

# 单字允许的词性
allowed_pos = {
    'a','ag','an','g','i','j','n','ng','nr','ns','nt','nz','s','tg','t','v','vd','vg','vn','un'
}
# 没有找到合适的同义词库，暂时不做同义词合并
# # 同义词合并
# synonyms_file_path = 'synonym.txt'
# def load_synonyms(filepath: str) -> dict[str, str]:
#     synonyms = {}
#     with open(filepath, 'r', encoding='utf-8') as file:
#         for line in file:
#             words = line.strip().split()
#             for word in words[2:]:
#                 synonyms[word] = words[1]
#     return synonyms
# synonyms = load_synonyms(synonyms_file_path)

# def replace_synonyms(token: str, synonyms: dict[str, str]) -> str:
#     if token in synonyms:
#         print( synonyms[token]+'->'+token)
#     return synonyms.get(token, token)

# 数据压缩

import struct
id_file_path = 'id_list.bin'
dict_file_path = 'token2id_map.bin'
def compress2f(file_path: str, index_file_path: str, map: dict[tuple[str, str], list[int]]) -> None:
    with open(file_path, 'wb') as f:
        # Create a new dictionary to store the file offsets and lengths
        index_data = bytearray()
        for key, value in map.items():
            # Compress the list of integers using variable-length encoding
            compressed_data = bytearray()
            # 8位 最高位指示是否还有下一个字节 如果是1则还有下一个字节
            for number in value:
                while number >= 0x80:
                    compressed_data.append((number & 0x7F) | 0x80)
                    number >>= 7
                compressed_data.append(number & 0x7F)
            
            offset = f.tell()
            f.write(compressed_data)
            length = len(compressed_data)
            
            # Convert the key to a string and encode it to bytes
            key=(key[0].replace(':', ''),key[1])
            key_str = f"{key[0]}:{key[1]}"
            key_bytes = key_str.encode('utf-8')
            key_length = len(key_bytes)
            
            # Pack the key length, key bytes, offset, and length into the index data
            index_data.extend(struct.pack('B', key_length))
            index_data.extend(key_bytes)
            index_data.extend(struct.pack('Q', offset))
            index_data.extend(struct.pack('I', length))
    
    # Save the index data to a binary file
    with open(index_file_path, 'wb') as index_file:
        index_file.write(index_data)
    



# 数据处理
# ->tuple[list[tuple[int, list[tuple[str, str]]]], dict[int, int]]
def process_tags(data: list[tuple[int, list[str]]], model: TokenModel):
    id_map = {}
    token_idlist_map:dict[tuple[str,str],list[int]] = {}
    for i, (movie_id, tags) in enumerate(data):
        # id压缩
        j=0
        if movie_id not in id_map:
            id_map[movie_id] = j
            j=j+1
        # 分词
        tokens:list[tuple[str,str]] = [token for tag in tags for token in tokenize_tags(tag, model)]
        # 去除非中文字符，英文进行翻译后保留
        tokens = [(processed_token,_) for token, _ in tokens if (processed_token := process_text(token))]
        # 相同词合并
        tokens = list(set(tokens))
        # 过滤停用词
        tokens = [token for token in tokens if not is_stopword(token[0], stopwords)]
        # 单字过滤
        tokens = [token for token in tokens if len(token[0])>1 or token[1] in allowed_pos]
        # # 同义词合并
        # tokens = [(replace_synonyms(token[0], synonyms), token[1]) for token in tokens]
        # # 去重
        # tokens = list(set(tokens))
        # 生成token_idlist_map
        for token in tokens:
            if token not in token_idlist_map:
                token_idlist_map[token] = []
            token_idlist_map[token].append(i)
        data[i] = (movie_id, tokens) #type: ignore
        if(i%100==0):
            print(f'已完成：{i}')
    print(f'开始压缩')
    # import pickle
    # with open('data.pkl', 'wb') as f:
    #     pickle.dump(data, f)
    compress2f(id_file_path,dict_file_path,token_idlist_map)
    return data, id_map


if __name__ == '__main__':
    data = read_csv(csv_file_path, 1200)
    process_tags(data, jiebaModel())
