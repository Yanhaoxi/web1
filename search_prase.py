# 输入语法 (op exp exp ...) 类lisp语法
# op: and|or|not
# id: 查询关键字
# 语法解析器:将()转换为Exp对象
from __future__ import annotations
from enum import Enum
from dataclasses import dataclass
import logging
import functools
import io
from contextlib import redirect_stdout

# 设置日志配置
logging.basicConfig(filename="project.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", encoding="utf-8")

def log_output(level=logging.INFO):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            log_stream = io.StringIO()  # 用于临时捕获输出
            with redirect_stdout(log_stream):  # 重定向 print 输出
                result = func(*args, **kwargs)
            # 根据指定的日志等级记录日志
            message = log_stream.getvalue()
            if level == logging.DEBUG:
                logging.debug(message)
            elif level == logging.INFO:
                logging.info(message)
            elif level == logging.WARNING:
                logging.warning(message)
            elif level == logging.ERROR:
                logging.error(message)
            elif level == logging.CRITICAL:
                logging.critical(message)
            return result
        return wrapper
    return decorator


class Operator(Enum):
    AND = "and"
    OR = "or"
    NOT = "not"

@dataclass
class Exp:
    op: Operator
    exp: list[Exp | str]
    good:bool=False
    tag:int=0

    def __repr__(self) -> str:
        return f"({self.op.value} {' '.join(map(repr, self.exp))})"
    
    def __hash__(self) -> int:
        return hash((self.op, tuple(self.exp)))

# 用户查询条件的解析器
def tokenize(s: str) -> list[str]:
    "Convert a string into a list of tokens."
    return s.replace('(', ' ( ').replace(')', ' ) ').split()

def parse(tokens: list[str]) -> Exp | str:
    if not tokens:
        raise ValueError("Unexpected EOF")
    token = tokens.pop(0)
    if token == '(':
        sublist = []
        while tokens[0] != ')':
            sublist.append(parse(tokens))
        tokens.pop(0)  # pop off ')'
        op_token = sublist.pop(0)
        # check if the first element is operator
        if isinstance(op_token, str):
            op_token = op_token.upper()
        else:
            raise ValueError("Operator must be the first element in the list")
        if op_token not in Operator.__members__:
            raise ValueError(f"Invalid operator: {op_token},only support {Operator.__members__.keys()}")

        op = Operator[op_token]
        # check if the number of arguments is correct
        if op==Operator.NOT:
            if len(sublist)!=1:
                raise ValueError("NOT operator must have exactly one argument")
        elif len(sublist)<2:
            raise ValueError(f"{op.value} operator must have at least two arguments")
        return Exp(op, sublist)
    elif token == ')':
        raise ValueError("Unexpected )")
    else:
        return token
    
def process(s: str) -> Exp | str:
    """用户输入的查询条件转为Exp对象"""
    return parse(tokenize(s))

class counter:
    def __init__(self) -> None:
        self.count=0
        self.cache:dict[int|str,OrderedInvertedIndex]={}
    def __call__(self):
        self.count+=1
        return self.count
    
    
exp_count=counter()

class ExpOptimizer:
    @staticmethod
    def not_down(exp: Exp|str) -> Exp|str:
        if isinstance(exp, str):
            return exp
        if exp.op == Operator.NOT:
            subexp = exp.exp[0]
            if isinstance(subexp, str):
                return exp
            if subexp.op == Operator.NOT:
                return ExpOptimizer.not_down(subexp.exp[0])
            elif subexp.op == Operator.AND:
                return Exp(Operator.OR, [ExpOptimizer.not_down(Exp(Operator.NOT, [subsubexp])) for subsubexp in subexp.exp])
            elif subexp.op == Operator.OR:
                return Exp(Operator.AND, [ExpOptimizer.not_down(Exp(Operator.NOT, [subsubexp])) for subsubexp in subexp.exp])
        if exp.op == Operator.AND:
            return Exp(Operator.AND, [ExpOptimizer.not_down(subexp) for subexp in exp.exp])
        if exp.op == Operator.OR:
            return Exp(Operator.OR, [ExpOptimizer.not_down(subexp) for subexp in exp.exp])

    @staticmethod
    def flatten(exp: Exp|str) -> Exp|str:
        if isinstance(exp, str):
            return exp
        if exp.op == Operator.AND or exp.op == Operator.OR:
            flattened_exp:list[Exp|str] = []
            for subexp in exp.exp:
                subexp = ExpOptimizer.flatten(subexp)
                if isinstance(subexp, Exp) and subexp.op == exp.op:
                    flattened_exp.extend(subexp.exp)
                else:
                    flattened_exp.append(subexp)
            return Exp(exp.op, flattened_exp)
        return exp
    
    @staticmethod
    def _and_on_not(exp: Exp|str) -> bool:
        if isinstance(exp, str):
            return True
        if exp.op == Operator.AND:
            exp.good=any(ExpOptimizer._and_on_not(_) for _ in exp.exp)
            return exp.good
        if exp.op == Operator.OR:
            exp.good=all(ExpOptimizer._and_on_not(_) for _ in exp.exp)
            return exp.good
        if exp.op == Operator.NOT:
            exp.good=False
            return False
    
    @staticmethod
    def refuse_search(exp: Exp|str) -> None:
        ExpOptimizer._and_on_not(exp)
        if isinstance(exp, Exp):
            if exp.good==False:
                raise ValueError("Invalid query: 范围过大请调整查询语句") 
            

    @staticmethod
    def and_on_not(exp: Exp|str) -> Exp|str:
        # and -> or -> not => or -> and -> not
        if isinstance(exp, str):
            return exp
        if exp.op == Operator.AND:
            good:list[Exp|str]=[]
            bad:list[Exp]=[]
            for subexp in exp.exp:
                if isinstance(subexp, str):
                    good.append(subexp)
                elif subexp.good:
                    good.append(ExpOptimizer.and_on_not(subexp))
                else:
                    bad.append(subexp)
            if bad:
                # 把好的集结并存入缓存,good_now是可计算的
                if len(good)==1:
                    good_now=good[0]
                    if isinstance(good_now, str):
                        pass
                    else:
                        good_now.tag=exp_count()
                else:
                    good_now = Exp(Operator.AND,good,True,exp_count()) 
                # (and (or (not a) b) c)=>(or (and (not a) c) (and b c))
                while bad:
                    bad_now=bad.pop()
                    if bad_now.op==Operator.OR:
                        good_now=Exp(Operator.OR,[ExpOptimizer.and_on_not(Exp(Operator.AND,[good_now,subexp],True)) for subexp in bad_now.exp],True,exp_count())
                    if bad_now.op==Operator.AND:
                        tmp=Exp(Operator.AND,[_ for _ in bad_now.exp]+[good_now],True,exp_count())
                        good_now=ExpOptimizer.and_on_not(tmp)#type:ignore
                    if bad_now.op==Operator.NOT:
                        good_now=Exp(Operator.AND,[good_now,bad_now],True,exp_count())
                return good_now
            else:
                return Exp(Operator.AND,good,True)
        if exp.op == Operator.OR:
            return Exp(Operator.OR, [ExpOptimizer.and_on_not(subexp) for subexp in exp.exp],True)
        return exp
                    
# #test
# print(ExpOptimizer.flatten(ExpOptimizer.not_down(process("(or (not (and a (and b c))) (or e f))"))))
# # (or (not 'a') (not 'b') (not 'c') 'e' 'f')
def optimize_expression(query: str) -> Exp|str:
    exp = process(query)
    exp = ExpOptimizer.not_down(exp)
    exp = ExpOptimizer.flatten(exp)
    ExpOptimizer.refuse_search(exp)
    return ExpOptimizer.and_on_not(exp)

# try:
#     print(optimize_expression("(or (not (and a (and b c))) (or e f))"))
# except ValueError as e:
#     print(e)

# print(optimize_expression("(and (or (not a) (and (not d) (not e))) c)"))

# # (or 
# #   (and 
# #       'c' 
# #       (not 'a')) 
# #   (and 
# #       (and 
# #           'c' 
# #           (not 'e')) 
# #       (not 'd')))

import  struct
def unzip_token2id_map(dict_file_path: str) -> dict[tuple[str, str], tuple[int, int]]:
    result = {}
    with open(dict_file_path, 'rb') as index_file:
        index_data = index_file.read()
        offset = 0
        length = len(index_data)
        while offset < length:
            # Read the key length
            key_length = struct.unpack_from('B', index_data, offset)[0]
            offset += 1
            
            # Read the key bytes
            key_bytes = index_data[offset:offset + key_length]
            key_str = key_bytes.decode('utf-8')
            key:tuple[str,str] = tuple(key_str.split(':')[0:2])#type:ignore
            offset += key_length
            
            # Read the offset and length
            data_offset:int = struct.unpack_from('Q', index_data, offset)[0]
            offset += 8
            data_length:int = struct.unpack_from('I', index_data, offset)[0]
            offset += 4
            
            # Store the offset and length in the result dictionary
            result[key] = (data_offset, data_length)
    return result

from process_data import dict_file_path,id_file_path
token2id_map=unzip_token2id_map(dict_file_path)

def decompress_from_file(file_path: str, start: int, length: int) -> list[int]:
    with open(file_path, 'rb') as data_file:
        data_file.seek(start)
        compressed_data = data_file.read(length)
        
        decompressed_data = []
        number = 0
        shift = 0
        for byte in compressed_data:
            if byte & 0x80:
                number |= (byte & 0x7F) << shift
                shift += 7
            else:
                number |= byte << shift
                decompressed_data.append(number)
                number = 0
                shift = 0
        return decompressed_data
    
# 加载同义词词典
synonyms_file_path = 'dict_similar.txt'
def load_synonyms(synonyms_file_path: str) -> dict[str, str]:
    synonyms = {}
    with open(synonyms_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            words = line.strip().split()
            for word in words[1:]:
                synonyms[word] = words[0]
    return synonyms

synonyms_dict=load_synonyms(synonyms_file_path)

def similar_evaluator(token1: str, token2: str, synonyms: dict[str, str]) -> int:
    if value1:=synonyms.get(token1,token1):
        if value2:=synonyms.get(token2,token2):
            min_length = min(len(value1), len(value2))
            for i in range(min_length):
                if value1[i] != value2[i]:
                    return i
            return min_length
    return 0

def similar_graph_gen(token_list: list[str], synonyms: dict[str, str]) -> list[list[tuple[str, str]]]:
    result:list[list[tuple[str, str]]] = [[] for _ in range(9)]
    for i in range(len(token_list)):
        for j in range(i + 1, len(token_list)):
            token1 = token_list[i]
            token2 = token_list[j]
            similarity = similar_evaluator(token1, token2, synonyms)
            result[similarity].append((token1, token2))
    return result


# 如果使用了and or not 默认用户是高级用户，不需要再做分词
# 对于普通用户需要对查询的长句子进行进一步的分词处理
from process_data import process_text, tokenize_tags, is_stopword, allowed_pos,stopwords,jiebaModel
from invert_index import OrderedInvertedIndex

@log_output(level=logging.INFO)
def find_token(token: tuple[str,str]) -> OrderedInvertedIndex|None:
    if token in token2id_map:
        print(f"找到{token}相关词条")
        position=token2id_map[token]
        # 实时加载
        if isinstance(position, tuple):
            token2id_map[token] = OrderedInvertedIndex(sorted(decompress_from_file(id_file_path, position[0], position[1]))) #type:ignore
            position = token2id_map[token]
        print(f"词条相关:{position}")
        return position #type:ignore
    else:
        print(f"未找到{token}相关词条")
        return None

def search_and(tokens: list[tuple[str,str]]) -> OrderedInvertedIndex:
    result:dict[str,OrderedInvertedIndex]={}
    for token in tokens:
            if position:=find_token(token):
                result.update({token[0]:position})#type:ignore
    # 将结果按照做合适的顺序作合并
    # 把token先按照相似性分级
    # 从相似性最低且和最小的开始合并
    return token_and(result)

@log_output(level=logging.INFO)
def token_and(result: dict[str, OrderedInvertedIndex]) -> OrderedInvertedIndex:
    token_list = [token for token in result.keys()]
    similar_graph = similar_graph_gen(token_list, synonyms_dict)
    for i in range(9):
        if similar_graph[i] is not []:
            # 按照长度排序
            sorted_list = sorted(similar_graph[i], key=lambda x: len(result[x[0]]) + len(result[x[1]]))
            # 从最小token对的开始合并，合并后两个token指向同一倒排序表
            for token1, token2 in sorted_list:
                if result[token1] is not result[token2]:
                    result[token1] = OrderedInvertedIndex.quick_and(result[token1], result[token2])
                    result[token2] = result[token1]
    print(f"(and {' '.join(result.keys())}) 查询语句结果: {result[token_list[0]]}")
    return result[token_list[0]]


# 对朴素句子的查询
def process_query(input: str) -> OrderedInvertedIndex:
    exp=input
    tokens=list(tokenize_tags(exp,jiebaModel()))
    tokens = [(processed_token,_) for token, _ in tokens if (processed_token := process_text(token))]
    tokens = list(set(tokens))
    tokens = [token for token in tokens if not is_stopword(token[0], stopwords)]
    tokens = [token for token in tokens if len(token[0])>1 or token[1] in allowed_pos]
    if not tokens:
        return OrderedInvertedIndex()
    return search_and(tokens)

# 对查询条件的计算
@log_output(level=logging.INFO)
def evaluate_query(query: str,ready:OrderedInvertedIndex) -> OrderedInvertedIndex:
    exp = optimize_expression(query)
    if isinstance(exp, str):
        return OrderedInvertedIndex.quick_and(ready,_evaluate_query(exp))
    if ready == OrderedInvertedIndex():
        if exp.good:
            return _evaluate_query(exp)
        else:
            raise ValueError("Invalid query: 范围过大请调整查询语句")
    else:
        if exp.good:
            return OrderedInvertedIndex.quick_and(ready,_evaluate_query(exp))
        else:
            # ready用占位符替代
            exp=Exp(Operator.AND,[exp,'*'],True)
            exp=ExpOptimizer.and_on_not(exp)
            print(f"转换后的{exp}")
            return _evaluate_query(exp) #type:ignore
        
from process_data import all_pos
from contextlib import redirect_stdout
import io

@log_output(level=logging.INFO)
def _evaluate_query(query: Exp|str) -> OrderedInvertedIndex:
    if isinstance(query, str):
        if query in exp_count.cache:
            return exp_count.cache[query]
        else:
            now=OrderedInvertedIndex()
            for pos in all_pos:
                if position:=find_token((query,pos)):
                    now=OrderedInvertedIndex.or_op(now,position)
            exp_count.cache[query]=now
            print(f"{query} 查询语句结果:{now}")
            return now
    if query.tag!=0:
        # 尝试从缓存中读取
        if exp_count.cache.get(query.tag):
            return exp_count.cache[query.tag]

    if query.op == Operator.AND:
        good_exp=[]
        good_str=[]
        bad=[]
        for subexp in query.exp:
            if isinstance(subexp, str):
                good_str.append(subexp)
            elif subexp.good:
                good_exp.append(subexp) 
            else:
                bad.append(subexp)
            # check
        if bad==[]:
            now=token_and({token:_evaluate_query(token) for token in good_str})
            for exp in good_exp:
                now=OrderedInvertedIndex.quick_and(now,_evaluate_query(exp))
        else:
            assert len(bad)==1 and len(good_str)+len(good_exp)==1 and bad[0].op==Operator.NOT
            if good_str!=[]:
                now=_evaluate_query(good_str[0])
            else:
                now=_evaluate_query(good_exp[0])
            now=OrderedInvertedIndex.and_not_op(now,_evaluate_query(bad[0].exp[0]))
        print(f"{query} 查询语句结果:{now}")
        if query.tag!=0:
            exp_count.cache[query.tag]=now
        return now
        
    elif query.op == Operator.OR:
        now=OrderedInvertedIndex.or_op(*[_evaluate_query(exp) for exp in query.exp])
        print(f"{query} 查询语句结果:{now}")
        if query.tag!=0:
            exp_count.cache[query.tag]=now
        return now
    else :
        raise ValueError("Invalid query: 请检查查询语句")

# 允许附加查询条件
# 句子 (查询语句) @ (附加查询条件)
if __name__=='__main__':
    get_input=input("请输入查询语句:")
    # 战争纪录 @ 英国 => input_str=战争纪录,query=英国
    if '@' in get_input:
        input_str,query=get_input.split('@')
        input_str=input_str.strip()
        query=query.strip()
        print(f"查询语句:{input_str},附加查询条件:{query}")
        print(f"附加条件后最终结果:{evaluate_query(query,process_query(input_str))}")
    else:
        input_str=get_input
        print(f"查询语句:{input_str}")
        process_query(input_str)

    # 测试
    # exp=optimize_expression("(and 生活 早晨 美好)")
    # for i in range(99):
    #     _evaluate_query(exp)
    #     exp_count.cache.clear()
    # from invert_index import execution_times
    # print(_evaluate_query(exp))
    # print(execution_times)


# "(and 花 婚姻 感情 生活)" 查询时间:{'quick_and': 0.005951881408691406} 查询时间:{'and_op': 0.026388168334960938}
# "(and 记录 大学 爱情)" 查询时间:{'quick_and': 0.020933151245117188}  查询时间:{'and_op': 0.03333687782287598}
# "(and 生活 早晨 美好)" 查询时间:{'quick_and': 0.0011165142059326172} 查询时间:{'and_op': 0.0015380382537841797}
    

# test
# print(process("(or(and a b)(and c d))"))
# print(process("OK"))
# print(ExpOptimizer.flatten(process("(or(and a (and b c))(or e f))")))
# print(ExpOptimizer.refuse_search(process("(and (not a) b)")))
# print(process_query("战争纪录片"))

# 生活 @(and 花 (or (not 婚姻) (not 感情)))