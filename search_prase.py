# 输入语法 (op exp exp ...) 类lisp语法
# op: and|or|not
# id: 查询关键字
# 语法解析器:将()转换为Exp对象
from __future__ import annotations
from enum import Enum
from dataclasses import dataclass

class Operator(Enum):
    AND = "and"
    OR = "or"
    NOT = "not"

@dataclass
class Exp:
    op: Operator
    exp: list[Exp | str]

    def __repr__(self) -> str:
        return f"({self.op.value} {' '.join(map(repr, self.exp))})"

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

# 


# 优化查询条件
class ExpOptimizer:
    @staticmethod
    def flatten(exp: Exp) -> Exp:
        """(and a (and b c)) -> (and a b c) 便于后续and条件的合并"""
        if isinstance(exp, str):
            return exp
        flattened_exp = []
        for subexp in exp.exp:
            subexp = ExpOptimizer.flatten(subexp)
            if isinstance(subexp, Exp) and subexp.op == exp.op:
                flattened_exp.extend(subexp.exp)
            else:
                flattened_exp.append(subexp)
        return Exp(exp.op, flattened_exp)

print(process("(or(and a b)(and c d))"))

# 查询优化复杂条件的化简


# 查询优化一，优先and长度最短的条件
# 查询优化二，转换关于条件的bool表达式，尽可能减少重复计算，尽可能对非相关条件进行and操作