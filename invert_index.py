from heapq import heappush, heappop, heapify
from collections import UserList
# 测试模块
import time
from functools import wraps

# 全局字典，用于存储每个函数的累计执行时间
execution_times = {}#type:ignore

def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # 更新全局字典中的累计执行时间
        if func.__name__ in execution_times:
            execution_times[func.__name__] += elapsed_time
        else:
            execution_times[func.__name__] = elapsed_time
        
        return result
    return wrapper


class OrderedInvertedIndex(UserList):
    def __init__(self, data=[]):
        super().__init__(data)  # 确保数据有序

    @timing_decorator
    @staticmethod
    def and_op(one: 'OrderedInvertedIndex', other: 'OrderedInvertedIndex') -> 'OrderedInvertedIndex':
        # (and one other)，两个排序列表 small -> large
        result = []
        i, j = 0, 0
        try:
            while True:
                if one[i] == other[j]:
                    result.append(one[i])
                    i += 1
                    j += 1
                elif one[i] < other[j]:
                    i += 1
                else:
                    j += 1
        except IndexError:
            pass
        return OrderedInvertedIndex(result)
    
    @staticmethod
    def sub_op(one: 'OrderedInvertedIndex', other: 'OrderedInvertedIndex') -> 'OrderedInvertedIndex':
        # one-other 仅被and_not_op调用(此时one>other)
        result = []
        i, j = 0, 0
        while j < len(other):
            if one[i] == other[j]:
                i += 1
                j += 1
            else:
                result.append(one[i])
                i += 1
        # 添加剩余的元素
        while i < len(one):
            result.append(one[i])
            i += 1
        return OrderedInvertedIndex(result)

    @staticmethod
    def and_not_op(one: 'OrderedInvertedIndex', other: 'OrderedInvertedIndex') -> 'OrderedInvertedIndex':
        # (and one (not other)) one-other
        return OrderedInvertedIndex.sub_op(one,OrderedInvertedIndex.quick_and(one, other))

    @timing_decorator
    @staticmethod
    def quick_and(one, other):
        # (and one other)，两个排序列表 small -> large
        # 以索引的大幅度增长来模拟跳表
        # 对于one的索引我们有（other对称）
        # 快速增长：每次将跳转长度增长一倍，直到找到一个大于等于other的值
        # 事件一：找到一个大于other的值确定了区间，二分法找到other的值，给索引赋值,跳转长度减半
        # 事件二：等于other的值，将结果放入result，跳转长度减半
        # 事件三：超出one的范围，跳转长度调整为one的长度减去上一个索引
        result = []
        i, j = 0, 0
        i_jump , j_jump = 1, 1
        def half_jump(i_jump):
            return i_jump//2 if i_jump>1 else 1
        try:#采取try-except结构,减少if-else的判断
            while True:
                if one[i] == other[j]:
                    # 事件二：找到相等的值，加入结果
                    result.append(one[i])
                    i += 1
                    j += 1
                    i_jump = half_jump(i_jump)
                    j_jump = half_jump(j_jump)

                elif one[i] < other[j]:
                    # 对 one 的索引实现跳跃式查找
                    try:#快启动跳跃
                        while one[i + i_jump] < other[j]:
                            i=i+i_jump
                            i_jump *= 2  # 快速增长跳跃步长
                        high = i + i_jump
                        i_jump = half_jump(i_jump) #惩罚跳跃步长
                    except IndexError:#跳出了one的范围
                        if one[-1]==other[j]:
                            result.append(one[-1])
                            break
                        i_jump = 1#惩罚跳跃步长
                        high = len(one) - 1
                    i = OrderedInvertedIndex.binary_search(one, other[j], i, high)
                else:
                    #  对 other 的索引实现跳跃式查找
                    try:#快启动跳跃
                        while other[j + j_jump] < one[i]:
                            j=j+j_jump
                            j_jump *= 2  # 快速增长跳跃步长
                        high = j + j_jump
                        j_jump = half_jump(j_jump) #惩罚跳跃步长
                    except IndexError:#跳出了one的范围
                        if other[-1]==one[i]:
                            result.append(other[-1])
                            break
                        j_jump = 1#惩罚跳跃步长
                        high = len(other) - 1
                    j = OrderedInvertedIndex.binary_search(other, one[i], j, high)

        except IndexError:
            pass
        return OrderedInvertedIndex(result)

    @staticmethod
    def binary_search(arr, target, low, high):
        # 标准二分查找，在确定区间中找到第一个 >= target 的位置
        while low < high:
            mid = (low + high) // 2
            if arr[mid] < target:
                low = mid + 1
            else:
                high = mid
        return low if low < len(arr) and arr[low] >= target else low + 1
 
    @staticmethod
    def or_op(*lists: 'OrderedInvertedIndex') -> 'OrderedInvertedIndex':
        #  ndlogd
        # 使用最小堆合并多个有序列表
        min_heap:list[tuple[int,int]] = []
        iterators = [iter(lst) for lst in lists]
        result = []

        # 初始化堆，将每个列表的第一个元素放入堆中
        for idx, it in enumerate(iterators):
            first_item = next(it, None)
            if first_item is not None:
                heappush(min_heap, (first_item, idx))

        last_added = None  # 跟踪上一个加入结果的值，避免重复添加
        while min_heap:
            smallest, idx = heappop(min_heap)
            if smallest != last_added:  # 避免重复
                result.append(smallest)
                last_added = smallest

            # 从对应列表中获取下一个元素并加入堆
            next_item = next(iterators[idx], None)
            if next_item is not None:
                heappush(min_heap, (next_item, idx))

        return OrderedInvertedIndex(result)
    