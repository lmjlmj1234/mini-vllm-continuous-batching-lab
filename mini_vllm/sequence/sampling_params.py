from dataclasses import dataclass, field
from typing import List #用于类型提示（Type Hinting）


@dataclass #加上 @dataclass 后，Python 解释器会自动帮你生成 __init__、__repr__、__eq__ 等方法。
class SamplingParams:
    max_tokens: int = 16 #生成最大的token数，达到这个长度，生成会自动停止
    temperature: float = 1.0 #采样温度，控制随机性，值越大输出越随机，越低越重复
    top_p: float = 1.0 #核采样，每一步只从累计概率不超过top_p 的最小 token 集合中采样。1表示关闭top-p，值越小采样范围越集中，质量通常更高。
    top_k: int = -1#每一步只从概率最高的 top_k 个 token 中采样。top_k=-1 表示关闭 top-k 过滤（考虑所有 token）。设为正数（如 40、50）则只从概率最高的 K 个 token 中采样。
    stop_token_ids: List[int] = field(default_factory=list) #遇到这些 token ID 时停止生成。用 field(default_factory=list) 而非直接将默认值设为 []，让实例维护自己的 list 对象。
    stop_strings: List[str] = field(default_factory=list)#比如设 stop_strings=["\n\n"] 表示遇到两个换行就停止。同样是可变类型，使用 field(default_factory=list)。
    #注意：
    # 如果在普通
    # Class的    # __init__ 里写 = []，Python 每次都会新建一个列表；
    # 但如果你直接在 @ dataclass 的类属性里写 = []，Python 就会把这一个列表当成全局共享的，导致所有对象互相污染！