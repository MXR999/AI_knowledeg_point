import random

rules = """    
    复合句子 = 句子|连词 |句子
    连词 = 而且 | 但是 | 不过
    句子 = 主语 谓语 宾语
    主语 = 你| 我 | 他|他们
    谓语 = 吃| 玩
    宾语 = 桃子| 皮球 |篮球
    """
grammer = {}
for r in rules.split("\n"):
    if r.strip() == "":
        continue
    target, expand = r.split("=")
    # print(target, type(target))
    # expand = expand.split("|")
    grammer[target.strip()] = [e.strip() for e in expand.split("|")]


# print(grammer)


def generate(grammer, target="句子"):
    if target not in grammer:
        return target
    return "".join([generate(grammer, t) for t in
        random.choice(grammer[target]).split()])


for _ in range(10):
    print(generate(grammer))

