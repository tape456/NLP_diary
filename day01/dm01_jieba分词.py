"""
案例:
    演示 jieba 分词库的应用.

分词的相关介绍:
    概述:
        分词过程= 找到 分界符 的过程,找到分界符就可以分词了
        每个分词结果 = 1个Token
    常见的分词包:
        jieba:  精确模式,全模式,搜索引擎模式,繁体字符,用户自定义字典...
        IK分词器: ElasticSearch搜索引擎用的多
        SnowNLP:基于概率算法的中文自然语言处理工具包
        pyltp: 哈工大的
        THULAC:清华的...

    jieba 的作用介绍:
        A. 支持多种分词模式
        B. 支持中文繁体分词
        C. 支持用户自定义词典
        总结:
            上述的分词模式,其实就是 分词粒度不同. 例如:"软件工程" ->  软件工程(粗粒度) ,软件,工程(细粒度)

额外做的事：
    为了不和前面的环境冲突，建议大家创建1个新的沙箱使用

细节：
    jieba 属于第三方的库，用之前需要先安装一下，即：pip install jieba
"""
# 导包
import jieba

# todo 1.定义函数，演示jieba精确分词模式，适用于：文本分析
def dm01():
    # 1.定义待分词的文本内容
    content='传智教育是一家上市公司，旗下有黑马程序员品牌。我是在黑马这里学习人工智能'

    # 2.使用jieba进行 精确模式分词(默认模式) -> cut_all=False
    # result1:生成器对象,好处：节省内存，只能遍历一次
    result1=jieba.cut(content,cut_all=False)
    print(f'result:{result1}')

    # 3.从生成器对象中,获取所有的分词结果.
    # 3.1 思路1:next()函数,逐个获取下个元素
    print(next(result1))
    print(next(result1))
    print(' -.- '*10)

    # 3.2 思路2:遍历方式,从生成器中获取元素
    for item in result1:
        print(item)
    print(' =.= '*10)

    # 4.如果要列表怎么办? 即:[词1,词2,词3...]
    # 思路1:直接把上述的生成器 -> 转成 列表
    list1=list(result1)
    print(f'list1:{list1}')

    # 思路2:切词时,直接返回list,相当于:语法糖.
    list2=jieba.lcut(content,cut_all=False)
    print(f'list2:{list2}')


# todo 6.测试代码
if __name__ == '__main__':
    import jieba
    print(jieba.__file__)
    dm01()


