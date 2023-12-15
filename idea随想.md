- 对于SASRec，其每个user对应的item序列，每一个item生成一个对应embedding，这个embedding来源于语言模型对其feature等的编码（参考Text Is All You Need，或许可以构成类似字典的东西？但是这么结构化的东西或许直接转化成自然语言为好，但是考虑到最后接的是BERT，最好能够用某种手段将LLM增强的语句转换成词？或者专门针对token调一下PLM）
- Once修改了LLaMA的最后一层，让它改为一个Linear+Attention层，从而输出LLaMA的最后一层所有token（去除[PAD]）汇聚的编码
- 修改一下ml的处理，把所有负节点加上

