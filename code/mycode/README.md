# 后面可以不需要自己写train.py了， 可以直接跑run.py
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python run.py --config configs/run/bert_tuning.yml --distributed_train True
```
配置文件仿照着configs/run/bert_tuning.yml写 \
如果写了新的dataset或者data collator需要在module/data.py下更改一下get_dataset和get_data_collator
# 如何跑bert模型
```
CUDA_VISIBLE_DEVICES=1 python bert_tuning.py --config configs/bert_tuning.yml
```

可以比较方便不通过修改yml文件就修改配置参数，比如要进行测试的话
```
CUDA_VISIBLE_DEVICES=1 python bert_tuning.py --config configs/bert_tuning.yml --mode test --checkpoint ckpt/mt/bert_2022-11-28_21:55:13/model.bin --test_data_path data/mt/test_in_train --description test_in_train
```
直接在命令行后面跟想要修改的参数就行了
这个是通过chanfig实现，可以看看他的文档https://github.com/ZhiyuanChen/CHANfiG
# 如何添加模型
在module/models.py文件中写好模型，然后将模型映射关系添加进solver.py的model_dict中
在module/data.py中写好数据的输入方式，然后仿照bert_tuning.py写一个训练文件就可以了

# 分布式训练
加上 --distributed_train True
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python bert_tuning.py --config configs/bert_tuning.yml --distributed_train True
```

