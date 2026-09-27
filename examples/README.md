# 最小使用

安装后，从任意工作目录执行：

```sh
hrl-v2 --workspace ./experiment check
hrl-v2 --workspace ./experiment dry-run
hrl-v2 --workspace ./experiment validate
```

前两项不进行物理采样或优化器更新。第三项执行有界测试。完整600轮训练需显式 `hrl-v2 --workspace ./experiment train --seed 4101`；示例不会自动执行它。
