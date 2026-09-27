# 安装与环境

推荐Python 3.11独立环境。Windows和Linux均使用标准pip安装；代码不固定盘符、用户名、Conda名称或解释器绝对路径。参考机器是RTX 5060、CUDA 12.8、Python 3.9.25和PyTorch 2.7 nightly；其他环境不冒充已完成数值验收。

1. 安装适合显卡的NVIDIA驱动。
2. 创建并激活Python虚拟环境。
3. 使用 [PyTorch官方选择器](https://pytorch.org/get-started/locally/) 安装对应CUDA wheel；不要把参考nightly版本当作所有平台通用锁文件。
4. 在项目根目录执行 `python -m pip install .`；开发使用 `python -m pip install -e ".[dev,baselines]"`。
5. `hrl-v2 check` 应能报告CUDA；再执行 `hrl-v2 validate`、`hrl-v2 validate-cuda`。

CPU环境支持安装、帮助、dry-run和 `python -m unittest discover -s tests -v`。当前完整实现套件含CUDA测试，不自动退回CPU。正式 train/resume 入口禁用，需另行确认持续服务研究协议。

无需Git参与运行。wheel包含默认配置、物理代码、参考源码和有界测试。rollout_seconds可配置；参考源码不定义当前持续服务语义。它不包含训练权重、私有数据、完整Python环境或大型fixture。

构建方式采用 [setuptools的src布局](https://setuptools.pypa.io/en/latest/userguide/package_discovery.html)，防止误从当前目录导入未安装源码。安装包可以放在只读位置，所有可写输出使用 `--workspace` 指定。
