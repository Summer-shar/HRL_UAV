> 归档说明：以下内容是当前仓库自带的先前工程记录，不定义本轮持续服务合同，也不代表本轮测试或训练结果。当前说明见 [场景与控制合同](design_specification.md) 和 [本轮验证](CONTINUING_SERVICE_VALIDATION.md)。

# 标准项目交付验证

本机以离线方式构建wheel，安装到新的虚拟环境；解释器路径不同于原固定Conda路径。测试环境共享已有PyTorch/NumPy依赖，没有下载或升级依赖，不冒充从互联网全新解析全部依赖。

- 5项安装／协议可移植性测试通过。
- 安装后的wheel中51项实现测试通过。
- 安装后的wheel中4项CUDA基线对照通过。
- 从项目外目录执行通用CLI，dry-run为0步、0更新。
- 实际验证前 `training_ready=false`，本地验证通过后为true；并未启动训练。
- 核对wheel资源和源码字节；算法核心16个模块、原物理代码不变，配置仅去掉解释器绝对路径。

机器结果和误差见 `evidence/portable_validation.json`。测试平台为Windows、RTX 5060、CUDA 12.8；Linux CPU CI已经提供但未在本机执行。完整研究预算保持不变，无600轮训练结果。
