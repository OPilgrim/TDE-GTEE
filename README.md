# TDE-GTEE

# 论文的实现（被EACL拟接收）
# 目前暂时开放baseline的实现，这是基于DY的简单复现，后续论文发表之后会开放完整版本

# Datasets
# 本文用到了两个数据集 ACE 和 ERE，由于保密协议，需要自行下载并解析
### ACE
论文中涉及了两个数据集ACE05-E和ACE05-E+，第一步先按照D++的方法得到D++格式的ACE05-E，然后按照Oneie的方法拿到ONEIE格式的ACE05-E和ACE05-E+

### ERE
论文中用的是xxx的数据切分方法

# Train
生成模型和分类模型
`bash scripts train`