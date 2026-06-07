# 实验分析总结

# 2026.5.28
分类器lr0.003 + 分类器训练了100 epochs + 传统数据增强情况下 0.681972135545076
以下增强图片的推理步数为100，hard样本筛选使用了上面的传统数据增强的baseline，加传统数据增强
lora rank32 + 分类器lr0.003 + 分类器训练了100 epochs + random150张数据增强情况下 0.6942783303070381
lora rank32 + 分类器lr0.003 + 分类器训练了100 epochs + hard150张数据增强情况下 0.7066670344500541

分析：相比较传动数据增强，扩散模型增强轻微。相比于random增强，hard增强轻微。
结论：扩散模型方法现在有轻微增强；先都训练到epoch 100，模型后期可能还在提升；选择best epoch进行比较；

lora rank16 + 分类器lr0.003 + 分类器训练了60 epochs + random150张数据增强情况下 0.7094652456390848
lora rank16 + 分类器lr0.003 + 分类器训练了60 epochs + hard150张数据增强情况下 0.7110602292986467
<!-- lora rank16 + 分类器lr0.003 + 分类器训练了60 epochs + random100张数据增强情况下 0.7160771717719933
lora rank16 + 分类器lr0.003 + 分类器训练了60 epochs + hard100张数据增强情况下 0.6936443812186328 -->
lora rank16 + 分类器lr0.001 + 分类器训练了100 epochs + random100张数据增强情况下 0.7052526797123289
lora rank16 + 分类器lr0.001 + 分类器训练了100 epochs + hard100张数据增强情况下 0.7264969028145184

以下增强图片的推理步数为250，且无传统数据增强
lora rank16 + 分类器lr0.001 + 分类器训练了100 epochs + random100张数据增强情况下 0.5989839269884623
lora rank16 + 分类器lr0.001 + 分类器训练了100 epochs + hard100张数据增强情况下 0.6223415001789311


分类器lr0.001 + 分类器训练了100 epochs + 无传统数据增强情况下 0.6090130672698074
以下hard样本筛选使用了上面的无传统数据增强的baseline，增强图片的推理步数为250 + 传统数据增强
lora rank16 + 分类器lr0.001 + 分类器训练了100 epochs + random100张数据增强情况下 0.6757379210378841
lora rank16 + 分类器lr0.001 + 分类器训练了100 epochs + hard100张数据增强情况下 0.6956535511635984

以下hard样本筛选使用了上面的无传统数据增强的baseline，增强图片的推理步数为100，且无传统数据增强
lora rank16 + 分类器lr0.001 + 分类器训练了100 epochs + random100张数据增强情况下 0.6040521389287393
lora rank16 + 分类器lr0.001 + 分类器训练了100 epochs + random300张数据增强情况下 0.6190759152207119
lora rank16 + 分类器lr0.001 + 分类器训练了100 epochs + hard100张数据增强情况下  0.5969044641572978
lora rank16 + 分类器lr0.001 + 分类器训练了100 epochs + hard300张数据增强情况下  0.632218192250877

分析：相比于rank16，rank32稍微更差。相比于100张，150张稍微提升，300张继续稍微提升。相比于推理步数为100，推理步数为250似乎没什么改变。有无统数据增强上述结果似乎改变不大。

结论：在当前strength=0.45情况下，理论上推理步数为112左右为最佳。

固定实验配置：（重要的是对比）
lora rank16 + 分类器lr0.0005 + 分类器100 epochs + 无传统数据增强 + 推理步数100 + 300张数据增强 + hard样本筛选使用无传统数据增强的baseline + strength0.45 + hardratio


分类器lr0.001 + 分类器训练了100 epochs + 传统数据增强情况下 0.682903001570617
以下传统实验使用了传统数据增强
lora rank16 + 分类器lr0.001 + 分类器训练了100 epochs + hard300张数据增强情况下 0.6927886782101379 
lora rank16 + 分类器lr0.001 + 分类器训练了100 epochs + hard100张数据增强情况下 0.6762338922939959
lora rank16 + 分类器lr0.001 + 分类器训练了100 epochs + random300张数据增强情况下 0.7106448183163698
lora rank16 + 分类器lr0.001 + 分类器训练了100 epochs + random100张数据增强情况下 0.7078868510828238

lora rank16 + 分类器lr0.001 + 分类器训练了100 epochs + hard100random200张数据增强情况下 0.6813148954415418

lora rank16 + 分类器lr0.001 + 分类器训练了100 epochs + hard300张数据增强情况下(ratio0.3 seed30) 0.6927886782101379
lora rank16 + 分类器lr0.001 + 分类器训练了100 epochs + hard100张数据增强情况下(ratio0.1 seed10) 0.7052618583287504
lora rank16 + 分类器lr0.001 + 分类器训练了100 epochs + hard200张数据增强情况下(ratio0.1 seed10 aug 20) 0.6748695281760665
lora rank16 + 分类器lr0.001 + 分类器训练了100 epochs + hard100张数据增强情况下(ratio0.05 seed5 aug 20) 0.7024588869372481

<!-- 过去的传统增强
lora rank16 + 分类器lr0.001 + 分类器训练了100 epochs + hard300张数据增强情况下 
lora rank16 + 分类器lr0.001 + 分类器训练了100 epochs + random300张数据增强情况下  -->

以下改变strength为0.6
lora rank16 + 分类器lr0.001 + 分类器训练了100 epochs + hard150张数据增强情况下 0.6973508444272992 
lora rank16 + 分类器lr0.001 + 分类器训练了100 epochs + random150张数据增强情况 0.6998441525571837

分类器lr0.0005 + 分类器训练了100 epochs + 传统数据增强情况下 0.6751190768867422/0.6167800529548331/
lora rank16 + 分类器lr0.0005 + 分类器训练了100 epochs + hard300张数据增强情况下(ratio0.3 seed30) 0.7224662175272074 / 0.7104527385341717
lora rank16 + 分类器lr0.0005 + 分类器训练了100 epochs + hard300张数据增强情况下(ratio0.3 seed30 使用增强baseline筛选+epoch80)  0.7220760322019586
lora rank16 + 分类器lr0.0005 + 分类器训练了100 epochs + random300张数据增强情况下(seed30 epoch80) 0.683508346825893
lora rank16 + 分类器lr0.0005 + 分类器训练了100 epochs + hard300张数据增强情况下(seed30 epoch120) 0.6837372300294532

lora rank16 + 分类器lr0.0005 + 分类器训练了100 epochs + hard300张数据增强情况下(ratio0.3 seed30 使用增强baseline筛选+epoch60) 0.7247981084091829/  /0.6906252680201731
lora rank16 + 分类器lr0.0005 + 分类器训练了100 epochs + random300张数据增强情况下(seed30 epoch60) 0.7027214618743391/  /0.7032484721458544

lora rank16 + 分类器lr0.0005 + 分类器训练了100 epochs + hard300张数据增强情况下(ratio0.3 seed30 除去nv) 0.6799970706630437
lora rank16 + 分类器lr0.0005 + 分类器训练了100 epochs + hard300张数据增强情况下(ratio0.3 seed30 strength0.3) 0.6844039844455735
lora rank16 + 分类器lr0.0005 + 分类器训练了100 epochs + hard300张数据增强情况下(ratio0.3 seed30 strength0.2) 0.6804598032563793
lora rank16 + 分类器lr0.0005 + 分类器训练了100 epochs + hard300张数据增强情况下(ratio0.3 seed30 strength0.8) 0.6806425959564296
lora rank16 + 分类器lr0.0005 + 分类器训练了100 epochs + hard300张数据增强情况下(ratio0.3 seed30 除去VASC) 0.67966385735633

lora rank16 + 分类器lr0.0005 + 分类器训练了100 epochs + random300张数据增强情况下 0.7100746270330094 
lora rank16 + 分类器lr0.0005 + 分类器训练了100 epochs + random300张数据增强情况下(seed100 * aug3) 0.7020303315561565

lora rank16 + 分类器lr0.0005 + 分类器训练了100 epochs + hard300张数据增强情况下(ratio0.3 seed30 lora80epoch) 0.68437638354669
lora rank16 + 分类器lr0.0005 + 分类器训练了100 epochs + hard300张数据增强情况下(ratio0.1 seed30 lora40epoch) 0.6470935076163861

lora rank16 + 分类器lr0.0005 + 分类器训练了100 epochs + hard300张数据增强情况下(ratio0.1 seed10 aug 30) 0.7220282215583115
lora rank16 + 分类器lr0.0005 + 分类器训练了100 epochs + hard600张数据增强情况下(ratio0.1 seed10 aug 60) 0.6463700135173388
lora rank16 + 分类器lr0.0005 + 分类器训练了100 epochs + hard1000张数据增强情况下(ratio0.1 seed10 aug 100) 0.6933177592103844

lora rank16 + 分类器lr0.0005 + 分类器训练了100 epochs + hard100张数据增强情况下(ratio0.05 seed5 aug 20) 0.6696991465342649
lora rank16 + 分类器lr0.0005 + 分类器训练了100 epochs + random600张数据增强情况下 0.6701234931236412
lora rank16 + 分类器lr0.0005 + 分类器训练了100 epochs + random600张数据增强情况下(seed43) 0.6846848354445113

lora rank16 + 分类器lr0.0005 + 分类器训练了100 epochs + hard300张数据增强情况下(分类器过滤了50%) 0.6573051555785904 (换了更好的classifier过滤)-> 0.6784740710105314
lora rank16 + 分类器lr0.0005 + 分类器训练了100 epochs + hard插值数据增强情况下 0.6958569804320651
lora rank16 + 分类器lr0.0005 + 分类器训练了100 epochs + hard插值数据增强情况下(过滤了) 0.6700365745114966


# 分析
MEL和DF的recall最低，BKLVASC和NV最高
