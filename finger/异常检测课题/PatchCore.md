Method
>1.Locally aware patch features
>>训练图片集$\mathcal{X}_{N}:\forall x \in\mathcal{X}_{N},y_{x}=0$
>>训练测试集$\mathcal{X}_{N}:\forall x \in\mathcal{X}_{N},y_{x}\in \{0,1\}$
>>$\phi_{i,j}=\phi_{j}(x_{i})$使用预训练模型提取第i个图的第j层特征
>>文中使用包含中层或中间特征表示的内存库 M，以利用提供的训练上下文，比如Resnet Layer2 和layer3提取得到的特征
>>构建每个块级特征表示时，通过局部邻域聚合来增加感受野大小并提高对小空间变化的鲁棒性，同时不损失空间分辨率或特征图的可用性

>2.Coreset-reduced patch-feature memory bank
>>接使用所有块级特征可能会导致记忆库过大，从而增加存储需求和推理时间，因此需要对记忆库进行缩减。核心集缩减是一种减少数据集大小而不显著损失信息的技术。
>>核心集是从原始数据集中选取的代表性子集，它可以近似地表示整个数据集的结构。在 PatchCore 中，通过核心集选择算法从块级特征中选取最具代表性的特征子集。
>>文中它结合了迭代贪婪近似（iterative greedy approximation）和 Johnson-Lindenstrauss 定理来减少计算和存储成本
>
>3.Anomaly Detection with PatchCore
>>对于图像级异常得分的计算
>>>给定特征记忆库 M，对于测试图像 $x^{test}$，首先同样使用预训练网络 ϕ 提取其pacth feature，得到一个patch 特征集合$P(x_{test})$,对于其中的每一个特征$m_{test}$，在M中找到距离其最近的特征$m^{*}$，计算它们的距离，图像级异常得分则是这些距离的最大值，数学表示为：
>>>𝑚test,∗,𝑚∗=arg⁡max⁡𝑚test∈𝑃(𝑥test)(arg⁡min⁡𝑚∈𝑀∥𝑚test−𝑚∥2)mtest,∗​,m∗=argmaxmtest​∈P(xtest​)​(argminm∈M​∥mtest​−m∥2) 𝑠∗=∥𝑚test,∗−𝑚∗∥2s∗=∥mtest,∗​−m∗∥2
$$
m^{test,*},m^{*}=
\underset{m^{test}\in \mathcal{P}(x^{test})}{\text{argmax}}\underset{m\in \mathcal{M}}{\text{argmax}}||m^{test}-m||_{2}
$$
$$
s^{*}=||m^{test,*}-m^{*}||_{2}
$$


>>由于计算图像异常得分的过程中需要计算每一个测试patch的最近邻距离，因此异常分割图也可以在这个过程中得到，同时为了将pacth的大小匹配输入图的大小，使用双线性插值进行上采样，同时使用高斯核进行平滑

Code
>1.在训练阶段，需要完成特征记忆库的构建，核心函数为_fill_memory_bank()这个函数的流程为:
>>对于每个batch的图像数据，首先需要使用_embed()函数提取其中的特征，\_embed函数对应着上述Method 中的1.Locally aware patch features。\_embed的主要流程如下：
>>>假设输入数据的维度为torch.Size([2, 3, 224, 224])，2为batchsize，剩下的三个维度表示图像的形状，首先使用"feature_aggregator"进行特征提取，这里的feature_aggregator是wide_resnet50_2与训练模型，经过这个模型的处理后得到features是一个字典，包含两个键值对，对应两个层次的特征，其中
>>>>features[layer2]的shape为torch.Size([2, 512, 28, 28])
>>>>features[layer3]的shape为torch.Size([2, 1024, 14, 14])
>>
>>>然后提取这两个层的特征得到一个长度为2的list，其中：
>>>>features[0].shape = torch.Size([2, 512, 28, 28])
>>>>features[1].shape = torch.Size([2, 1024, 14, 14])

>>>然后对得到的features进行pactify操作，得到的是一个长度为2的list，
>>>>features\[0\].shape=torch.Size([2, 784, 512, 3, 3])
>>>>features\[1\].shape=torch.Size([2, 196, 1024, 3, 3])

>>>由于features[0]和features[1]大小不一致，因此接下来对features[1]进行一些处理使得二者能够对齐，对于features[1]进行双线性重采样，使得它的第二个维度196对齐784
>>
>>>然后进入preprocessing进行处理，这一步主要进行自适应平均池化得到的features的shape为
>>>>features.shape = torch.Size([1568,2, 1024])

>>>最后再进行preadapt_aggregator,得到的features的shape为
>>>>features.shape = torch.Size([1568, 1024])
>>对于每个图像得到的特征保存在features list中，利用GreedyCoresetSampler进行核心自己采样，再使用anomaly_scorer中的fit方法构建特征记忆库M，构建的方式是使用faiss.GpuIndexFlatL2()方法

>2.在predict阶段,对于输入的一个batsize=2的数据，同样先使用_embed方法得到大小为[1568, 1024]的特征，然后利用image_scores中的predict方法得到一个patch_scores，这个表示每一个patch的异常得分，最后利用convert_to_segmentation得到异常划分图

>>>
>
>
>
>
>
>
>
>
>
