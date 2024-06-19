>对于许多模态，可以假设观察的数据可以由看不见的潜在变量(latent variable)表示或者生成，使用随机变量z表示。

>在生成模型中，我们通常寻求学习较低维度的潜在表示，而不是高维的潜在表示。 这是因为如果没有强大的先验知识，尝试学习比观察更高维度的表示是徒劳的。

>从数学方面将，将潜在变量和观察的数据建模为一个联合概率分布$f(x,y)$
>>在基于似然的模型中，有两种方式来利用这个联合概率分布得到已观察数据$p(x)$的概率
>>$p(x)=\int p(x,z)dz$
>>
>>或者$p(x)=\frac{p(x,z)}{p(z|x)}$
>>直接最大化p(x)的概率是比较困难的，因为第一种方式需要对所有的潜在变量z进行积分，第二种方法需要知道groud truth编码器
>>使用这两个公式，可以推导得到证据下界ELBO(Evidence Lower Bound)
>>在这里，证据指的是已观察数据对数似然$\log p(x)$,利用证据下界，优化这个潜在概率模型的问题转换为最大化证据下界ELBO。其中证据下届ELBO具体如下：
>>$$E_{q_{\phi}(z|x)}[\log \frac{p(x,z)}{q_{\phi}(z|x)}]$$
>>证据与证据下界的关系如下：
>>$$\log p(x) \ge E_{q_{\phi}(z|x)}[\log \frac{p(x,z)}{q_{\phi}(z|x)}]$$
>>其中$q_{\phi}(z|x)$是一个针对潜在变量$z$的概率分布的近似，被用来近似真实的后验分布$p(z|x)$，$\phi$是一个优化参数，通过优化它使得$q_{\phi}(z|x)$尽可能的接近真实的后验分布$p(z|x)$
>>$$
\begin{aligned}
\log p(x) &=\log p(x)\int q_{\phi}(z|x)dz \\ 
 & = \int q_{\phi}(z|x)(\log p(x))dz \\
 & = E_{q_{\phi}(z|x)}[\log p(x)] \\ 
 & = E_{q_{\phi}(z|x)}[\log \frac{p(x,z)}{p(z|x)}]\\
 & = E_{q_{\phi}(z|x)}[\log \frac{p(x,z)q_{\phi}(z|x)}{p(z|x)q_{\phi}(z|x)}]\\
  & = E_{q_{\phi}(z|x)}[\log \frac{p(x,z)}{q_{\phi}(z|x)}]+E_{q_{\phi}(z|x)}[\log \frac{q_{\phi}(z|x)}{p(z|x)}]\\
  & = E_{q_{\phi}(z|x)}[\log \frac{p(x,z)}{q_{\phi}(z|x)}]+D_{KL}(q_{\phi}(z|x)||p(z|x))\\
  & \ge E_{q_{\phi}(z|x)}[\log \frac{p(x,z)}{q_{\phi}(z|x)}]\\
\end{aligned}$$
>>为什么要寻求最大化ELBO：
>>>我们的目标是学习描述我们观察到的数据的潜在潜在结构，通过优化变分后验$q_{\phi}(z|x)$的参数$\phi$来尽可能匹配真实后验$p(z|x)$，这是通过最小化二者的KL散度得到. 
>>>由于ELBO与DL散度的和为一个常数(即$\log p(x)$)，所以让ELBO尽可能大会使得DL散度尽可能小。
>>>
>>