   DDPM：降噪扩散概率模型，分为前向扩散过程和反向逆扩散过程
>1.前向扩散过程
>>给定初始数据分布$x_{0}\sim q(x)$，不断向分布中添加高斯噪声，该噪声的标准差是以固定值$\beta_{t}$而确定的，均值是以$\beta_{t}$和当前时刻的数据$x_{t}$决定的，这个过程是一个马尔可夫链过程
>>$q(x_{t}|x_{t-1})=N(x_{t};\sqrt{1-\beta_{t}}x_{t-1},\beta_{t}I)$
>>$q(x_{1:T}|x_{0})=\prod\limits_{t=1}\limits^{T}q(x_{t}|x_{t-1})$
>>推导：
>>>任意时刻的$q(x_{t})$可以根据$x_{0}$和$\beta_{t}$计算出来，无需迭代
>>>首先需要知道对于两个正态分布$X\sim N(\mu_{1},\sigma_{1})$和$Y\sim N(\mu_{2},\sigma_{2})$,其线性组合依旧是一个正态分布，aX+bY的均值为$a\mu_{1}+b\mu_{2}$，方差为$a^{2}\mu_{1}^{2}+b^{2}\mu_{2}^{2}$
>>>令$\alpha_{t}=1-\beta_{t}$
>>>则
>>>$x_{t}=\sqrt{\alpha_{t}}x_{t-1}+\sqrt{1-\alpha_{t}}z_{t-1}$
>>>$$
\begin{aligned}
x_{t} &=\sqrt{\alpha_{t}}x_{t-1}+\sqrt{1-\alpha_{t}}z_{t-1}\\
 & = \sqrt{\alpha_{t}\alpha_{t-1}}x_{t-2}+\sqrt{\alpha_{t}(1-\alpha_{t-1})}z_{t-2}+ \sqrt{1-\alpha_{t}}z_{t-1} \\ 
 & = \sqrt{\alpha_{t}\alpha_{t-1}}x_{t-2}+ \sqrt{1-\alpha_{t}\alpha_{t-1}}z\\
 & = \cdots \\ 
 & = \sqrt{\bar{\alpha_{t}}}x_{0}+\sqrt{1-\bar{\alpha_{t}}}z\\
 &where\ \bar{\alpha_{t}}=\prod\limits_{t=1}\limits^{T}\alpha_{t}
\end{aligned}
$$
>>>这样得到了$x_{t}$和$x_{0}$的关系$x_{0}=\frac{1}{\sqrt{\bar{\alpha_{t}}}}(x_{t}-\sqrt{1-\bar{\alpha_{t}}}z)$
>2.反向扩散过程
>>逆扩散过程从高斯噪声中恢复原始数据
>>$p(x_{0:T})=p(x_{T})\prod\limits_{t=1}\limits^{T}p_{\theta}(x_{t-1}|x_{t})$
>>$p_{\theta}(x_{t-1}|x_{t})=N(x_{t-1};\mu_{\theta}(x_{t},t),\Sigma_{\theta}(x_{t},t))$
>>后验的扩散条件概率可以表示出来
>>$q_{\theta}(x_{t-1}|x_{t},x_{0})=N(x_{t-1};\widetilde\mu_{\theta}(x_{t},x_{0}),\widetilde\beta I)$
>>>$$
\begin{aligned}
q(x_{t-1}|x_{t},x_{0})&=\frac{q(x_{t-1},x_{t},x_{0})}{q(x_{t},x_{0})}\\
 & = \frac{q(x_{t}|x_{t-1},x_{0}){q(x_{t-1},x_{0})}}{q(x_{t},x_{0})}\\ 
  & = \frac{q(x_{t}|x_{t-1},x_{0}){q(x_{t-1}|x_{0})}}{q(x_{t}|x_{0})}\\ 
 & = exp(-\frac{1}{2}(\frac{(x_{t} - \sqrt{\alpha_{t}}x_{t-1})^2}{\beta{t}} +\frac{(x_{t-1}-\sqrt{\bar\alpha_{t-1}}x_{0})^{2}}{1-\bar\alpha_{t-1}} - \frac{(x_{t}-\sqrt{\alpha_{t}}x_{0})^{2}}{1-\bar\alpha_{t}} ))\\
& =exp(-\frac{1}{2}((\frac{\alpha_{t}}{\beta_{t}}+\frac{1}{1-\bar\alpha_{t-1}})x_{t-1}^{2}
-(\frac{2\sqrt{\alpha_t}}{\beta_{t}}x_{t}+\frac{2\sqrt{\bar\alpha_{t-1}}}{1-\bar\alpha_{t-1}}x_{0})x_{t-1}
+C(x_{t},x_{0})\\
&where C(x_{t},x_{0})是关于x_{t},x_{0}的常函数
\end{aligned}
$$
>>>配方之后可以求出这个高斯分布的均值和方差因此可以得到
>>>$\widetilde\beta_{t}=\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_{t}}\beta_{t}$
>>>$\widetilde\mu_{\theta}(x_{t},x_{0})=\frac{\sqrt\alpha_{t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_{t}}x_{t}+\frac{\sqrt{\bar\alpha_{t-1}}\beta_{t}}{1-\bar\alpha_{t}}x_{0}$     
>>>由$x_{0}和x_{t}$的关系可得
>>>$\widetilde\mu_{t}=\frac{1}{\sqrt{\alpha_{t}}}(x_{t}-\frac{\beta{t}}{\sqrt{1-\bar{\alpha_{t}}}}z_{t})$        
>3.目标数据分布的似然函数
>>基于扩散的生成模型的训练目标相当于"在反向过程结束时最大化生成的属于原始数据分布的样本的对数似然"
>>>$$
\begin{aligned}
-\log p_{\theta}(x_{0})&\le -\log p_{\theta}(x_{0})+D_{KL}(q(x_{1:T})||p(x_{1:T})|x_{0})\\
 &= -\log p_{\theta}(x_{0})+E_{x_{1:T}\sim q(x_{1:T}|x_{0})}[\log \frac{q(x_{1:T}|x_{0})}{p_{\theta}(x_{0:T})/p_{\theta}(x_{0})}]\\ 
  & = -\log p_{\theta}(x_{0})+E_{q}[\log \frac{q(x_{1:T}|x_{0})}{p_{\theta}(x_{0:T})}+\log p_{\theta}(x_{0})]\\ 
 & =  E_{q}[\log \frac{q(x_{1:T}|x_{0})}{p_{\theta}(x_{0:T})}]\\
&let\ L_{VLB} = E_{q}[\log \frac{q(x_{1:T}|x_{0})}{p_{\theta}(x_{0:T})}]\ge-E_{q(x_{0})}\log p_{\theta}(x_{0})
\end{aligned}

$$
>>>以下式子的推导利用到$q(x_{t}|x_{t-1})=q(x_{t}|x_{t-1},x_{0})=\frac{q(x_{t},x_{t-1},x_{0})}{q(x_{t-1},x_{0})}=\frac{q(x_{t-1}|x_{t},x_{0})q(x_{t}|x_{0})q(x_{0})}{q(x_{t-1},x_{0})}=\frac{q(x_{t-1}|x_{t},x_{0})q(x_{t}|x_{0})}{q(x_{t-1}|x_{0})}$
>>>$$
\begin{aligned}
L_{VLB} &= E_{q}[\log \frac{q(x_{1:T}|x_{0})}{p_{\theta}(x_{0:T})}]\\
 &= E_{q}[\log \frac{\prod\limits_{t=1}\limits^{T}q(x_{t}|x_{t-1})}{p_{\theta}(x_{T})\prod\limits_{t=1}\limits^{T}p_{\theta}(x_{t-1}|x_{t})}]\\
  &= E_{q}[-\log p_{\theta}(x_{T})+\sum\limits_{t=1}\limits^{T} \log\frac{q(x_{t}|x_{t-1})}{p_{\theta}(x_{t-1}|x_{t})}]\\
 &= E_{q}[-\log p_{\theta}(x_{T})+\sum\limits_{t=2}\limits^{T} \log\frac{q(x_{t}|x_{t-1})}{p_{\theta}(x_{t-1}|x_{t})}+\log\frac{q(x_{1}|x_{0})}{p_{\theta}(x_{0}|x_{1})}]\\
 &= E_{q}[-\log p_{\theta}(x_{T})+\sum\limits_{t=2}\limits^{T} \log\frac{q(x_{t-1}|x_{t},x_{0})}{p_{\theta}(x_{t-1}|x_{t})}\cdot\frac{q(x_{t}|x_{0})}{q(x_{t-1}|x_{0})}+\log\frac{q(x_{1}|x_{0})}{p_{\theta}(x_{0}|x_{1})}]\\
 &= E_{q}[-\log p_{\theta}(x_{T})+\sum\limits_{t=2}\limits^{T} \log\frac{q(x_{t-1}|x_{t},x_{0})}{p_{\theta}(x_{t-1}|x_{t})}+\sum\limits_{t=2}\limits^{T}\log
\frac{q(x_{t}|x_{0})}{q(x_{t-1}|x_{0})}+\log\frac{q(x_{1}|x_{0})}{p_{\theta}(x_{0}|x_{1})}]\\
 &= E_{q}[-\log p_{\theta}(x_{T})+\sum\limits_{t=2}\limits^{T} \log\frac{q(x_{t-1}|x_{t},x_{0})}{p_{\theta}(x_{t-1}|x_{t})}+\log
\frac{q(x_{T}|x_{0})}{q(x_{1}|x_{0})}+\log\frac{q(x_{1}|x_{0})}{p_{\theta}(x_{0}|x_{1})}]\\
&= E_{q}[\log\frac{q(x_{T}|x_{0})}{p_{\theta}(x_{T})}
+\sum\limits_{t=2}\limits^{T} \log\frac{q(x_{t-1}|x_{t},x_{0})}{p_{\theta}(x_{t-1}|x_{t})}
-\log p_{\theta}(x_{0}|x_{1})]\\
\end{aligned}
$$      
>>>       
>>>      
>>>
>>>
>>>
>>
>>
>>
>>
>>
>>
>>
>>
>>
>>
>>
>>
>>
>>
>>
>>
