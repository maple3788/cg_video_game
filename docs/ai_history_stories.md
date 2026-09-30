# AI 历史故事库：SAI Game 72 关的复核与补充

> 依据《SAI Game》设计文档（Act I–VII 共 72 关，外加 Level 73）整理，内容分四块：
> ① 核对现有关卡的年代和史实；② 给每关找能放进「Historical Reveal」的真实细节；
> ③ 补上表里缺、又适合做成关卡的故事；④ 提出跨关卡 callback 和试玩版选关建议。
>
> 说明：PDF 里的关卡表是截图，右侧的玩法列被截掉了，下文只依据能看到的「年代 / 关卡 / 场景」三列。
> 标有 🗣️ 的是传闻、流传版本或半虚构内容，游戏里用到时需要注明。

---

## 0. 结论速览

1. **骨架没问题。** 72 关从 1943 年一直讲到 2026 年，Eureka / Experiment / Battle / Story 四种玩法交替出现，这个设计是对的。
2. **有 8 处年代或史实要改（见 §1）。** 最要紧的是：L47 Attention 应为 **2014**，L50 The Bitter Lesson 应为 **2019**，L25 的 1997 年版 LSTM **还没有遗忘门**，L6 所在的 1957 年 **Mark I 还没造出来**。
3. **最大的缺口是语言模型的前史。** Shannon 1951 年的「猜下一个字母」、Bengio 2003 年的神经语言模型、word2vec（2013）、Seq2Seq（2014）都不在表里。你们已有的 Demo（本仓库的 TOKEN 预告片）偏偏讲了 word2vec、Seq2Seq 和 RAG，两边对不上。
   另外，强化学习（DQN、AlphaGo Zero）、扩散模型、多模态、自动驾驶和机器人这几块几乎是空的。
4. **建议加两条贯穿全作的主线：**
   - 「机器里藏着人」：1770 年的土耳其行棋傀儡 → Deep Blue → 给 ImageNet 做标注的 Mechanical Turk → RLHF 标注员 → L73，机器里第一次可能真的没有人。
   - 「标签从哪来」：专家写规则 → 众包标注 → 文本自带标签（下一个词）→ 人类偏好 → 可验证奖励。
5. **L73 需要的铺垫，2025–2026 年已经真实发生了。**
   - 2025-05：AlphaEvolve 打破了 Strassen **1969** 年的纪录。1969 年也正是 XOR 之门关上的那一年。
   - 2025-10：「GPT-5 解决了 10 个 Erdős 问题」被证实只是检索到了已有论文。
   - 2026-05：OpenAI 的模型真正反证了 Erdős 1946 年提出的单位距离猜想。
   - 从「检索」走到「发现」，这本身就是一段现成的剧情。

---

## 1. 需要修正的史实

| 关卡 | 文档写法 | 史实 | 建议 |
|---|---|---|---|
| L6 The Perceptron | 1957，Rosenblatt 面前是一台巨大的 Mark I | 1957 年的感知机只是 IBM 704 上的软件模拟。Mark I 硬件 1959 年才组装，**1960-06-23** 首次公开演示 | 场景改成 IBM 704，或把年代写成 1957–1960 |
| L25 LSTM | 1997，操纵 forget / input / output 门 | 1997 年原版只有输入门、输出门和恒定误差传送带（CEC）。**遗忘门是 1999–2000 年 Gers、Schmidhuber、Cummins 加上的** | 正好拆成两段：先用 1997 版过关，再到长序列上「记忆塞满」，让玩家自己发明遗忘门 |
| L34 Dropout | 2013 | 2012-07 上了 arXiv（Hinton 等），同年的 AlexNet 已经在用；JMLR 正式版是 2014 年 | 改成 2012，放到 L33 之前，或与 L33 合并 |
| L47 Attention | 2017，翻译长句时 RNN 忘了开头 | 这是 **2014-09** Bahdanau、Cho、Bengio 的工作（ICLR 2015） | 改成 2014，挪到 Act IV，和新增的 Seq2Seq 关配成一对 |
| L50 The Bitter Lesson | 2018 | Sutton 的短文发表于 **2019-03-13** | 改成 2019 |
| L55 Double Descent | 2020 | Belkin 等 2018-12 上 arXiv（PNAS 2019）；Nakkiran 等的 *Deep Double Descent* 是 2019-12 | 改成 2019 |
| L62 Emergent Abilities | 2021 | Wei 等的论文发表于 **2022-06**。2023 年 Schaeffer 等撰文指出「涌现」可能是评测指标造成的错觉（*Mirage*，NeurIPS 2023 杰出论文） | 改成 2022，并把 Mirage 这个反转做进关卡（见 §2） |
| L72 Reasoning Models | 2025–26 | 第一个公开的推理模型 o1-preview 发布于 **2024-09** | 改成 2024–26 |

另外，L47 挪走以后，Act V「2016–2020」这个年代范围才能对得上。

---

## 2. 现有关卡：可以直接用的真实细节（Historical Reveal 素材）

### Act I · 机器能思考吗？（1943–1969）

- **L1 The Artificial Neuron（1943）**
  - Walter Pitts 12 岁时为躲避街头霸凌钻进底特律的公共图书馆，花三天读完罗素与怀特海的《数学原理》，还写信给罗素指出书里的问题。15 岁时他离家出走去了芝加哥。
  - McCulloch 把无家可归的 Pitts 接到自己家里住，两人常常熬夜讨论，合写出了 1943 年的论文。
  - 结局很悲凉：Pitts 后来烧掉了自己没发表的论文手稿；**1969 年，Pitts 和 McCulloch 相继去世，那一年《Perceptrons》出版，XOR 之门关上了**。这一点可以和 L12、L13 呼应。
- **L2 The Imitation Game（1950）**
  - 论文第一句就是 *"I propose to consider the question, 'Can machines think?'"*，正好是 Act I 的标题。
  - Turing 预言：到 2000 年，机器能在 5 分钟对话里让普通提问者有 30% 的概率认错。
  - 最后一节「Learning Machines」提出：与其模拟成人的大脑，不如模拟儿童的大脑再去教它。这是机器学习的第一张草图。
  - 论文还专门反驳了「Lovelace 夫人的反对意见」，也就是「机器不能创造任何新东西」。这一点可以留到 L73 回收（见 §3 S12）。
- **L3 The Rat in the Maze（1951）**
  - SNARC 大约有 40 个「神经元」，用了约 3000 个真空管，还拆用了一台 B-24 轰炸机的剩余自动驾驶部件。
  - **讽刺的伏笔**：亲手造出早期神经网络机器的 Minsky，18 年后与 Papert 合写了让神经网络研究降温的《Perceptrons》（L13）。
- **L4 Machines That Learn（1952）**
  - 1956 年跳棋程序上了电视，据说 IBM 股价随后大涨 🗣️。
  - 1959 年 Samuel 在论文标题里用了 *Machine Learning* 这个词；1962 年程序赢了跳棋高手 Robert Nealey。
  - 可以回收到 2007 年：Chinook「解决」了跳棋，双方都走最优棋时必然和棋（见 §3）。
- **L5 Dartmouth Summer（1956）**
  - 1955 年的提案里写道：如果精心挑一群科学家一起干一个夏天，就能在其中一个或多个问题上取得重大进展。这大概是 AI 史上最乐观的一句话，适合当开场白。
  - 「起名字」小游戏的候选名可以全用真的：
    - *Automata Studies*：McCarthy 与 Shannon 刚编完的论文集书名；
    - *Cybernetics*：McCarthy 说，起新名字的原因之一就是不想被当成 Wiener 的门徒；
    - *Complex Information Processing*：Newell 和 Simon 更喜欢的叫法；
    - *Artificial Intelligence*。
  - 会上唯一真正能跑起来的程序，是 Newell 和 Simon 带来的 Logic Theorist（见 §3）。
- **L6 The Perceptron**
  - Mark I 的「视网膜」是 20×20 的光电管阵列，权重是电动机拧动的电位器。学习过程能被「看见」也能被「听见」，很适合做音画。
  - Minsky 和 Rosenblatt 都在纽约的 Bronx 科学高中读过书，年龄只差一岁。
- **L7 The Machine That Walks（1958）**
  - 关卡名出自 1958-07-08 的《纽约时报》：海军展示了一台电子计算机的雏形，预计它将能够行走、说话、看、写、自我复制，并意识到自己的存在。
  - 可以做成「报纸头条 vs 实验室现实」的对照玩法。
- **L8 ADALINE（1960）**
  - 1959 年秋，Widrow 和他的第一个博士生 Ted Hoff 推导出 LMS 算法（Widrow–Hoff 规则），本质上就是随机梯度下降。
  - **彩蛋**：Hoff 后来加入 Intel，是 1971 年第一款商用微处理器 4004 的关键设计者之一，可以埋进 Compute 线。
- **L9 ELIZA（1966）**
  - Weizenbaum 的秘书看着他把程序写出来，自己试用时却请他离开房间，好跟 ELIZA 单独聊。
  - 这件事让 Weizenbaum 后半生都在警告人们别过度信任计算机（《Computer Power and Human Reason》，1976）。
  - 名字来自《卖花女》里的 Eliza Doolittle。
  - 2022 年的 LaMDA 事件（§3）是它 56 年后的回声。
- **L10 The Translation Disaster（1966）**
  - 1954 年的 Georgetown–IBM 演示把 60 多句俄语译成了英语，主持者预言机器翻译 3–5 年内就能解决。
  - 1960 年，Bar-Hillel 用一句 **"The box was in the pen."** 论证机器翻译离不开世界知识：pen 在这里是「围栏」，不是「钢笔」。这句话本身就是一道完美的谜题。
  - 1966 年的 ALPAC 报告终结了美国对机器翻译的资助。
  - 后面有三次回收：2014 年的注意力、2016 年的 Google 神经机器翻译（GNMT）、2017 年的 Transformer（它的第一个任务就是翻译）。
- **L11 Shakey（1968）**
  - 副产品包括 A* 搜索（1968）和 STRIPS 规划器。
  - 1970 年《Life》杂志称它为 *first electronic person*。
- **L12–13 XOR / The Book（1969）**
  - Minsky 和 Papert 其实知道多层网络能表示 XOR，难处在于当时没有训练多层网络的方法；他们推测把研究扩展到多层会是「贫瘠的」。书中的核心例子是 parity 和 connectedness。
  - 1971 年，Rosenblatt 在 43 岁生日那天划船遇难，没能看到反向传播。
  - 「一本书杀死了神经网络」是简化过的说法，事实的复杂本身就是人文感的来源（见 §7）。

### Act II · AI 的寒冬与地下火种（1970–1989）

- **L14 SHRDLU（1970）**：名字来自排字机键盘上按字母频率排列的 "ETAOIN SHRDLU"，正好接上 Shannon 的字母频率（§3 S2）。
- **L15 The Funding Review（1973）**：Lighthill 报告出来后，BBC 在英国皇家研究院录了一场电视辩论，Lighthill 对阵 Donald Michie、John McCarthy 和 Richard Gregory，是现成的 Battle 关舞台。
- **L16 Winter Is Coming（1974）**：「AI 寒冬」这个词直到 1984 年才由 Roger Schank 和 Minsky 在 AAAI 年会上提出（类比「核冬天」），而且是作为对未来的警告。1974 年时还没人这么叫。
- **L17 MYCIN（1976）**：在 1979 年的盲评中，MYCIN 的治疗建议评分不低于斯坦福的传染病专家，但它从没进过临床，卡在责任归属和与医院流程的衔接上。「它是对的，但没人用它」，是个很好的反高潮。
- **L18 The Expert System Factory（1980）**：DEC 的 XCON/R1 靠几千条规则配置电脑订单，据称每年省下几千万美元；可规则一多就没人维护得动，这就是「知识工程瓶颈」。同一时期日本启动了「第五代计算机」计划（1982）。
- **L19 Hopfield's Memory（1982）**：**42 年后，Hopfield 和 Hinton 共同获得 2024 年诺贝尔物理学奖**，是最有力的跨关卡回收之一。
- **L20 Backprop（1986）**
  - 反向传播被独立发现过好几次：Linnainmaa 1970 年的反向模式自动微分、Werbos 1974 年的博士论文、Parker、LeCun 1985 年。Rumelhart、Hinton 和 Williams 1986 年那篇 Nature 论文的贡献，是展示了它能学出有意义的内部表示。「谁发明了 backprop」可以做一条 Mystery 支线。
  - 1987 年 Hinton 去了多伦多，原因之一是不想拿美国军方（DARPA）的钱。加拿大（CIFAR）成了「地下火种」的避难所。
- **L21 The Second Winter（1987）**：专用的 Lisp 机器输给了更便宜的通用工作站；30 年后，反而是专用芯片（GPU、TPU）赢了。这是 Compute 线上的一次反转。
- **L22 LeNet（1989）**：前身是福岛邦彦 1980 年在 NHK 研究所提出的 Neocognitron。据 LeCun 说，90 年代末基于它的支票识别系统处理了全美一成以上的支票 🗣️。

### Act III · 神经网络的漫长潜伏（1990–2009）

- **L23 The Vanishing Signal（1991）**：最早系统分析梯度消失的，是 Sepp Hochreiter 用**德语写的 Diplom 毕业论文**（导师 Schmidhuber），此后多年几乎没人注意到。
- **L24 SVM Strikes Back（1995）**
  - 1995-03-14，Bell Labs 的 Larry Jackel 和 Vapnik 立了两个赌约，赌注都是一顿大餐，见证人签名是 **Yann LeCun**：
    - 到 2000 年，会不会出现能解释大型神经网络为什么有效的理论？Jackel 赌会，**他输了**。
    - 到 2005 年，还会不会有人用 1995 年那种神经网络？Vapnik 赌不会，**他也输了**。
  - 连 Cortes 和 Vapnik 1995 年的论文标题都叫 *Support-Vector **Networks***。
- **L25 LSTM（1997）**：修正见 §1。LSTM 被冷落近 20 年后爆发，撑起了 Google 语音识别（2015）和 Google 翻译（2016）。
- **L26 Deep Blue（1997）**
  - 第一局第 44 步，一个 bug 让 Deep Blue 随机走了一步，Kasparov 却把它理解成「更深的智能」，心神不宁。这个说法来自团队成员 Murray Campbell，见 Nate Silver 的《信号与噪声》。
  - 第二局，Kasparov 在其实可以长将逼和的局面下认输，还怀疑 IBM 在幕后有人操纵。**这正好回收了 1770 年的土耳其行棋傀儡（§3 S1）**。
  - Simon 1958 年预言电脑「十年内」成为国际象棋冠军，实际用了 39 年。
- **L27 Gradient-Based Learning（1998）**：LeNet-5 和 MNIST 同年发表，是 Data 线的第一个节点。
- **L28 Deep Learning Returns（2006）**：CIFAR 从 2004 年起资助 Hinton、LeCun、Bengio 的小圈子。他们有意把「神经网络」改叫「深度学习」，因为旧名字在审稿人那里名声太差。
- **L29 ImageNet（2009）**
  - 李飞飞读大学期间要帮父母打理干洗店（见回忆录《我看见的世界》）。
  - ImageNet 靠 Amazon Mechanical Turk 众包标注，而 Mechanical Turk 这个名字就来自 1770 年那台行棋傀儡。
  - ImageNet 论文在 CVPR 2009 上只是一张 poster。
- **L30 The GPU Gamble（2009）**：Raina、Madhavan、吴恩达 2009 年报告，用 GPU 训练比 CPU 快约 70 倍。

### Act IV · Deep Learning Big Bang（2010–2016）

- **L31 Watson（2011）**
  - Final Jeopardy 那道「美国城市」题，Watson 答的是 *"What is Toronto?????"*。
  - Ken Jennings 在最后一题的答题板上写下 *"I, for one, welcome our new computer overlords."*
  - Watson 的本质是「检索 + 候选答案打分」，**和 2020 年的 RAG 是同一个思路**（§3）。后来 Watson Health 的挫折，是讲「炒作」的好注脚。
- **L32 The Cat（2012）**
  - 1.6 万个 CPU 核、1000 台机器、1000 万张 YouTube 截图。《纽约时报》的标题是 *How Many Computers to Identify a Cat? 16,000*。
  - 一年后，Coates 和吴恩达只用 3 台 GPU 服务器就训练出了同等规模的网络，回收 L30。
- **L33 AlexNet（2012）**
  - 模型是在 Alex 父母家卧室里的两块 GTX 580 上训练的，跑了 5–6 天。top-5 错误率 15.3%，第二名是 26.2%。
  - 同年 12 月 NIPS 在太浩湖召开，Hinton 在 Harrah's 赌场酒店的房间里为三个人的公司 DNNresearch 主持了一场竞拍。他因为背伤只能站着。Google 最终买下了公司（常见说法是 4400 万美元，官方未披露）🗣️。
  - 2025 年，计算机历史博物馆公开了 AlexNet 的原始代码。
- **L34 Dropout（2012）**
  - Hinton 自述灵感来自银行：柜员总在轮换，他意识到这是为了防止员工串通作弊。随机丢掉一部分神经元，就能防止它们「串通」起来过拟合。
  - JMLR 版论文里还拿有性繁殖做了类比。
- **L35–36 Very Deep / Going Deeper（2014）**：GoogLeNet 论文的第一条参考文献是网络梗 *We need to go deeper*（出自电影《盗梦空间》）；GoogLeNet 这个名字则是向 LeNet 致敬。
- **L37–38 The Bar / GAN（2014）**：Goodfellow 在蒙特利尔的 Les 3 Brasseurs 酒吧给朋友庆祝毕业，朋友们正讨论怎么生成照片。他提出让两个网络互相对抗，当晚回家就写了代码，第一次跑就成功了。（文档里写的 *Friday evening* 在主要报道中没有出处，按半虚构处理即可。）
- **L39 Mode Collapse（2014）**：原始 GAN 论文已经提到这个问题，管它叫 *the Helvetica scenario*。
- **L40 BatchNorm（2015）**：原论文说它有效是因为减少了 internal covariate shift。2018 年 Santurkar 等指出这个解释站不住，更可能的原因是它让损失曲面变平滑了。「它有效，但我们说错了原因」，直接为 Science of AI 做了铺垫。
- **L41–42 Too Deep / Residual（2015）**
  - ResNet 论文的 Figure 1 就是 20 层对 56 层的退化曲线，可以直接当关卡画面。
  - 152 层的 ResNet top-5 错误率为 3.57%，低于 Karpathy 亲手标注测出的人类水平 5.1%。可以让玩家自己当这个「人类基线」（§3）。
  - 何恺明是 2003 年广东省高考理科状元。据 Nature 2025 年的统计，ResNet 是 21 世纪被引用最多的论文。
- **L43 Look Once（2015）**：YOLO 的作者 Joseph Redmon 在 2020 年宣布退出计算机视觉研究，因为他无法无视这项技术的军事用途和隐私风险。这是个很有分量的人文结尾。
- **L44–45 Move 37 / Move 78（2016）**
  - AlphaGo 估计人类下出第 37 手的概率只有约万分之一。樊麾看完说：这不是人类会下的棋……太美了。
  - 第四局，李世石的第 78 手被称为「神之一手」，AlphaGo 同样估计它只有约万分之一的概率。这是人类唯一赢下的一局。
  - 李世石 2019 年退役时说，AI 是「无法被打败的存在」。
  - 2017 年，柯洁在乌镇 0:3 负于 AlphaGo Master，比赛中落泪。同年，AlphaGo Zero 不看任何人类棋谱，以 100:0 击败了 AlphaGo Lee（§3 S9）。

### Act V · Architecture Wars（2016–2020）

- **L46 Dense Connections**：故事性偏弱，建议换成 NAS（§3 S8）。
- **L47 Attention（→ 2014）**：Bahdanau 当时是 Bengio 实验室的实习生，灵感来自初中学英语时的翻译练习：翻译时目光在原文和译文之间来回移动。模型原名 *RNNsearch*，*attention* 这个词是 Bengio 在最后几轮改稿时加的。这些细节出自 Karpathy 2024 年 12 月公开的两人通信。
- **L48 Attention Is All You Need（2017）**
  - 标题致敬披头士的 *All You Need Is Love*。
  - *Transformer* 这个名字是 Jakob Uszkoreit 起的，因为他喜欢这个词的发音。
  - 论文脚注写着作者「排序随机」。
  - 八位作者后来都离开了 Google（Shazeer 2024 年又回去了）。其中 Llion Jones 联合创立了 Sakana AI，正是它做出了 *The AI Scientist*，和 L73 同名，可以回收。
- **L49 BERT vs GPT（2018）**：ELMo 和 BERT 开启了「芝麻街」命名潮。2018–2019 年 BERT 在各大榜单上全面领先 GPT-1，当时多数人认为双向更好。这是一场「延迟裁决」的 Battle：玩家的选择可以留到 L57（GPT-3）才揭晓对错。
- **L50 The Bitter Lesson（→ 2019）**：原文第一句大意是：70 年 AI 研究给出的最大教训是，利用算力的通用方法最终最有效，而且优势巨大。Sutton 和 Barto 获得了 2024 年度图灵奖（2025 年公布）。
- **L51 GPT-2（2019）**：2019-02，OpenAI 以「太危险」为由没有放出完整模型，示例是一段「科学家在安第斯山脉发现会说完美英语的独角兽」的新闻。同年 11 月，完整的 15 亿参数版本发布。
- **L52 Lottery Ticket（2019）**：拿了 ICLR 2019 最佳论文。
- **L53 Scaling Laws（2020）**：百度硅谷 AI 实验室的 Hestness 等人 2017 年就系统报告过深度学习的幂律缩放。Dario Amodei 也回忆说，他最早是在百度做语音识别时注意到这个规律的。
- **L55 Double Descent（→ 2019）/ L56 Neural Collapse（2020）**：都属于「训练里出现了教科书解释不了的现象」。建议统一包装成「异常现象档案」，为 L73 的 Science of AI 做铺垫。

### Act VI · Foundation Model 时代（2020–2023）

- **L57 Few-Shot Magic（2020）**：GPT-3 论文获得 NeurIPS 2020 最佳论文。BERT vs GPT 的 Battle 在这里揭晓。
- **L58–60 Grokking（2021）**：据《MIT Technology Review》2024 年的报道，Burda 和 Edwards「意外地」让实验跑得比计划久得多，是几天而不是几小时；回来一看，模型学会了。「去度假」是流传版本 🗣️，文档把 L59 标成半虚构是对的。
- **L61 Induction Head（2021–22）**：出自 Anthropic 2021-12 和 2022-03 的两篇论文。
- **L62 Emergent Abilities（→ 2022）**：可以加一个反转。把评测指标从「完全匹配」换成连续指标后，「突然涌现」的曲线就变平滑了（Schaeffer 等，2023）。让玩家亲手做这次切换，是一个很好的 Experiment。
- **L63 Chinchilla（2022）**：参数 70B、训练 1.4T token 的 Chinchilla 击败了 280B 的 Gopher，得出「每个参数约配 20 个 token」。这和 Kaplan 2020 年的结论正面冲突，适合做成 Battle。
- **L65 RLHF（2022）**
  - 可以闪回到 2017 年：在 Christiano 等人的实验里，人类只需看两段视频、选哪段更像后空翻，大约 900 次比较、不到一小时，就教会了模拟机器人后空翻。
  - InstructGPT 里 13 亿参数的模型，比 1750 亿参数的 GPT-3 更受人类评审偏好。
- **L66 ChatGPT（2022-11-30）**：内部项目名叫 *Chat with GPT-3.5*，发布时的定位是「低调的研究预览」，低调到有些员工都不知道它上线了。5 天后用户破百万。

### Act VII · Open Models, Reasoning & Science（2023–2026）

- **L67 The Open Model（2023）**
  - LLaMA 在 2023-02-24 只对申请的研究者开放，大约一周后权重就在 4chan 上泄露了。
  - 5 月，Google 的内部备忘录 *We Have No Moat, And Neither Does OpenAI* 外泄。
  - 建议把 2025 年 DeepSeek-R1、Qwen 等开源权重也接进这一关。
- **L68 Alpaca（2023）**：用 text-davinci-003 自动生成了 5.2 万条指令数据，总成本不到 600 美元。在线 demo 几天后就因为幻觉和成本问题下线了。
- **L69 Mixture of Experts（2023）**
  - MoE 的想法出自 1991 年 Jacobs、Jordan、Nowlan 和 **Hinton** 的论文。
  - 2017 年，Noam Shazeer（也是 Transformer 作者之一）做出了稀疏门控 MoE。
  - 2023-12，Mistral 只发了一条磁力链接就放出了 Mixtral。
  - DeepSeek-V3 总共 671B 参数，每个 token 只激活 37B。
  - 又一个在地下埋了二十多年的想法。
- **L70 KAN（2024）**：Kolmogorov–Arnold 表示定理证明于 1957 年，和感知机同一年。1989 年，Girosi 和 Poggio 发表了《Kolmogorov 定理是无关的》；35 年后 KAN 挑战了这个判断，是天然的跨时代 Battle。
- **L71 AlphaFold（2020 / 2024）**：真正的「发现瞬间」是 2020-11 的 CASP14。2024 年，Hassabis 和 Jumper（与 David Baker 一起）获得诺贝尔化学奖，和 Hopfield、Hinton 的物理学奖在同一周公布。
- **L72 Reasoning Models（2024–26）**
  - DeepSeek-R1-Zero 训练时，模型自己写下了 *"Wait, wait. Wait. That's an aha moment I can flag here."*
  - **整部游戏的核心奖励是玩家的 Eureka，而这里是模型第一次自己 Eureka。**
  - R1 的论文 2025-09 登上了 Nature 封面，报告的训练成本为 29.4 万美元，也是第一个经过同行评审的主流大模型。
  - 2025 年夏天，AI 在 IMO 拿到了金牌水平；9 月的 ICPC 全球总决赛，OpenAI 的系统 12 题全对，Gemini 做对 10 题。

---

## 3. 建议补充的故事

### S 级：强烈建议加，能补上主线缺口

**S1 · 1770 The Turk（序章 Level 0 · Mystery）**
- 故事：1770 年，Kempelen 造了一台会下国际象棋的「土耳其人」自动机，在欧洲巡演了几十年，据说还和拿破仑下过棋 🗣️。其实柜子里藏着一个真人棋手。
- 玩法：玩家检查机器，推理它是怎么下棋的。Eureka 是：里面有人。
- 回收：Deep Blue（Kasparov 怀疑幕后有人）→ Amazon Mechanical Turk（借用了它的名字，给 ImageNet 做标注）→ RLHF 标注员 → L73：这一次，机器里可能真的没有人。

**S2 · 1951 The Guessing Game（Act I，放在 L2 之后 · Experiment）**
- 故事：1948 年，Shannon 用字母和单词的 n-gram 统计生成了一段「伪英语」。1951 年，他在《Prediction and Entropy of Printed English》里让受试者一个字母一个字母地猜下一个字母，估算出英语每个字母只携带约 0.6–1.3 比特的信息。
- 玩法：玩家亲手猜下一个字母或下一个词，系统实时算熵。揭晓时告诉玩家：你刚才做的，就是 70 年后 GPT 的训练目标。
- 回收：word2vec → GPT → L72，以及 TOKEN 预告片的结尾「下一个词，由你写下」。**这是整部游戏和 LLM 之间最自然的桥。**

**S3 · 1959 The Cat's Eye（Act I · 意外发现型 Eureka）**
- 故事：Hubel 和 Wiesel 在猫的视觉皮层里记录神经元，给它看各种光点都没反应。直到换玻片时，玻片边缘在投影上划出一条线，神经元突然「像机关枪一样」放电。他们由此发现了对方向敏感的「简单细胞」，1981 年获诺贝尔奖。
- 玩法：玩家试遍各种刺激都没反应，一个偶然的「换片」动作触发了 Eureka。接着搭出从简单细胞到复杂细胞的层级。
- 回收：Neocognitron（1980）→ LeNet（L22）→ Google 的猫（L32）。「猫」可以做成贯穿全作的收集线。

**S4 · 2003 A Neural Probabilistic Language Model（Act III · Eureka）**
- 故事：n-gram 的组合多到数不过来，这就是维数灾难。Bengio 等人让每个词变成一个可学习的向量，再用神经网络预测下一个词，第一次把「词向量」和「神经语言模型」放到了一起。
- 玩法：玩家先用计数表预测下一个词，一遇到没见过的组合就崩。把意思相近的词在空间里放近以后，模型就会泛化了。
- 回收：word2vec（2013）、GPT。

**S5 · 2013 King − Man + Woman（Act IV · Experiment）**
- 故事：Mikolov 的 word2vec。原始论文被第一届 ICLR（2013）拒了稿，而那届的接收率约为 70%。十年后，它的后续论文拿到了 NeurIPS 2023 时间检验奖。
- 玩法：在词向量空间里拖动箭头做类比。可以直接复用 Demo 第四章「意义之海」的素材。

**S6 · 2014 Sequence to Sequence（Act IV · Eureka，和挪过来的 L47 组成两段关）**
- 故事：Sutskever、Vinyals、Le 用两个 LSTM 做翻译。有一个看似毫无道理的技巧：把源句倒过来输入，BLEU 就从 25.9 涨到了 30.6。Sutskever 在 NIPS 2014 的报告里说：如果你有一个大数据集，再训练一个很大的神经网络，成功就是有保证的。
- 玩法：先体会把整句话压进一个向量的瓶颈（句子一长就崩），再发现「倒序输入」这个怪招，最后引出 Bahdanau 的注意力（L47）。
- 回收：十年后的 2024 年，同一个人在同一个会议上（Seq2Seq 获时间检验奖的报告）宣布「我们只有一个互联网，预训练将会终结」（见 A 级）。

**S7 · 2013 / 2015 Breakout（Act IV · Eureka，开启 RL 线）**
- 故事：DeepMind 的 DQN 只看屏幕像素学打 Atari 游戏。在 Breakout 里，它自己发现了一种策略：在墙边打出一条隧道，让球在砖墙后面来回弹。2014 年，Google 收购了 DeepMind。
- 玩法：玩家先正常玩一局，再看 AI 的训练回放，自己找出那条「隧道」。
- 回收：TD-Gammon（1992）→ Move 37（L44）→ AlphaGo Zero。

**S8 · 2016 Neural Architecture Search（替换 L46 · Experiment）**
- 故事：
  - Zoph 和 Le 让一个 RNN 控制器通过强化学习去「设计」网络结构，据称用了约 800 块 GPU。
  - 之后是 2017 年的 NASNet、2019 年的 EfficientNet。
  - 2020 年，AutoML-Zero 从最基础的数学运算开始进化，**自己重新发现了用反向传播训练的两层网络**。
- 为什么要加：这是 L73「Design a machine that designs models」最直接的历史伏笔。放在 Act V，结尾就不会显得突兀。
- 回收：L20（反向传播被机器重新发现一遍）、L73。

**S9 · 2017 Tabula Rasa（Act V · Eureka / Battle）**
- 故事：AlphaGo Zero 不看任何人类棋谱，从零开始自我对弈，3 天后以 100:0 击败了赢过李世石的 AlphaGo Lee。同年 5 月，柯洁在乌镇 0:3 落败。
- 玩法：玩家把「人类棋谱」这路输入拔掉，模型反而变强了，这很反直觉。
- 回收：L50 The Bitter Lesson、L72（R1-Zero 也叫「Zero」：不靠人类示范，只靠可以验证的奖励）。

**S10 · 2020–2022 Diffusion（Act VI · Battle + Story）**
- 故事：
  - 2015 年，Sohl-Dickstein 受非平衡热力学启发提出扩散模型，之后沉寂了 5 年。
  - 2020 年出现 DDPM；2021 年有论文直接叫 *Diffusion Models Beat GANs*。
  - 2022-08 Stable Diffusion 开源。差不多同一时间，一幅用 Midjourney 生成的《太空歌剧院》在科罗拉多州博览会的美术比赛中获奖，引发了「AI 画算不算艺术」的争论。
- 玩法：GAN vs Diffusion 的 Battle，回收 L37–39 那三关 GAN。「一点点去噪」的过程本身就很适合水彩风格的画面。

**S11 · 2025 Aha Moment（并入 L72 · Eureka 的主客反转）**
- 细节见 §2 L72。
- 再补一个画面：2025-01 DeepSeek-R1 开源后，英伟达单日市值蒸发约 5900 亿美元，创下美股历史上单日最大跌幅。中国玩家会非常有共鸣。

**S12 · 2025–2026 The Last Invention（L73 前奏 · Story / Mystery）**
- 1965 年，曾在布莱切利园与 Turing 共事的 I. J. Good 写道：第一台超智能机器将是人类需要完成的最后一项发明，因为设计机器本身也是智力活动之一。这段话适合放在 L73 黑屏之前当引言。
- 1843 年，Lovelace 写道：分析机并不自命能创造出任何东西。Turing 1950 年专门反驳过这句话（L2）。L73 可以用它做首尾呼应。
- 2025-05，AlphaEvolve 找到了只用 48 次乘法完成 4×4 复数矩阵乘法的算法，打破了 **Strassen 1969 年**的纪录，而 1969 年正是 XOR 之门关上的那一年。它还把 Gemini 训练用的一个核心算子提速了 23%，让总训练时间缩短了 1%。这是 AI 在给训练自己的流程提速。
- 2025-10，OpenAI 一位高管发推称 GPT-5「解决了 10 个未解的 Erdős 问题」。维护 Erdős 问题网站的数学家 Thomas Bloom 指出，它只是找到了一些他本人不知道的已有论文；Hassabis 回了一句 *"this is embarrassing"*。**检索不等于发现**，正好接上 Demo 里的 RAG 章节。
- 2026-05-20，OpenAI 宣布它的内部推理模型自主反证了 Erdős 1946 年提出的单位距离猜想。Alon、Bloom、Gowers 等 9 位数学家为此写了配套论文，逐步核验证明。
- 2024-08，Sakana AI 发布了 *The AI Scientist*。2025 年，它的 v2 版写出的一篇论文通过了 ICLR 2025 研讨会的双盲评审（实验事先和组委会约定好，并在评审后撤稿）。
  - 注意：L73 和这个系统同名，而 Sakana 的联合创始人 Llion Jones 就是 Transformer 的作者之一。可以当成致敬，也可以考虑改名以免混淆。

### A 级：推荐，可以做成关卡或过场

| 年份 | 故事 | 建议位置 · 类型 | 亮点 |
|---|---|---|---|
| 1956 | Logic Theorist | 并入 L5，或单独成关 · Eureka | Simon 1956 年 1 月对学生说：「圣诞节期间，Newell 和我发明了一台会思考的机器。」它证明了《数学原理》第二章前 52 个定理中的 38 个，其中定理 2.85 的证明比原书更优雅。罗素本人看了很高兴，《符号逻辑杂志》却以「初等定理的新证明不值得发表」拒了稿。可回收 L1（Pitts 读《数学原理》）和 2025 年的 IMO 金牌 |
| 1980 | Searle 的中文屋 | Act II · Battle | 对中国玩家有天然的反讽张力：Searle 选中文，恰恰因为他一个字都不懂。可串成一条线：ELIZA（L9）→ 中文屋 → 「随机鹦鹉」（2021）→ LaMDA（2022） |
| 1984 | Cyc，以及「AI 寒冬」一词诞生 | Act II · Story | Lenat 花了近 40 年手写常识规则，2023 年去世。可以和 GPT-3 从网页里「读」出常识做对比，回收 L50 |
| 1986 | NETtalk | 放在 L20 之后 · Experiment | Sejnowski 的网络学习朗读英文，录音从婴儿般的咿呀声慢慢变成清晰的单词，正好发挥你们的程序化音频 |
| 1989 / 1995 / 2005 | ALVINN → No Hands Across America → DARPA 挑战赛 | Act III · Story | 1989 年 ALVINN 用神经网络学开车。1995 年同一个 CMU 实验室的 Navlab 5 从匹兹堡开到圣迭戈，98% 的路程是自动转向（转向由视觉系统 RALPH 负责）。2004 年 DARPA 挑战赛全军覆没，走得最远的车只开了 7 英里多；2005 年 Stanley 夺冠。这一块补的是机器人和自动驾驶的空白 |
| 1992 | TD-Gammon | Act III · Experiment | Tesauro 的网络靠自我对弈把西洋双陆棋下到了世界级水平，还改变了人类高手的开局理论。AI 教人类下棋，比 AlphaGo 早了 24 年 |
| 1994 / 2007 | Chinook 与 Marion Tinsley | Act III · Story | Tinsley 在 45 年里只输过 7 盘棋，其中 2 盘输给 Chinook。1994 年的人机对决下到第 6 盘（全是和棋）时，他因病退赛，随即确诊胰腺癌，7 个月后去世。2007 年，跳棋被「解决」了，回收 L4 的 Samuel |
| 2004 / 2007 | ESP Game 与 reCAPTCHA | Act III · Story / 小游戏 | von Ahn 让人们在游戏里免费给图片打标签（Google 后来把它做成了 Image Labeler），又让人们在验证码里帮着数字化图书（reCAPTCHA 2009 年被 Google 收购）。「你正在玩的游戏，本身就在给 AI 标注数据」，是一个很妙的 meta 时刻 |
| 2006–2009 | Netflix Prize | Act III · Story / Experiment | 奖金 100 万美元。最后两支队伍成绩相同，冠军只比对手早交了 20 分钟。这是集成学习的胜利；后续比赛因为隐私问题被取消 |
| 2009–2012 | 语音识别的深度学习革命 | Act IV · Story | 比 AlexNet 更早落地工业界。2012-10，微软研究院的 Rick Rashid 在天津演讲，他的英文被实时识别、翻译，再用他自己的声音说成中文，全场鼓掌。中国玩家的共鸣点 |
| 2014 / 2015 | 「你就是人类基线」 | 并入 L42 · 机制 | Karpathy 亲手标注 ImageNet，测出人类的 top-5 错误率约 5.1%。让玩家也标一轮，得到自己的错误率，然后看 ResNet 把它超过去 |
| 2016-03 | Tay vs 小冰 | 放在 L44–45 附近 · Story | 和 AlphaGo 同一个月，微软的 Tay 上线 16 小时就被网友教坏下线；而它的「姐姐」小冰从 2014 年起就在中国很受欢迎 |
| 2016-11 | Google 翻译一夜变强 | Act V 开头 · Story | 东京大学教授暦本纯一发现，Google 翻译几乎一夜之间能流畅翻译海明威《乞力马扎罗的雪》的开头了。回收 L10，跨度整整 50 年 |
| 2017-12 | Rahimi：「机器学习已经变成了炼金术」 | Act V · Battle | NIPS 时间检验奖演讲，对上 LeCun 的反驳：严谨对经验。这是 Science of AI 的开场辩论 |
| 2020 | RAG（Lewis 等） | Act VI · Eureka | 不必把一切都记进权重，而是「开卷考试」。回收 Watson（L31）。作者后来说，要是知道它会这么流行，当初就会起个好听点的名字。和 Demo 第八章一致 |
| 2021-01 | CLIP 与 DALL·E | Act VI · Experiment | 「牛油果形状的扶手椅」，补上多模态的空白 |
| 2022-06 | LaMDA「有意识」事件 | Act VI · Story | Google 工程师 Blake Lemoine 公开声称聊天模型有感知能力，随后被解雇。这是 ELIZA 效应 56 年后的回声 |
| 2023-05 | Hinton 离开 Google | Act VII · Story | 花了 50 年让神经网络成功的人，离开是为了能自由地谈论它的风险 |
| 2024-10 | 诺贝尔周 | Act VII · Story | 同一周里，物理学奖（Hopfield、Hinton）和化学奖（Hassabis、Jumper、Baker）都颁给了 AI 相关的工作。Hinton 接到电话时，人在加州一家廉价旅馆。一次回收 L19、L20、L28、L71 |
| 2024-12 | Ilya：「数据是 AI 的化石燃料」 | Act VII · Story | 在 NeurIPS 的 Seq2Seq 时间检验奖报告里，他说我们只有一个互联网，预训练将会终结。这是 Data 线的终点，也是推理模型（L72）的起点 |
| 2025-07 / 09 | IMO 金牌 / ICPC 12 题全对 | Act VII · Story | 从「机器能思考吗？」（L2）走到奥赛金牌 |

### B 级：适合做资料卡、收集品或彩蛋

- **1949 Hebb 学习规则**。注意：*"neurons that fire together wire together"* 这句话不是 Hebb 说的，是 1992 年 Carla Shatz 的概括。
- **1980 Neocognitron**（福岛邦彦，NHK）。
- **2001 Banko & Brill《数据比算法更重要》/ 2009《数据的不合理有效性》**：适合放进 L29 的揭晓环节。
- **2013 年的 TPU 来历**：Google 估算，如果每人每天用 3 分钟语音搜索，数据中心的算力就得翻倍，于是用 15 个月造出了 TPU。
- **2014 Adam 优化器**。
- **2015 DeepDream**。
- **2019 年公布的 2018 年度图灵奖**（Hinton、LeCun、Bengio）：三位「地下火种」终于被承认，回收 Act II。
- **2019 Bengio 与 Marcus 的蒙特利尔辩论**：可以并入 L50。
- **2021「随机鹦鹉」论文**：做 Battle 时要注意平衡呈现双方观点。
- **2021 Copilot**：AI 写代码，往后就是 AI 写 AI。
- **2022 AlphaTensor、2023 AlphaDev 和 FunSearch**：可以和 S12 串成「AI 发现算法」一条线。
- **2023 Lion 优化器**：是用程序搜索找到的。
- **2025 Muon 与 Kimi K2 的 MuonClip**：Muon 出自一个开放的 NanoGPT 速通竞赛。
- **2019 OpenAI Five、AlphaStar、Pluribus**。
- **2014 DeepFace / GaussianFace**：人脸识别在 LFW 数据集上逼近乃至超过人眼。

---

## 4. 跨关卡 callback 与隐藏主线

文档已经提出了 Architecture（深度）、Data、Compute 三条线，外加最终的「四个旋钮」。在此基础上建议再补下面几条：

| 主线 | 节点（★ 为新增） | 最后怎么回收 |
|---|---|---|
| 机器里藏着人 | ★ The Turk 1770 → L26 Deep Blue → L29 Mechanical Turk 标注 → L65 RLHF 标注员 | L73：机器里第一次可能没有人 |
| 下一个词 | ★ Shannon 1951 → ★ Bengio 2003 → ★ word2vec → ★ Seq2Seq → L49 / L57 GPT → L72 | 片尾「下一个词，由你写下」（TOKEN 预告片的结尾） |
| 翻译 | L10（1954 / 1966）→ ★ Bar-Hillel 的 pen → L47（2014）→ ★ GNMT 2016 → L48（第一个任务就是翻译） | 机器翻译的失败引来第一个寒冬，机器翻译的成功催生了 Transformer |
| 猫 | ★ Hubel & Wiesel 1959 → L22 LeNet → L32 Google 的猫 → L33 ImageNet 里的猫类 | 可做收集品：每一幕藏一只猫 |
| 棋盘 | L4 跳棋 → L26 Deep Blue → ★ TD-Gammon → ★ Chinook 解决跳棋 → L44–45 → ★ AlphaGo Zero | 李世石退役；「AI 教人类下棋」 |
| 标签从哪来（贴合监督学习） | L17–18 专家写规则 → L27 / L29 众包标注（★ ESP Game）→ 文本自己就是标签（Shannon → word2vec → GPT）→ L65 人类偏好 → L72 可验证奖励 | L73：AI 自己设计实验，自己产生数据 |
| XOR 与 1969 | L12 XOR → L20 Backprop 重开 XOR 之门 → ★ AlphaEvolve 打破 1969 年的 Strassen 纪录 | 1969 年同时发生了：XOR 之门关上、Pitts 与 McCulloch 去世、Strassen 算法问世 |
| 诺贝尔周 | L19 Hopfield、L20 / L28 Hinton、L71 AlphaFold | ★ 2024 诺贝尔周，一关里召回多个老角色 |
| 炒作与寒冬 | L7《纽约时报》1958 → L11《Life》1970 → L15 / L16 → L21 → L31 Watson Health → L51 GPT-2 → ★ 2025-10 Erdős 乌龙 | 给「对 AI 焦虑的人」一种尺度感，也就是文档里说的人文感 |
| 谁拥有 Eureka | 每一关都是玩家的 Eureka → L72 模型的 aha moment → ★ Lovelace 1843 与 I. J. Good 1965 | L73：Design a machine that designs models |

---

## 5. 名额从哪来（如果坚持 72 关）

- **合并：**
  - L35 + L36（都是 2014 年 ILSVRC 的「更深」）；
  - L38 + L39（mode collapse 当 GAN 关的第二阶段）；
  - L54 + L64（CNN vs ViT 打两个回合的 Battle）。
- **替换：** L46 DenseNet 换成 NAS（S8）。
- **重排：** L47 改成 2014，和新增的 Seq2Seq 合成一个两段关。
- **并入现有关卡、不占名额：**
  - S11 并入 L72；
  - S12 做 L73 的前奏；
  - 1770 The Turk 可以像 L73 一样，作为不计入编号的序章。
- **算一下：** 这样能腾出 3 个位置。剩下的 S 级新关有 Shannon、Hubel & Wiesel、Bengio 2003、word2vec、DQN、AlphaGo Zero、Diffusion，共 7 个。
  - 如果把 Bengio 2003 做成 word2vec 关的前奏，把 AlphaGo Zero 做成 L45 的尾声，总数约为 74 关。
  - 72 这个数字本身不必死守。

---

## 6. 试玩版 7 关建议（每幕 1 关）

| 幕 | 首选 | 类型 | 理由 | 备选 |
|---|---|---|---|---|
| I | **L12 XOR**（L13 做关尾过场） | Experiment（注定失败） | 「你没有输，是这个模型根本做不到」，同时给第二幕埋下 callback | L9 ELIZA |
| II | **L20 Backprop** | Eureka | 文档自己点名的第一个 Eureka 关，XOR 之门在这里重开 | L15 Lighthill 辩论（Battle） |
| III | **L29 ImageNet** | Story + 轻实验 | 玩家先当标注员（ESP Game / MTurk 式小游戏），再发现真正缺的是数据，从而引出 Data 线 | L25 LSTM（和 Demo 第三章的素材复用度高） |
| IV | **L37 → L38 The Bar → GAN** | Story → Experiment | 文档点名的「酒吧故事关 → 公寓实验关」，L39 可以当第二阶段 | L41 → L42 ResNet（何恺明，中国观众共鸣强） |
| V | **L48 Attention Is All You Need**（用 2014 年的 L47 做回忆开场） | Eureka | 删掉 recurrence 那一下；结尾给出 BERT / GPT 两扇门，预告下一场 Battle。可以复用 Demo 的「色彩绽放」 | L49 BERT vs GPT（纯 Battle） |
| VI | **L58–60 Grokking 压成一关** | Mystery / Experiment | 最能体现「经历而不是学习」 | L65 → L66 RLHF → ChatGPT（大众认知度最高） |
| VII | **L72 Reasoning**（R1 的 aha moment），再接 L73 黑屏预告 | Eureka → Story | Eureka 从玩家转到了模型身上，直接引出 Science of AI | L67 Open Model（中国开源模型） |

- **玩法覆盖：** 首选这一组把 Experiment、Eureka、Story、Mystery 都覆盖到了，Battle 只在第五幕的结尾出现。如果试玩版需要一关完整的 Battle，可以把第五幕换成 L49，或把第三幕换成 L24（带上 Jackel–Vapnik 的赌约）。
- **美术复用：** XOR / 感知机、反向传播、注意力、RLHF 这几关，TOKEN 预告片里已经有对应的画面语言，能省下不少美术时间。

---

## 7. 使用时要标注的传闻和简化说法

- **"The spirit is willing but the flesh is weak" → "The vodka is good but the meat is rotten"**：机器翻译史上流传最广的笑话，但没有证据真的发生过。如果用，就明确说它是传说。反过来揭穿「这是个传说」，本身也能做成一个 Mystery。
- **Grokking 的「度假」**：一手报道只说实验被「意外地」多跑了几天。
- **Goodfellow 的「周五晚上」**：主要报道里没有这个细节。
- **「《Perceptrons》杀死了神经网络」**：这是简化的说法。Minsky 和 Papert 知道多层网络能表示 XOR；资助方向转变也有多方面的原因。
- **Deep Blue 第 44 步的 bug**：来自 Murray Campbell 的转述，它到底对 Kasparov 影响有多大，至今有争议。
- **DNNresearch 的收购价、Samuel 让 IBM 股价大涨、LeNet 处理了一成支票、土耳其傀儡和拿破仑下棋**：都是广为流传但细节不一的说法。
- **Hebb 的名言**：*"Fire together, wire together"* 不是 Hebb 本人的原话。

---

## 8. 参考来源

**本次联网核实过的：**
- Grokking：[MIT Technology Review (2024)](https://www.technologyreview.com/2024/03/04/1089403/large-language-models-amazing-but-nobody-knows-why/) · [Power et al., MATH-AI @ ICLR 2021](https://mathai-iclr.github.io/papers/papers/MATHAI_29_paper.pdf)
- 注意力的起源：[Kseniase 对 Karpathy 与 Bahdanau 通信的整理](https://huggingface.co/posts/Kseniase/505294090801530) · [Turing Post](https://www.turingpost.com/p/attention)
- word2vec 被 ICLR 2013 拒稿：[Hacker News 转载的 Mikolov 原帖](https://news.ycombinator.com/item?id=38654056) · [Jeff Dean](https://x.com/JeffDean/status/1734720190401634474?lang=en)
- Transformer 的命名：[Wikipedia: Attention Is All You Need](https://en.wikipedia.org/wiki/Attention_Is_All_You_Need)（引自 Wired 2024 年 Steven Levy 的长篇报道）
- DeepSeek-R1 登上 Nature：[Nature 论文](https://www.nature.com/articles/s41586-025-09422-z) · [Nature 新闻](https://www.nature.com/articles/d41586-025-03015-6)
- ICPC 2025：[Google DeepMind](https://deepmind.google/blog/gemini-achieves-gold-medal-level-at-the-international-collegiate-programming-contest-world-finals/) · [TechRepublic](https://www.techrepublic.com/article/openai-deepmind-icpc-2025-results/)
- Jackel–Vapnik 赌约：[Yury Kashnitsky](https://yorko.github.io/2022/vapnik-jackel-bet/)
- Mark I 感知机：[Wikipedia: Mark I Perceptron](https://en.wikipedia.org/wiki/Mark_I_Perceptron)
- Widrow 与 Hoff：[Wikipedia: Bernard Widrow](https://en.wikipedia.org/wiki/Bernard_Widrow)
- Dropout 的银行灵感：[neuralthreads](https://neuralthreads.medium.com/dropout-regularization-technique-that-clicked-in-geoffrey-hintons-mind-at-a-bank-fa7fa8c5e1fb)
- AlphaEvolve：[Google DeepMind](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/)
- The AI Scientist 通过评审：[Sakana AI](https://sakana.ai/ai-scientist-first-publication/)
- Erdős 乌龙（2025-10）：[TechCrunch](https://techcrunch.com/2025/10/19/openais-embarrassing-math/)
- 单位距离猜想被反证（2026-05）：[OpenAI](https://openai.com/index/model-disproves-discrete-geometry-conjecture/) · [arXiv 2605.20695](https://arxiv.org/abs/2605.20695) · [Gil Kalai 的博客](https://gilkalai.wordpress.com/2026/05/21/amazing-erdos-unit-distance-problem-was-disproved-it-was-achieved-by-ai/)
- 从人类偏好中学习（后空翻）：[OpenAI (2017)](https://openai.com/index/learning-from-human-preferences/)
- Logic Theorist：[Wikipedia: Logic Theorist](https://en.wikipedia.org/wiki/Logic_Theorist)
- Deep Blue：[Wikipedia: Deep Blue](https://en.wikipedia.org/wiki/Deep_Blue_(chess_computer)) · [NBC News](https://www.nbcnews.com/tech/tech-news/glitch-may-have-helped-supercomputer-beat-chess-champ-historic-match-flna6206532)
- AlexNet：[Wikipedia: AlexNet](https://en.wikipedia.org/wiki/AlexNet) · [CHM 公开 AlexNet 源码](https://computerhistory.org/blog/chm-releases-alexnet-source-code/) · [TechCrunch：DNNresearch 被收购](https://techcrunch.com/2013/03/12/google-scoops-up-neural-networks-startup-dnnresearch-to-boost-its-voice-and-image-search-tech)
- Navlab 与 No Hands Across America：[Wikipedia: Navlab](https://en.wikipedia.org/wiki/Navlab)
- Marion Tinsley：[Wikipedia: Marion Tinsley](https://en.wikipedia.org/wiki/Marion_Tinsley)
- RAG 的命名：[NVIDIA Blog](https://blogs.nvidia.com/blog/what-is-retrieval-augmented-generation/) · [Lewis et al. 2020](https://arxiv.org/abs/2005.11401)
- ChatGPT 的发布细节：[Yahoo Tech](https://tech.yahoo.com/ai/articles/chatgpt-just-turned-one-even-060127893.html)

**其余内容依据公认的原始文献和常见史料：** McCulloch & Pitts 1943；Turing 1950；Shannon 1948 / 1951；Rosenblatt 1958；Minsky & Papert 1969；Rumelhart, Hinton & Williams 1986；Hochreiter & Schmidhuber 1997；Gers et al. 2000；Bengio et al. 2003；Mikolov et al. 2013；Sutskever et al. 2014；Bahdanau et al. 2014；Vaswani et al. 2017；Sutton 2019；Kaplan et al. 2020；Hoffmann et al. 2022；Wei et al. 2022；Schaeffer et al. 2023；DeepSeek-AI 2025，以及 Amanda Gefter 在 Nautilus 上写 Pitts 的文章、Cade Metz《Genius Makers》、Nate Silver《信号与噪声》、李飞飞《我看见的世界》等。
