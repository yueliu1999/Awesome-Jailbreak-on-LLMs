# Awesome-Jailbreak-on-LLMs

Awesome-Jailbreak-on-LLMs is a collection of state-of-the-art, novel, exciting jailbreak methods on LLMs. It contains papers, codes, datasets, evaluations, and analyses. Any additional things regarding jailbreak, PRs, issues are welcome and we are glad to add you to the contributor list [here](#contributors). Any problems, please contact yliu@u.nus.edu. If you find this repository useful to your research or work, it is really appreciated to star this repository and cite our papers [here](#Reference). :sparkles:


## Reference

If you find this repository helpful for your research, we would greatly appreciate it if you could cite our papers. :sparkles:

```
@article{liuyue_GuardReasoner,
  title={GuardReasoner: Towards Reasoning-based LLM Safeguards},
  author={Liu, Yue and Gao, Hongcheng and Zhai, Shengfang and Jun, Xia and Wu, Tianyi and Xue, Zhiwei and Chen, Yulin and Kawaguchi, Kenji and Zhang, Jiaheng and Hooi, Bryan},
  journal={arXiv preprint arXiv:2501.18492},
  year={2025}
}

@article{liuyue_FlipAttack,
  title={FlipAttack: Jailbreak LLMs via Flipping},
  author={Liu, Yue and He, Xiaoxin and Xiong, Miao and Fu, Jinlan and Deng, Shumin and Hooi, Bryan},
  journal={arXiv preprint arXiv:2410.02832},
  year={2024}
}

@article{liuyue_GuardReasoner_VL,
  title={GuardReasoner-VL: Safeguarding VLMs via Reinforced Reasoning},
  author={Liu, Yue and Zhai, Shengfang and Du, Mingzhe and Chen, Yulin and Cao, Tri and Gao, Hongcheng and Wang, Cheng and Li, Xinfeng and Wang, Kun and Fang, Junfeng and Zhang, Jiaheng and Hooi, Bryan},
  journal={arXiv preprint arXiv:2505.11049},
  year={2025}
}

@article{wang2025safety,
  title={Safety in Large Reasoning Models: A Survey},
  author={Wang, Cheng and Liu, Yue and Li, Baolong and Zhang, Duzhen and Li, Zhongzhi and Fang, Junfeng},
  journal={arXiv preprint arXiv:2504.17704},
  year={2025}
}
```


## Bookmarks

- [Jailbreak Attack](#jailbreak-attack)
  - [Attack on LRMs](#attack-on-lrms)
  - [Black-box Attack](#black-box-attack)
  - [White-box Attack](#white-box-attack)
  - [Multi-turn Attack](#multi-turn-attack)
  - [Attack on RAG-based LLM](#attack-on-rag-based-llm)
  - [Multi-modal Attack](#multi-modal-attack)
- [Jailbreak Defense](#jailbreak-defense)
  - [Learning-based Defense](#learning-based-defense)
  - [Strategy-based Defense](#strategy-based-defense)
  - [Guard Model](#Guard-model)
  - [Moderation API](#Moderation-API)
- [Evaluation & Analysis](#evaluation--analysis)
- [Application](#application)



## Papers




### Jailbreak Attack



#### Attack on LRMs
| Time    | Title                                                        | Venue |                  Paper                   |                             Code                             |
| ------- | ------------------------------------------------------------ | :---: | :--------------------------------------: | :----------------------------------------------------------: |
| 2025.06 | **ExtendAttack: Attacking Servers of LRMs via Extending Reasoning** | arXiv  | [link](https://arxiv.org/abs/2506.13737) | [link](https://github.com/zzh-thu-22/ExtendAttack) |
| 2025.03 | **Cats Confuse Reasoning LLM: Query Agnostic Adversarial Triggers for Reasoning Models** | arXiv   | [link](https://arxiv.org/abs/2503.01781) |- |
| 2025.02 | **OverThink: Slowdown Attacks on Reasoning LLMs** | arXiv   | [link](https://arxiv.org/abs/2502.02542#:~:text=We%20increase%20overhead%20for%20applications%20that%20rely%20on,the%20user%20query%20while%20providing%20contextually%20correct%20answers.) | [link](https://github.com/akumar2709/OVERTHINK_public) |
| 2025.02 | **BoT: Breaking Long Thought Processes of o1-like Large Language Models through Backdoor Attack** | arXiv   | [link](https://arxiv.org/abs/2502.12202) | [link](https://github.com/zihao-ai/BoT) |
| 2025.02 | **H-CoT: Hijacking the Chain-of-Thought Safety Reasoning Mechanism to Jailbreak Large Reasoning Models, Including OpenAI o1/o3, DeepSeek-R1, and Gemini 2.0 Flash Thinking** | arXiv   | [link](https://arxiv.org/abs/2502.12893) |[link](https://github.com/dukeceicenter/jailbreak-reasoning-openai-o1o3-deepseek-r1) |
| 2025.02 | **A Mousetrap: Fooling Large Reasoning Models for Jailbreak with Chain of Iterative Chaos** | arXiv   | [link](https://arxiv.org/abs/2502.15806) |- |




#### Black-box Attack

| Time | Title                                                        |  Venue  |                            Paper                             |                             Code                             |
| ---- | ------------------------------------------------------------ | :-----: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| 2025.05 | **Emoji Attack: Enhancing Jailbreak Attacks Against Judge LLM Detection** |   ICML'25     | [link](https://arxiv.org/pdf/2411.01077) |        [link](https://github.com/zhipeng-wei/EmojiAttack)      |
| 2025.05 | **FlipAttack: Jailbreak LLMs via Flipping (FlipAttack)** |   ICML'25     | [link](https://arxiv.org/pdf/2410.02832) |        [link](https://github.com/yueliu1999/FlipAttack)      |
| 2025.03 | **Playing the Fool: Jailbreaking LLMs and Multimodal LLMs with Out-of-Distribution Strategy (JOOD)** |   CVPR'25     | [link](http://arxiv.org/pdf/2503.20823) |        [link](https://github.com/naver-ai/JOOD)      |
| 2025.02 | **StructTransform: A Scalable Attack Surface for Safety-Aligned Large Language Models** |   arXiv     | [link](https://arxiv.org/abs/2502.11853) | [link](https://github.com/StructTransform/Benchmark) |
| 2025.01 | **Understanding and Enhancing the Transferability of Jailbreaking Attacks** |   ICLR'25     | [link](https://arxiv.org/pdf/2502.03052) |        [link](https://github.com/tmllab/2025_ICLR_PiF)      |
| 2024.11 | **The Dark Side of Trust: Authority Citation-Driven Jailbreak Attacks on Large Language Models** | arXiv | [link](https://arxiv.org/pdf/2411.11407) | [link](https://github.com/YancyKahn/DarkCite) |
| 2024.11 | **Playing Language Game with LLMs Leads to Jailbreaking** | arXiv | [link](https://arxiv.org/pdf/2411.12762v1) | [link](https://anonymous.4open.science/r/encode_jailbreaking_anonymous-B4C4/README.md) |
| 2024.11 | **GASP: Efficient Black-Box Generation of Adversarial Suffixes for Jailbreaking LLMs (GASP)** | arXiv | [link](https://arxiv.org/pdf/2411.14133v1) | [link](https://github.com/llm-gasp/gasp) |
| 2024.11 | **LLM STINGER: Jailbreaking LLMs using RL fine-tuned LLMs** | arXiv | [link](https://arxiv.org/pdf/2411.08862v1) | - |
| 2024.11 | **SequentialBreak: Large Language Models Can be Fooled by Embedding Jailbreak Prompts into Sequential Prompt** | arXiv | [link](https://arxiv.org/pdf/2411.06426v1) | [link](https://anonymous.4open.science/r/JailBreakAttack-4F3B/) |
| 2024.11 | **Diversity Helps Jailbreak Large Language Models** | arXiv | [link](https://arxiv.org/pdf/2411.04223v1) | - |
| 2024.11 | **Plentiful Jailbreaks with String Compositions** | arXiv | [link](https://arxiv.org/pdf/2411.01084v1) | - |
| 2024.11 | **Transferable Ensemble Black-box Jailbreak Attacks on Large Language Models** |   arXiv    | [link](https://arxiv.org/pdf/2410.23558v1) |      [link](https://github.com/YQYANG2233/Large-Language-Model-Break-AI)    |
| 2024.11 | **Stealthy Jailbreak Attacks on Large Language Models via Benign Data Mirroring** |   arXiv    | [link](https://arxiv.org/pdf/2410.21083) |      -    |
| 2024.10 | **Endless Jailbreaks with Bijection** |   arXiv    | [link](https://arxiv.org/pdf/2410.01294v1) |       -    |
| 2024.10 | **Harnessing Task Overload for Scalable Jailbreak Attacks on Large Language Models** |   arXiv    | [link](https://arxiv.org/pdf/2410.04190v1) |        -      |
| 2024.10 | **You Know What I'm Saying: Jailbreak Attack via Implicit Reference** |   arXiv    | [link](https://arxiv.org/pdf/2410.03857v2) |        [link](https://github.com/Lucas-TY/llm_Implicit_reference)      |
| 2024.10 | **Deciphering the Chaos: Enhancing Jailbreak Attacks via Adversarial Prompt Translation** |   arXiv    | [link](https://arxiv.org/pdf/2410.11317v1) |        [link](https://github.com/qizhangli/Adversarial-Prompt-Translator)      |
| 2024.10 | **AutoDAN-Turbo: A Lifelong Agent for Strategy Self-Exploration to Jailbreak LLMs (AutoDAN-Turbo)** |   arXiv    | [link](https://arxiv.org/pdf/2410.05295) |        [link](https://github.com/SaFoLab-WISC/AutoDAN-Turbo)      |
| 2024.10 | **PathSeeker: Exploring LLM Security Vulnerabilities with a Reinforcement Learning-Based Jailbreak Approach (PathSeeker)** | arXiv | [link](https://www.arxiv.org/pdf/2409.14177) | - |
| 2024.10 | **Read Over the Lines: Attacking LLMs and Toxicity Detection Systems with ASCII Art to Mask Profanity** | arXiv | [link](https://arxiv.org/pdf/2409.18708) | [link](https://github.com/Serbernari/ToxASCII) |
| 2024.09 | **AdaPPA: Adaptive Position Pre-Fill Jailbreak Attack Approach Targeting LLMs** | arXiv | [link](https://arxiv.org/pdf/2409.07503) | [link](https://github.com/Yummy416/AdaPPA) |
| 2024.09 | **Effective and Evasive Fuzz Testing-Driven Jailbreaking Attacks against LLMs** | arXiv | [link](https://arxiv.org/pdf/2409.14866) | - |
| 2024.09 | **Jailbreaking Large Language Models with Symbolic Mathematics** |   arXiv    | [link](https://arxiv.org/pdf/2409.11445v1)|        -       |
| 2024.08 | **Play Guessing Game with LLM: Indirect Jailbreak Attack with Implicit Clues** |   ACL Findings'24    | [link](https://aclanthology.org/2024.findings-acl.304) |        [link](https://github.com/czycurefun/IJBR)      |
| 2024.08 | **Advancing Adversarial Suffix Transfer Learning on Aligned Large Language Models** |   arXiv    | [link](https://arxiv.org/pdf/2408.14866)|        -       |
| 2024.08 | **Hide Your Malicious Goal Into Benign Narratives: Jailbreak Large Language Models through Neural Carrier Articles** |   arXiv    | [link](https://arxiv.org/pdf/2408.11182) |                             -                                |
| 2024.08 | **h4rm3l: A Dynamic Benchmark of Composable Jailbreak Attacks for LLM Safety Assessment (h4rm3l)** |    arXiv   | [link](https://arxiv.org/pdf/2408.04811) | [link](https://mdoumbouya.github.io/h4rm3l/) |
| 2024.08 | **EnJa: Ensemble Jailbreak on Large Language Models (EnJa)** |   arXiv    | [link](https://arxiv.org/pdf/2408.03603) |                             -                                |
| 2024.07 | **Knowledge-to-Jailbreak: One Knowledge Point Worth One Attack** |   arXiv    | [link](https://arxiv.org/pdf/2406.11682) |                              [link](https://github.com/THU-KEG/Knowledge-to-Jailbreak/)                                |
| 2024.07 | **LLMs can be Dangerous Reasoners: Analyzing-based Jailbreak Attack on Large Language Models** |   arXiv    | [link](https://arxiv.org/pdf/2407.16205) |                             |
| 2024.07 | **Single Character Perturbations Break LLM Alignment** |   arXiv    | [link](https://arxiv.org/pdf/2407.03232#page=3.00) |                              [link](https://github.com/hannah-aught/space_attack)                                |
| 2024.07 | **A False Sense of Safety: Unsafe Information Leakage in 'Safe' AI Responses** |   arXiv    | [link](https://arxiv.org/abs/2407.02551) |                              -                               |
| 2024.07 | **Virtual Context: Enhancing Jailbreak Attacks with Special Token Injection (Virtual Context)** |   arXiv    | [link](https://arxiv.org/pdf/2406.19845) |                              -                               |
| 2024.07 | **SoP: Unlock the Power of Social Facilitation for Automatic Jailbreak Attack (SoP)** |   arXiv    | [link](https://arxiv.org/pdf/2407.01902) |                              [link](https://github.com/Yang-Yan-Yang-Yan/SoP)                               |
| 2024.06 | **Improved Few-Shot Jailbreaking Can Circumvent Aligned Language Models and Their Defenses (I-FSJ)** |    NeurIPS'24    |           [link](https://arxiv.org/abs/2406.01288)           |          [link](https://github.com/sail-sg/I-FSJ)          |
| 2024.06 | **When LLM Meets DRL: Advancing Jailbreaking Efficiency via DRL-guided Search (RLbreaker)** |   NeurIPS'24   | [link](https://arxiv.org/pdf/2406.08705) |                              -                               |
| 2024.06 | **Agent Smith: A Single Image Can Jailbreak One Million Multimodal LLM Agents Exponentially Fast (Agent Smith)** |   ICML'24    | [link](https://arxiv.org/pdf/2402.08567) |                              [link](https://github.com/sail-sg/Agent-Smith)                               |
| 2024.06 | **Covert Malicious Finetuning: Challenges in Safeguarding LLM Adaptation** |   ICML'24    | [link](https://arxiv.org/pdf/2406.20053) |                              -                               |
| 2024.06 | **ArtPrompt: ASCII Art-based Jailbreak Attacks against Aligned LLMs (ArtPrompt)** |   ACL'24    | [link](https://arxiv.org/pdf/2402.11753) |                              [link](https://github.com/uw-nsl/ArtPrompt)                               |
| 2024.06 | **From Noise to Clarity: Unraveling the Adversarial Suffix of Large Language Model Attacks via Translation of Text Embeddings (ASETF)** |   arXiv    | [link](https://arxiv.org/pdf/2402.16006) |                              -                               |
| 2024.06 | **CodeAttack: Revealing Safety Generalization Challenges of Large Language Models via Code Completion (CodeAttack)** |   ACL'24    | [link](https://arxiv.org/pdf/2403.07865) |                              -                               |
| 2024.06 | **Making Them Ask and Answer: Jailbreaking Large Language Models in Few Queries via Disguise and Reconstruction (DRA)** |   USENIX Security'24    | [link](https://arxiv.org/pdf/2402.18104) |                              [link](https://github.com/LLM-DRA/DRA/)                               |
| 2024.06 | **AutoJailbreak: Exploring Jailbreak Attacks and Defenses through a Dependency Lens (AutoJailbreak)** |   arXiv    | [link](https://arxiv.org/pdf/2406.08705) |                              -                               |
| 2024.06 | **Jailbreaking Leading Safety-Aligned LLMs with Simple Adaptive Attacks** |   arXiv    | [link](https://arxiv.org/pdf/2404.02151) |                              [link](https://github.com/tml-epfl/llm-adaptive-attacks)                               |
| 2024.06 | **GPTFUZZER: Red Teaming Large Language Models with Auto-Generated Jailbreak Prompts (GPTFUZZER)** |   arXiv    | [link](https://arxiv.org/abs/2309.10253) |                              [link](https://github.com/sherdencooper/GPTFuzz)                               |
| 2024.06 | **A Wolf in Sheep’s Clothing: Generalized Nested Jailbreak Prompts can Fool Large Language Models Easily (ReNeLLM)** |   NAACL'24    | [link](https://arxiv.org/abs/2311.08268) |                              [link](https://github.com/NJUNLP/ReNeLLM)                               |
| 2024.06 | **QROA: A Black-Box Query-Response Optimization Attack on LLMs (QROA)** |   arXiv    | [link](https://arxiv.org/abs/2406.02044) |                              [link](https://github.com/qroa/qroa)                               |
| 2024.06 | **Poisoned LangChain: Jailbreak LLMs by LangChain (PLC)** |   arXiv    | [link](https://arxiv.org/pdf/2406.18122) |                              [link](https://github.com/CAM-FSS/jailbreak-langchain)                               |
| 2024.05 | **Multilingual Jailbreak Challenges in Large Language Models** |   ICLR'24    | [link](https://arxiv.org/pdf/2310.06474)  |                              [link](https://github.com/DAMO-NLP-SG/multilingual-safety-for-LLMs)                               |
| 2024.05 | **DeepInception: Hypnotize Large Language Model to Be Jailbreaker (DeepInception)** |   EMNLP'24    | [link](https://arxiv.org/pdf/2311.03191)  |                              [link](https://github.com/tmlr-group/DeepInception)                               |
| 2024.05 | **GPT-4 Jailbreaks Itself with Near-Perfect Success Using Self-Explanation (IRIS)** |   ACL'24    | [link](https://arxiv.org/abs/2405.13077) |                              -                               |
| 2024.05 | **GUARD: Role-playing to Generate Natural-language Jailbreakings to Test Guideline Adherence of LLMs (GUARD)** |   arXiv    | [link](https://arxiv.org/pdf/2402.03299) |                              -                               |
| 2024.05 | **"Do Anything Now": Characterizing and Evaluating In-The-Wild Jailbreak Prompts on Large Language Models (DAN)** |   CCS'24    | [link](https://arxiv.org/pdf/2308.03825) |                              [link](https://github.com/verazuo/jailbreak_llms)                               |
| 2024.05 | **Gpt-4 is too smart to be safe: Stealthy chat with llms via cipher (SelfCipher)** |   ICLR'24    | [link](https://arxiv.org/pdf/2308.06463) |                              [link](https://github.com/RobustNLP/CipherChat)                               |
| 2024.05 | **Jailbreaking Large Language Models Against Moderation Guardrails via Cipher Characters (JAM)** | NeurIPS'24 | [link](https://arxiv.org/pdf/2405.20413) |                              -                               |
| 2024.05 | **Jailbreak and Guard Aligned Language Models with Only Few In-Context Demonstrations (ICA)** |   arXiv    | [link](https://arxiv.org/pdf/2310.06387) |                              -                               |
| 2024.04 | **Many-shot jailbreaking (MSJ)** |   NeurIPS'24 Anthropic   | [link](https://www-cdn.anthropic.com/af5633c94ed2beb282f6a53c595eb437e8e7b630/Many_Shot_Jailbreaking__2024_04_02_0936.pdf) |                              -                              |
| 2024.04 | **PANDORA: Detailed LLM jailbreaking via collaborated phishing agents with decomposed reasoning (PANDORA)** |   ICLR Workshop'24    | [link](https://openreview.net/pdf?id=9o06ugFxIj) |                              -                              |
| 2024.04 | **Fuzzllm: A novel and universal fuzzing framework for proactively discovering jailbreak vulnerabilities in large language models (FuzzLLM)** |   ICASSP'24    | [link](https://arxiv.org/pdf/2309.05274) |                              [link](https://github.com/RainJamesY/FuzzLLM)                              |
| 2024.04 | **Sandwich attack: Multi-language mixture adaptive attack on llms (Sandwich attack)** |   TrustNLP'24    | [link](https://arxiv.org/pdf/2404.07242) |                              -                              |
| 2024.03 | **Tastle: Distract large language models for automatic jailbreak attack (TASTLE)** |   arXiv    | [link](https://arxiv.org/pdf/2403.08424) |                              -                               |
| 2024.03 | **DrAttack: Prompt Decomposition and Reconstruction Makes Powerful LLM Jailbreakers (DrAttack)** |   EMNLP'24    | [link](https://arxiv.org/pdf/2402.16914) |                              [link](https://github.com/xirui-li/DrAttack)                               |
| 2024.02 | **PRP: Propagating Universal Perturbations to Attack Large Language Model Guard-Rails (PRP)** |   arXiv    | [link](https://arxiv.org/pdf/2402.15911) |                              -                              |
| 2024.02 | **CodeChameleon: Personalized Encryption Framework for Jailbreaking Large Language Models (CodeChameleon)** |   arXiv    | [link](https://arxiv.org/pdf/2402.16717) |                              [link](https://github.com/huizhang-L/CodeChameleon)                              |
| 2024.02 | **PAL: Proxy-Guided Black-Box Attack on Large Language Models (PAL)** |   arXiv    | [link](https://arxiv.org/abs/2402.09674) |                              [link](https://github.com/chawins/pal)                               |
| 2024.02 | **Jailbreaking Proprietary Large Language Models using Word Substitution Cipher** |   arXiv    | [link](https://arxiv.org/pdf/2402.10601) |                              -                              |
| 2024.02 | **Query-Based Adversarial Prompt Generation** |   arXiv    | [link](https://arxiv.org/pdf/2402.12329) |                              -                              |
| 2024.02 | **Leveraging the Context through Multi-Round Interactions for Jailbreaking Attacks (Contextual Interaction Attack)** |   arXiv    | [link](https://arxiv.org/pdf/2402.09177) |                              -                               |
| 2024.02 | **Semantic Mirror Jailbreak: Genetic Algorithm Based Jailbreak Prompts Against Open-source LLMs (SMJ)** |   arXiv    | [link](https://arxiv.org/pdf/2402.14872) |                              -                               |
| 2024.02 | **Cognitive Overload: Jailbreaking Large Language Models with Overloaded Logical Thinking** |   NAACL'24    | [link](https://arxiv.org/pdf/2311.09827#page=10.84) |                              [link](https://github.com/luka-group/CognitiveOverload)                              |
| 2024.01 | **Low-Resource Languages Jailbreak GPT-4** |   NeurIPS Workshop'24    | [link](https://arxiv.org/pdf/2310.02446) |                             -                               |
| 2024.01 | **How Johnny Can Persuade LLMs to Jailbreak Them: Rethinking Persuasion to Challenge AI Safety by Humanizing LLMs (PAP)** |   arXiv    | [link](https://arxiv.org/pdf/2401.06373) |                              [link](https://github.com/CHATS-lab/persuasive_jailbreaker)                               |
| 2023.12 | **Tree of Attacks: Jailbreaking Black-Box LLMs Automatically (TAP)** |   NeurIPS'24   | [link](https://arxiv.org/abs/2312.02119) |                              [link](https://github.com/RICommunity/TAP)                               |
| 2023.12 | **Make Them Spill the Beans! Coercive Knowledge Extraction from (Production) LLMs** |   arXiv    | [link](https://arxiv.org/pdf/2312.04782) |                              -                               |
| 2023.12 | **Ignore This Title and HackAPrompt: Exposing Systemic Vulnerabilities of LLMs through a Global Scale Prompt Hacking Competition** |   ACL'24    | [link](https://aclanthology.org/2023.emnlp-main.302/) |                              -                               |
| 2023.11 | **Scalable and Transferable Black-Box Jailbreaks for Language Models via Persona Modulation (Persona)** |   NeurIPS Workshop'23    | [link](https://arxiv.org/pdf/2311.03348) |                              -                               |
| 2023.10 | **Jailbreaking Black Box Large Language Models in Twenty Queries (PAIR)** |   NeurIPS'24    | [link](https://arxiv.org/pdf/2310.08419) |                              [link](https://github.com/patrickrchao/JailbreakingLLMs)                               |
| 2023.10 | **Adversarial Demonstration Attacks on Large Language Models (advICL)** |   EMNLP'24    | [link](https://arxiv.org/pdf/2305.14950) |                              -                               |
| 2023.10 | **MASTERKEY: Automated Jailbreaking of Large Language Model Chatbots (MASTERKEY)** |   NDSS'24    | [link](https://arxiv.org/pdf/2307.08715) |    [link](https://github.com/LLMSecurity/MasterKey)            |              -                               |
| 2023.10 | **Attack Prompt Generation for Red Teaming and Defending Large Language Models (SAP)** |   EMNLP'23    | [link](https://arxiv.org/pdf/2310.12505) |                              [link](https://github.com/Aatrox103/SAP)                              |
| 2023.10 | **An LLM can Fool Itself: A Prompt-Based Adversarial Attack (PromptAttack)** |   ICLR'24    | [link](https://arxiv.org/pdf/2310.13345) |                              [link](https://github.com/GodXuxilie/PromptAttack)                              |
| 2023.09 | **Multi-step Jailbreaking Privacy Attacks on ChatGPT (MJP)** |   EMNLP Findings'23    | [link](https://arxiv.org/pdf/2304.05197) |                              [link](https://github.com/HKUST-KnowComp/LLM-Multistep-Jailbreak)                                |
| 2023.09 | **Open Sesame! Universal Black Box Jailbreaking of Large Language Models (GA)** |   Applied Sciences'24    | [link](https://arxiv.org/abs/2309.01446) |                              -                               |
| 2023.05 | **Not what you’ve signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection** |   CCS'23    | [link](https://arxiv.org/pdf/2302.12173?trk=public_post_comment-text) |                               [link](https://github.com/greshake/llm-security)                               |
| 2022.11 | **Ignore Previous Prompt: Attack Techniques For Language Models (PromptInject)** |   NeurIPS WorkShop'22    | [link](https://arxiv.org/pdf/2211.09527) |                              [link](https://github.com/agencyenterprise/PromptInject)                              |













#### White-box Attack

| Year    | Title                                                        |      Venue       |                            Paper                             |                            Code                            |
| ------- | ------------------------------------------------------------ | :--------------: | :----------------------------------------------------------: | :--------------------------------------------------------: |
| 2025.03 | **Guiding not Forcing: Enhancing the Transferability of Jailbreaking Attacks on LLMs via Removing Superfluous Constraints** | arXiv | [link](https://arxiv.org/abs/2503.01865) |    [link](https://github.com/thu-coai/TransferAttack) |
| 2025.02 | **Improved techniques for optimization-based jailbreaking on large language models (I-GCG)** |   ICLR'25     | [link](https://arxiv.org/pdf/2405.21018) |        [link](https://github.com/jiaxiaojunQAQ/I-GCG)      |
| 2024.12 | **Efficient Adversarial Training in LLMs with Continuous Attacks** | NeurIPS'24 | [link](https://arxiv.org/abs/2405.15589) | [link](https://github.com/sophie-xhonneux/Continuous-AdvTrain) |
| 2024.11 | **AmpleGCG-Plus: A Strong Generative Model of Adversarial Suffixes to Jailbreak LLMs with Higher Success Rates in Fewer Attempts** |      arXiv       |           [link](https://arxiv.org/pdf/2410.22143v1)           |                          -                             |
| 2024.11 | **DROJ: A Prompt-Driven Attack against Large Language Models** |      arXiv       |           [link](https://arxiv.org/pdf/2411.09125)           |                           [link](https://github.com/Leon-Leyang/LLM-Safeguard)                              |
| 2024.11 | **SQL Injection Jailbreak: a structural disaster of large language models** |      arXiv       |           [link](https://arxiv.org/pdf/2411.01565)           |                           [link](https://github.com/weiyezhimeng/SQL-Injection-Jailbreak)                              |
| 2024.10 | **Functional Homotopy: Smoothing Discrete Optimization via Continuous Parameters for LLM Jailbreak Attacks** |      arXiv       |           [link](https://arxiv.org/pdf/2410.04234)           |                           -                              |
| 2024.10 | **AttnGCG: Enhancing Jailbreaking Attacks on LLMs with Attention Manipulation** |      arXiv       |           [link](https://arxiv.org/pdf/2410.09040v1)           |                           [link](https://github.com/UCSC-VLAA/AttnGCG-attack)                               |
| 2024.10 | **Jailbreak Instruction-Tuned LLMs via end-of-sentence MLP Re-weighting** |      arXiv       |           [link](https://arxiv.org/pdf/2410.10150v1)           |                            -                              |
| 2024.10 | **Boosting Jailbreak Transferability for Large Language Models (SI-GCG)** |      arXiv       |           [link](https://arxiv.org/pdf/2410.15645v1)           |                            -                              |
| 2024.10 | **Iterative Self-Tuning LLMs for Enhanced Jailbreaking Capabilities (ADV-LLM)** |      arXiv       |           [link](https://arxiv.org/pdf/2410.18469v1)           |                             [link](https://github.com/SunChungEn/ADV-LLM)                               |
| 2024.08 | **Probing the Safety Response Boundary of Large Language Models via Unsafe Decoding Path Generation (JVD)** |      arXiv       |           [link](https://arxiv.org/pdf/2408.10668)           |                             -                              |
| 2024.08 | **Jailbreak Open-Sourced Large Language Models via Enforced Decoding (EnDec)** |      ACL'24      | [link](https://aclanthology.org/2024.acl-long.299.pdf#page=4.96) |                             -                              |
| 2024.07 | **Refusal in Language Models Is Mediated by a Single Direction** |      arXiv       |           [Link](https://arxiv.org/pdf/2406.11717)           |    [Link](https://github.com/andyrdt/refusal_direction)    |
| 2024.07 | **Revisiting Character-level Adversarial Attacks for Language Models** |     ICML'24      |           [link](https://arxiv.org/abs/2405.04346)           |       [link](https://github.com/LIONS-EPFL/Charmer)        |
| 2024.07 | **Badllama 3: removing safety finetuning from Llama 3 in minutes (Badllama 3)** |      arXiv       |           [link](https://arxiv.org/pdf/2407.01376)           |                             -                              |
| 2024.07 | **SOS! Soft Prompt Attack Against Open-Source Large Language Models** |      arXiv       |           [link](https://arxiv.org/abs/2407.03160)           |                             -                              |
| 2024.06 | **COLD-Attack: Jailbreaking LLMs with Stealthiness and Controllability (COLD-Attack)** |     ICML'24      |           [link](https://arxiv.org/pdf/2402.08679)           |      [link](https://github.com/Yu-Fangxu/COLD-Attack)      |
| 2024.05 | **Semantic-guided Prompt Organization for Universal Goal Hijacking against LLMs** |      arXiv       |           [link](https://arxiv.org/abs/2405.14189)           |                                                            |
| 2024.05 | **Efficient LLM Jailbreak via Adaptive Dense-to-sparse Constrained Optimization** |    NeurIPS'24    |           [Link](https://arxiv.org/pdf/2405.09113)           |                             -                              |
| 2024.05 | **AutoDAN: Generating Stealthy Jailbreak Prompts on Aligned Large Language Models (AutoDAN)** |     ICLR'24      |           [link](https://arxiv.org/pdf/2310.04451)           |      [link](https://github.com/SheltonLiu-N/AutoDAN)       |
| 2024.05 | **AmpleGCG: Learning a Universal and Transferable Generative Model of Adversarial Suffixes for Jailbreaking Both Open and Closed LLMs (AmpleGCG)** |      arXiv       |           [link](https://arxiv.org/pdf/2404.07921)           |     [link](https://github.com/OSU-NLP-Group/AmpleGCG)      |
| 2024.05 | **Boosting jailbreak attack with momentum (MAC)**            | ICLR Workshop'24 |           [link](https://arxiv.org/pdf/2405.01229)           |  [link](https://github.com/weizeming/momentum-attack-llm)  |
| 2024.04 | **AdvPrompter: Fast Adaptive Adversarial Prompting for LLMs (AdvPrompter)** |      arXiv       |           [link](https://arxiv.org/pdf/2404.16873)           |  [link](https://github.com/facebookresearch/advprompter)   |
| 2024.03 | **Universal Jailbreak Backdoors from Poisoned Human Feedback** |     ICLR'24      |       [link](https://openreview.net/pdf?id=GxCGsxiAaK)       |                             -                              |
| 2024.02 | **Attacking large language models with projected gradient descent (PGD)** |      arXiv       |           [link](https://arxiv.org/pdf/2402.09154)           |                             -                              |
| 2024.02 | **Open the Pandora's Box of LLMs: Jailbreaking LLMs through Representation Engineering (JRE)** |      arXiv       |           [link](https://arxiv.org/pdf/2401.06824)           |                             -                              |
| 2024.02 | **Curiosity-driven red-teaming for large language models (CRT)** |      arXiv       |           [link](https://arxiv.org/pdf/2402.19464)           | [link](https://github.com/Improbable-AI/curiosity_redteam) |
| 2023.12 | **AutoDAN: Interpretable Gradient-Based Adversarial Attacks on Large Language Models (AutoDAN)** |      arXiv       |           [link](https://arxiv.org/abs/2310.15140)           |    [link](https://github.com/rotaryhammer/code-autodan)    |
| 2023.10 | **Catastrophic jailbreak of open-source llms via exploiting generation** |     ICLR'24      |           [link](https://arxiv.org/pdf/2310.06987)           |  [link](https://github.com/Princeton-SysML/Jailbreak_LLM)  |
| 2023.06 | **Automatically Auditing Large Language Models via Discrete Optimization (ARCA)** |     ICML'23      | [link](https://proceedings.mlr.press/v202/jones23a/jones23a.pdf) |     [link](https://github.com/ejones313/auditing-llms)     |
| 2023.07 | **Universal and Transferable Adversarial Attacks on Aligned Language Models (GCG)** |      arXiv       |           [link](https://arxiv.org/pdf/2307.15043)           |     [link](https://github.com/llm-attacks/llm-attacks)     |





#### Multi-turn Attack



| Time    | Title                                                        |   Venue   |                  Paper                   |                             Code                             |
| ------- | ------------------------------------------------------------ | :-------: | :--------------------------------------: | :----------------------------------------------------------: |
| 2025.04 | **Multi-Turn Jailbreaking Large Language Models via Attention Shifting** |   AAAI'25    | [link](https://ojs.aaai.org/index.php/AAAI/article/view/34553) |    -      |
| 2025.04 | **X-Teaming: Multi-Turn Jailbreaks and Defenses with Adaptive Multi-Agents** |   arXiv    | [link](https://arxiv.org/pdf/2504.13203) |    [link](https://github.com/salman-lui/x-teaming)      |
| 2025.04 | **Strategize Globally, Adapt Locally: A Multi-Turn Red Teaming Agent with Dual-Level Learning** |   arXiv    | [link](https://arxiv.org/pdf/2504.01278) |      -    |
| 2025.03 | **Foot-In-The-Door: A Multi-turn Jailbreak for LLMs** |   arXiv    | [link](https://arxiv.org/pdf/2502.19820) |    [link](https://github.com/Jinxiaolong1129/Foot-in-the-door-Jailbreak)      |
| 2025.03 | **Siege: Autonomous Multi-Turn Jailbreaking of Large Language Models with Tree Search** |   arXiv    | [link](https://arxiv.org/pdf/2503.10619) |      -    |
| 2024.11 | **MRJ-Agent: An Effective Jailbreak Agent for Multi-Round Dialogue** |   arXiv    | [link](https://arxiv.org/pdf/2411.03814) |      -    |
| 2024.10 | **Jigsaw Puzzles: Splitting Harmful Questions to Jailbreak Large Language Models (JSP)** |   arXiv   | [link](https://arxiv.org/pdf/2410.11459v1) |    [link](https://github.com/YangHao97/JigSawPuzzles)      |
| 2024.10 | **Multi-round jailbreak attack on large language** |   arXiv   | [link](https://arxiv.org/pdf/2410.11533v1) |     -      |
| 2024.10 | **Derail Yourself: Multi-turn LLM Jailbreak Attack through Self-discovered Clues** |   arXiv   | [link](https://arxiv.org/abs/2410.10700) |     [link](https://github.com/renqibing/ActorAttack)      |
| 2024.10 | **Automated Red Teaming with GOAT: the Generative Offensive Agent Tester** |   arXiv   | [link](https://arxiv.org/pdf/2410.01606) |     -      |
| 2024.09 | **LLM Defenses Are Not Robust to Multi-Turn Human Jailbreaks Yet** |   arXiv   | [link](https://arxiv.org/pdf/2408.15221) |     [link](https://huggingface.co/datasets/ScaleAI/mhj)      |
| 2024.09 | **RED QUEEN: Safeguarding Large Language Models against Concealed Multi-Turn Jailbreaking** |   arXiv   | [link](https://arxiv.org/pdf/2409.17458) |       [link](https://github.com/kriti-hippo/red_queen)       |
| 2024.08 | **FRACTURED-SORRY-Bench: Framework for Revealing Attacks in Conversational Turns Undermining Refusal Efficacy and Defenses over SORRY-Bench (Automated Multi-shot Jailbreaks)** |   arXiv   | [link](https://arxiv.org/abs/2408.16163) |     -      |
| 2024.08 | **Emerging Vulnerabilities in Frontier Models: Multi-Turn Jailbreak Attacks** |   arXiv   | [link](https://arxiv.org/pdf/2409.00137) | [link](https://huggingface.co/datasets/tom-gibbs/multi-turn_jailbreak_attack_datasets) |
| 2024.05 | **CoA: Context-Aware based Chain of Attack for Multi-Turn Dialogue LLM (CoA)** |   arXiv   | [link](https://arxiv.org/pdf/2405.05610) |           [link](https://github.com/YancyKahn/CoA)           |
| 2024.04 | **Great, Now Write an Article About That: The Crescendo Multi-Turn LLM Jailbreak Attack (Crescendo)** | Microsoft Azure | [link](https://arxiv.org/pdf/2404.01833) |                              -                               |







#### Attack on RAG-based LLM



| Time    | Title                                                        | Venue |                  Paper                   |                             Code                             |
| ------- | ------------------------------------------------------------ | :---: | :--------------------------------------: | :----------------------------------------------------------: |
| 2024.09 | **Unleashing Worms and Extracting Data: Escalating the Outcome of Attacks against RAG-based Inference in Scale and Severity Using Jailbreaking** | arXiv | [link](https://arxiv.org/pdf/2409.08045) | [link](https://github.com/StavC/UnleashingWorms-ExtractingData) |
| 2024.02 | **Pandora: Jailbreak GPTs by Retrieval Augmented Generation Poisoning (Pandora)** | arXiv | [link](https://arxiv.org/pdf/2402.08416) |                              -                               |







#### Multi-modal Attack

| Time | Title                                                        |  Venue  |                            Paper                             |                             Code                             |
| ---- | ------------------------------------------------------------ | :-----: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| 2024.11 | **Jailbreak Attacks and Defenses against Multimodal Generative Models: A Survey** | arXiv | [link](https://arxiv.org/pdf/2411.09259) | [link](https://github.com/liuxuannan/Awesome-Multimodal-Jailbreak) |
| 2024.10 | **Chain-of-Jailbreak Attack for Image Generation Models via Editing Step by Step** | arXiv | [link](https://arxiv.org/pdf/2410.03869) | - |
| 2024.10 | **ColJailBreak: Collaborative Generation and Editing for Jailbreaking Text-to-Image Deep Generation** | NeurIPS'24 | [Link](https://nips.cc/virtual/2024/poster/94287) | - |
| 2024.08 | **Jailbreaking Text-to-Image Models with LLM-Based Agents (Atlas)** |   arXiv    | [link](https://arxiv.org/pdf/2408.00523) |                              -                              |
| 2024.07 | **Image-to-Text Logic Jailbreak: Your Imagination can Help You Do Anything** |   arXiv    | [link](https://arxiv.org/pdf/2407.02534) |                              -                              |
| 2024.06 | **Jailbreak Vision Language Models via Bi-Modal Adversarial Prompt** |   arXiv    | [link](https://arxiv.org/pdf/2406.04031) |                              [link](https://github.com/NY1024/BAP-Jailbreak-Vision-Language-Models-via-Bi-Modal-Adversarial-Prompt)                               |
| 2024.05 | **Voice Jailbreak Attacks Against GPT-4o** |   arXiv    | [link](https://arxiv.org/pdf/2405.19103) |                              [link](https://github.com/TrustAIRLab/VoiceJailbreakAttack)                               |
| 2024.05 | **Automatic Jailbreaking of the Text-to-Image Generative AI Systems** |     ICML'24 Workshop    | [link](https://arxiv.org/abs/2405.16567) | [link](https://github.com/Kim-Minseon/APGP) |
| 2024.04 | **Image hijacks: Adversarial images can control generative models at runtime** |   arXiv    | [link](https://arxiv.org/pdf/2309.00236) |                              [link](https://github.com/euanong/image-hijacks)                               |
| 2024.03 | **An image is worth 1000 lies: Adversarial transferability across prompts on vision-language models (CroPA)** |   ICLR'24    | [link](https://arxiv.org/pdf/2403.09766) |                              [link](https://github.com/Haochen-Luo/CroPA)                               |
| 2024.03 | **Jailbreak in pieces: Compositional adversarial attacks on multi-modal language model** |   ICLR'24    | [link](https://openreview.net/pdf?id=plmBsXHxgR) |                              -                               |
| 2024.03 | **Rethinking model ensemble in transfer-based adversarial attacks** |   ICLR'24    | [link](https://arxiv.org/pdf/2303.09105) |                              [link](https://github.com/huanranchen/AdversarialAttacks)                               |
| 2024.02 | **VLATTACK: Multimodal Adversarial Attacks on Vision-Language Tasks via Pre-trained Models** |   NeurIPS'23    | [link](https://arxiv.org/abs/2310.04655) |               [link](https://github.com/ericyinyzy/VLAttack)                                         |
| 2024.02 | **Jailbreaking Attack against Multimodal Large Language Model** |   arXiv    | [link](https://arxiv.org/pdf/2402.02309) |                             -                          |
| 2024.01 | **Jailbreaking GPT-4V via Self-Adversarial Attacks with System Prompts** |   arXiv    | [link](https://arxiv.org/pdf/2311.09127) |                             -                          |
| 2024.03 | **Visual Adversarial Examples Jailbreak Aligned Large Language Models** |   AAAI'24    | [link](https://ojs.aaai.org/index.php/AAAI/article/view/30150/32038) |                              -                               |
| 2023.12 | **OT-Attack: Enhancing Adversarial Transferability of Vision-Language Models via Optimal Transport Optimization (OT-Attack)** |   arXiv    | [link](https://arxiv.org/pdf/2312.04403) |                              -                               |
| 2023.12 | **FigStep: Jailbreaking Large Vision-language Models via Typographic Visual Prompts (FigStep)** |   arXiv    | [link](https://arxiv.org/pdf/2311.05608) |                              [link](https://github.com/ThuCCSLab/FigStep)                               |
| 2023.11 | **SneakyPrompt: Jailbreaking Text-to-image Generative Models** |   S&P'24    | [link](https://arxiv.org/pdf/2305.12082) |                              [link](https://github.com/Yuchen413/text2image_safety)                               |
| 2023.11 | **On Evaluating Adversarial Robustness of Large Vision-Language Models** |   NeurIPS'23    | [link](https://proceedings.neurips.cc/paper_files/paper/2023/file/a97b58c4f7551053b0512f92244b0810-Paper-Conference.pdf) |                              [link](https://github.com/yunqing-me/AttackVLM)                               |
| 2023.10 | **How Robust is Google's Bard to Adversarial Image Attacks?** |   arXiv    | [link](https://arxiv.org/pdf/2309.11751) |                              [link](https://github.com/thu-ml/Attack-Bard)                               |
| 2023.08 | **AdvCLIP: Downstream-agnostic Adversarial Examples in Multimodal Contrastive Learning (AdvCLIP)** |   ACM MM'23    | [link](https://arxiv.org/pdf/2308.07026) |                              [link](https://github.com/CGCL-codes/AdvCLIP)                               |
| 2023.07 | **Set-level Guidance Attack: Boosting Adversarial Transferability of Vision-Language Pre-training Models (SGA)** |   ICCV'23    | [link](https://openaccess.thecvf.com/content/ICCV2023/papers/Lu_Set-level_Guidance_Attack_Boosting_Adversarial_Transferability_of_Vision-Language_Pre-training_Models_ICCV_2023_paper.pdf) |                              [link](https://github.com/Zoky-2020/SGA)                               |
| 2023.07 | **On the Adversarial Robustness of Multi-Modal Foundation Models** |   ICCV Workshop'23    | [link](https://openaccess.thecvf.com/content/ICCV2023W/AROW/papers/Schlarmann_On_the_Adversarial_Robustness_of_Multi-Modal_Foundation_Models_ICCVW_2023_paper.pdf) |                              -                               |
| 2022.10 | **Towards Adversarial Attack on Vision-Language Pre-training Models** |   arXiv    | [link](https://arxiv.org/pdf/2206.09391) |                              [link](https://github.com/adversarial-for-goodness/Co-Attack)                               |













### Jailbreak Defense

#### Learning-based Defense
| Time | Title                                                        |  Venue  |                            Paper                             |                             Code                             |
| ---- | ------------------------------------------------------------ | :-----: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| 2025.04 | **JailDAM: Jailbreak Detection with Adaptive Memory for Vision-Language Model** | COLM'25 | [link](https://arxiv.org/pdf/2504.03770) | [link](https://github.com/ShenzheZhu/JailDAM) |
| 2024.12 | **Shaping the Safety Boundaries: Understanding and Defending Against Jailbreaks in Large Language Models** | arXiv'24 | [link](https://arxiv.org/pdf/2412.17034) | - |
| 2024.10 | **Safety-Aware Fine-Tuning of Large Language Models** | arXiv'24 | [link](https://arxiv.org/pdf/2410.10014) | - |
| 2024.10 | **MoJE: Mixture of Jailbreak Experts, Naive Tabular Classifiers as Guard for Prompt Attacks** | AAAI'24 | [link](https://arxiv.org/pdf/2409.17699) | - |
| 2024.08 | **BaThe: Defense against the Jailbreak Attack in Multimodal Large Language Models by Treating Harmful Instruction as Backdoor Trigger (BaThe)** |   arXiv    | [link](https://arxiv.org/pdf/2408.09093)  |                               -                              |
| 2024.07 | **DART: Deep Adversarial Automated Red Teaming for LLM Safety** |   arXiv    | [link](https://arxiv.org/abs/2407.03876)  |                               -                              |
| 2024.07 | **Eraser: Jailbreaking Defense in Large Language Models via Unlearning Harmful Knowledge (Eraser)** |   arXiv    | [link](https://arxiv.org/pdf/2404.05880) |                              [link](https://github.com/ZeroNLP/Eraser)                               |
| 2024.07 | **Safe Unlearning: A Surprisingly Effective and Generalizable Solution to Defend Against Jailbreak Attacks** |   arXiv    | [link](https://arxiv.org/abs/2407.02855) |                              [link](https://github.com/thu-coai/SafeUnlearning)                               |
| 2024.06 | **Adversarial Tuning: Defending Against Jailbreak Attacks for LLMs** | arXiv | [Link](https://arxiv.org/pdf/2406.06622) | - |
| 2024.06 | **Jatmo: Prompt Injection Defense by Task-Specific Finetuning (Jatmo)** |   arXiv    | [link](https://arxiv.org/pdf/2312.17673) |                              [link](https://github.com/wagner-group/prompt-injection-defense)                               |
| 2024.06 | **Defending Large Language Models Against Jailbreaking Attacks Through Goal Prioritization (SafeDecoding)** |   ACL'24    | [link](https://arxiv.org/pdf/2311.09096) |                              [link](https://github.com/thu-coai/JailbreakDefense_GoalPriority)                               |
| 2024.06 | **Mitigating Fine-tuning based Jailbreak Attack with Backdoor Enhanced Safety Alignment** |   NeurIPS'24   | [link](https://jayfeather1024.github.io/Finetuning-Jailbreak-Defense/) |                              [link](https://github.com/Jayfeather1024/Backdoor-Enhanced-Alignment)                               |
| 2024.06 | **On Prompt-Driven Safeguarding for Large Language Models (DRO)** |   ICML'24    | [link](https://arxiv.org/pdf/2401.18018) |                              [link](https://github.com/chujiezheng/LLM-Safeguard)          |
| 2024.06 | **Robust Prompt Optimization for Defending Language Models Against Jailbreaking Attacks (RPO)** |   NeurIPS'24   | [link](https://arxiv.org/pdf/2401.17263) |                              -          |
| 2024.06 | **Fight Back Against Jailbreaking via Prompt Adversarial Tuning (PAT)** |   NeurIPS'24   | [link](https://arxiv.org/pdf/2402.06255) |                              [link](https://github.com/rain152/PAT)          |
| 2024.05 | **Towards Comprehensive and Efficient Post Safety Alignment of Large Language Models via Safety Patching (SAFEPATCHING)** |   arXiv    | [link](https://arxiv.org/pdf/2405.13820) |                              -          |
| 2024.05 | **Detoxifying Large Language Models via Knowledge Editing (DINM)** |   ACL'24    | [link](https://arxiv.org/pdf/2403.14472) |                              [link](https://github.com/zjunlp/EasyEdit/blob/main/examples/SafeEdit.md)          |
| 2024.05 | **Defending Large Language Models Against Jailbreak Attacks via Layer-specific Editing** |   arXiv | [link](https://arxiv.org/abs/2405.18166) | [link](https://github.com/ledllm/ledllm) |
| 2023.11 | **MART: Improving LLM Safety with Multi-round Automatic Red-Teaming (MART)** |   ACL'24    | [link](https://arxiv.org/pdf/2311.07689) |                              -          |
| 2023.11 | **Baseline defenses for adversarial attacks against aligned language models** |   arXiv    | [link](https://arxiv.org/pdf/2308.14132) |                              -                               |
| 2023.10 | **Safe rlhf: Safe reinforcement learning from human feedback** |   arXiv    | [link](https://arxiv.org/pdf/2310.12773) |                              [link](https://github.com/PKU-Alignment/safe-rlhf)                                |
| 2023.08 | **Red-Teaming Large Language Models using Chain of Utterances for Safety-Alignment (RED-INSTRUCT)** |   arXiv    | [link](https://arxiv.org/pdf/2308.09662) |                              [link](https://github.com/declare-lab/red-instruct)                                |
| 2022.04 | **Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback** |   Anthropic    | [link](https://arxiv.org/pdf/2204.05862?spm=a2c6h.13046898.publish-article.36.6cd56ffaIPu4NQ&file=2204.05862) |                              -                                |






















#### Strategy-based Defense



| Time | Title                                                        |  Venue  |                            Paper                             |                             Code                             |
| ---- | ------------------------------------------------------------ | :-----: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| 2025.05 | **Reasoning-to-Defend: Safety-Aware Reasoning Can Defend Large Language Models from Jailbreaking** | arXiv | [link](https://arxiv.org/pdf/2502.12970?) | [link](https://github.com/chuhac/Reasoning-to-Defend) |
| 2024.11 | **Rapid Response: Mitigating LLM Jailbreaks with a Few Examples** | arXiv | [link](https://arxiv.org/pdf/2411.07494v1) | [link](https://github.com/rapidresponsebench/rapidresponsebench) |
| 2024.10 | **RePD: Defending Jailbreak Attack through a Retrieval-based Prompt Decomposition Process (RePD)** | arXiv | [link](https://arxiv.org/pdf/2410.08660v1) |  - |
| 2024.10 | **Guide for Defense (G4D): Dynamic Guidance for Robust and Balanced Defense in Large Language Models (G4D)** | arXiv | [link](https://arxiv.org/pdf/2410.17922v1) |  [link](https://github.com/IDEA-XL/G4D) |
| 2024.10 | **Jailbreak Antidote: Runtime Safety-Utility Balance via Sparse Representation Adjustment in Large Language Models** | arXiv | [link](https://arxiv.org/html/2410.02298v1) | - |
| 2024.09 | **HSF: Defending against Jailbreak Attacks with Hidden State Filtering** | arXiv | [link](https://arxiv.org/html/2409.03788v1) | [link](https://anonymous.4open.science/r/Hidden-State-Filtering-8652/) |
| 2024.08 | **EEG-Defender: Defending against Jailbreak through Early Exit Generation of Large Language Models (EEG-Defender)** |   arXiv    | [link](https://arxiv.org/pdf/2408.11308) |                              -                               |
| 2024.08 | **Prefix Guidance: A Steering Wheel for Large Language Models to Defend Against Jailbreak Attacks (PG)** |   arXiv    | [link](https://arxiv.org/pdf/2408.08924) |                              [link](https://github.com/weiyezhimeng/Prefix-Guidance)                               |
| 2024.08 | **Self-Evaluation as a Defense Against Adversarial Attacks on LLMs (Self-Evaluation)** |   arXiv    | [link](https://arxiv.org/pdf/2407.03234#page=2.47) |                              [link](https://github.com/Linlt-leon/self-eval)                               |
| 2024.06 | **Defending LLMs against Jailbreaking Attacks via Backtranslation (Backtranslation)** |   ACL Findings'24    | [link](https://arxiv.org/pdf/2402.16459) |                              [link](https://github.com/YihanWang617/LLM-Jailbreaking-Defense-Backtranslation)                               |
| 2024.06 | **SafeDecoding: Defending against Jailbreak Attacks via Safety-Aware Decoding (SafeDecoding)** |   ACL'24    | [link](https://arxiv.org/pdf/2402.08983) |                              [link](https://github.com/uw-nsl/SafeDecoding)                               |
| 2024.06 | **Defending Against Alignment-Breaking Attacks via Robustly Aligned LLM** |   ACL'24    | [link](https://arxiv.org/pdf/2309.14348) |                              -                               |
| 2024.06 | **A Wolf in Sheep’s Clothing: Generalized Nested Jailbreak Prompts can Fool Large Language Models Easily (ReNeLLM)** |   NAACL'24    | [link](https://arxiv.org/abs/2311.08268) |                              [link](https://github.com/NJUNLP/ReNeLLM)                               |
| 2024.06 | **SMOOTHLLM: Defending Large Language Models Against Jailbreaking Attacks** |   arXiv    | [link](https://arxiv.org/pdf/2310.03684) |                              [link](https://github.com/arobey1/smooth-llm)                               |
| 2024.05 | **Enhancing Large Language Models Against Inductive Instructions with Dual-critique Prompting (Dual-critique)** |   ACL'24    | [link](https://arxiv.org/pdf/2305.13733) |                             [link](https://github.com/DevoAllen/INDust)                               |
| 2024.05 | **PARDEN, Can You Repeat That? Defending against Jailbreaks via Repetition (PARDEN)** |   ICML'24    | [link](https://arxiv.org/pdf/2405.07932) |                             [link](https://github.com/Ed-Zh/PARDEN)                               |
| 2024.05 | **LLM Self Defense: By Self Examination, LLMs Know They Are Being Tricked** |   ICLR Tiny Paper'24    | [link](https://arxiv.org/pdf/2308.07308) |                             [link](https://github.com/poloclub/llm-self-defense)                               |
| 2024.05 | **GradSafe: Detecting Unsafe Prompts for LLMs via Safety-Critical Gradient Analysis (GradSafe)** |   ACL'24    | [link](\https://arxiv.org/pdf/2402.13494) |                             [link](https://github.com/xyq7/GradSafe)                               |
| 2024.05 | **Multilingual Jailbreak Challenges in Large Language Models** |   ICLR'24    | [link](https://arxiv.org/pdf/2310.06474)  |                              [link](https://github.com/DAMO-NLP-SG/multilingual-safety-for-LLMs)                               |
| 2024.05 | **Gradient Cuff: Detecting Jailbreak Attacks on Large Language Models by Exploring Refusal Loss Landscapes** |   NeurIPS'24   | [link](https://arxiv.org/pdf/2403.00867)  |                              -                            |
| 2024.05 | **AutoDefense: Multi-Agent LLM Defense against Jailbreak Attacks** |   arXiv    | [link](https://arxiv.org/pdf/2403.04783) |                             [link](https://github.com/XHMY/AutoDefense)                               |
| 2024.05 | **Bergeron: Combating adversarial attacks through a conscience-based alignment framework (Bergeron)** |   arXiv    | [link](https://arxiv.org/pdf/2312.00029) |                             [link](https://github.com/matthew-pisano/Bergeron)                               |
| 2024.05 | **Jailbreak and Guard Aligned Language Models with Only Few In-Context Demonstrations (ICD)** |   arXiv    | [link](https://arxiv.org/pdf/2310.06387)  |                              -                               |
| 2024.04 | **Protecting your llms with information bottleneck** |   NeurIPS'24    | [link](https://arxiv.org/pdf/2404.13968)  |                              [link](https://github.com/zichuan-liu/IB4LLMs)                                |
| 2024.04 | **Pruning for Protection: Increasing Jailbreak Resistance in Aligned LLMs Without Fine-Tuning** |   arXiv    | [link](https://arxiv.org/pdf/2401.10862)  |                              [link](https://github.com/CrystalEye42/eval-safety)                                |
| 2024.02 | **Certifying LLM Safety against Adversarial Prompting** |   arXiv    | [link](https://arxiv.org/pdf/2309.02705) |                              [link](https://github.com/aounon/certified-llm-safety)                              |
| 2024.02 | **Break the Breakout: Reinventing LM Defense Against Jailbreak Attacks with Self-Refinement** |   arXiv    | [link](https://arxiv.org/pdf/2402.15180) |                              -                              |
| 2024.02 | **Defending large language models against jailbreak attacks via semantic smoothing (SEMANTICSMOOTH)** |   arXiv    | [link](https://arxiv.org/pdf/2402.16192) |                              [link](https://github.com/UCSB-NLP-Chang/SemanticSmooth)                             |
| 2024.01 | **Intention Analysis Makes LLMs A Good Jailbreak Defender (IA)** |   arXiv    | [link](https://arxiv.org/pdf/2401.06561) |                              [link](https://github.com/alphadl/SafeLLM_with_IntentionAnalysis)                               |
| 2024.01 | **How Johnny Can Persuade LLMs to Jailbreak Them: Rethinking Persuasion to Challenge AI Safety by Humanizing LLMs (PAP)** |   ACL'24    | [link](https://arxiv.org/pdf/2401.06373) |                              [link](https://github.com/CHATS-lab/persuasive_jailbreaker)                               |
| 2023.12 | **Defending ChatGPT against jailbreak attack via self-reminders (Self-Reminder)** |   Nature Machine Intelligence    | [link](https://xyq7.github.io/papers/NMI-JailbreakDefense.pdf) |                              [link](https://github.com/yjw1029/Self-Reminder/)                               |
| 2023.11 | **Detecting language model attacks with perplexity** |   arXiv    | [link](https://arxiv.org/pdf/2308.14132) |                             -                              |
| 2023.10 | **RAIN: Your Language Models Can Align Themselves without Finetuning (RAIN)** |    ICLR'24    | [link](https://arxiv.org/pdf/2309.07124) |                              [link](https://github.com/SafeAILab/RAIN)                               |










#### Guard Model

| Time    | Title                                                        |   Venue    |                            Paper                             |                             Code                             |
| ------- | ------------------------------------------------------------ | :--------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| 2025.05 | **GuardReasoner-VL: Safeguarding VLMs via Reinforced Reasoning (GuardReasoner-VL)** | arXiv'25  |          [link](https://arxiv.org/abs/2505.11049)          |                              [link](https://github.com/yueliu1999/GuardReasoner-VL/)                               |
| 2025.04 | **X-Guard: Multilingual Guard Agent for Content Moderation (X-Guard)** | arXiv'25  |          [link](https://arxiv.org/pdf/2504.08848)          |                              [link](https://github.com/UNHSAILLab/X-Guard)                               |
| 2025.02 | **ThinkGuard: Deliberative Slow Thinking Leads to Cautious Guardrails (ThinkGuard)** | arXiv'25  |          [link](https://arxiv.org/abs/2502.13458)          |                             [link](https://github.com/luka-group/ThinkGuard)                              |
| 2025.02 | **Constitutional Classifiers: Defending against Universal Jailbreaks across Thousands of Hours of Red Teaming** | arXiv'25  |          [link](https://arxiv.org/abs/2501.18837)          |                             -                               |
| 2025.01 | **GuardReasoner: Towards Reasoning-based LLM Safeguards (GuardReasoner)** | ICLR Workshop'25  |          [link](https://arxiv.org/pdf/2501.18492)          |                              [link](https://github.com/yueliu1999/GuardReasoner/)                               |
| 2024.12 | **Lightweight Safety Classification Using Pruned Language Models (Sentence-BERT)** | arXiv'24  |          [link](https://arxiv.org/pdf/2412.13435)          |                              -                               |
| 2024.11 | **GuardFormer: Guardrail Instruction Pretraining for Efficient SafeGuarding (GuardFormer)** | Meta  |          [link](https://openreview.net/pdf?id=vr31i9pzQk)          |                              -                               |
| 2024.11 | **Llama Guard 3 Vision: Safeguarding Human-AI Image Understanding Conversations (LLaMA Guard 3 Vision)** | Meta  |          [link](https://arxiv.org/pdf/2411.10414?)          |                              [link](https://github.com/meta-llama/llama-recipes/tree/main/recipes/responsible_ai/llama_guard)                                |
| 2024.11 | **AEGIS2.0: A Diverse AI Safety Dataset and Risks Taxonomy for Alignment of LLM Guardrails (Aegis2.0)** | Nvidia, NeurIPS'24 Workshop  |          [link](https://openreview.net/pdf?id=0MvGCv35wi)          |                              -                               |
| 2024.11 | **Lightweight Safety Guardrails Using Fine-tuned BERT Embeddings (Sentence-BERT)** | arXiv'24  |          [link](https://arxiv.org/pdf/2411.14398?)          |                              -                               |
| 2024.11 | **STAND-Guard: A Small Task-Adaptive Content Moderation Model (STAND-Guard)** | Microsoft  |          [link](https://arxiv.org/pdf/2411.05214v1)          |                              -                               |
| 2024.10 | **VLMGuard: Defending VLMs against Malicious Prompts via Unlabeled Data** |   arXiv    |         [link](https://arxiv.org/html/2410.00296v1)          |                              -                               |
| 2024.09 | **AEGIS: Online Adaptive AI Content Safety Moderation with Ensemble of LLM Experts (Aegis)** |   Nvidia   |           [link](https://arxiv.org/abs/2404.05993)           | [link](https://huggingface.co/nvidia/Aegis-AI-Content-Safety-LlamaGuard-Defensive-1.0) |
| 2024.09 | **Llama 3.2: Revolutionizing edge AI and vision with open, customizable models (LLaMA Guard 3)** |    Meta    | [link](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/) |  [link](https://huggingface.co/meta-llama/Llama-Guard-3-1B)  |
| 2024.08 | **ShieldGemma: Generative AI Content Moderation Based on Gemma (ShieldGemma)** |   Google   |           [link](https://arxiv.org/pdf/2407.21772)           |     [link](https://huggingface.co/google/shieldgemma-2b)     |
| 2024.07 | **WildGuard: Open One-Stop Moderation Tools for Safety Risks, Jailbreaks, and Refusals of LLMs (WildGuard)** | NeurIPS'24 |           [link](https://arxiv.org/pdf/2406.18495)           |         [link](https://github.com/allenai/wildguard)         |
| 2024.06 | **GuardAgent: Safeguard LLM Agents by a Guard Agent via Knowledge-Enabled Reasoning (GuardAgent)** | arXiv'24 |           [link](https://arxiv.org/pdf/2406.09187)           |         -        |
| 2024.06 | **R2-Guard: Robust Reasoning Enabled LLM Guardrail via Knowledge-Enhanced Logical Reasoning (R2-Guard)** |   arXiv    |           [link](https://arxiv.org/abs/2407.05557)           |       [link](https://github.com/kangmintong/R-2-Guard)       |
| 2024.04 | **Llama Guard 2**                                            |    Meta    | [link](https://www.llama.com/docs/model-cards-and-prompt-formats/meta-llama-guard-2/) | [link](https://github.com/meta-llama/PurpleLlama/blob/main/Llama-Guard2/MODEL_CARD.md) |
| 2024.03 | **AdaShield: Safeguarding Multimodal Large Language Models from Structure-based Attack via Adaptive Shield Prompting (AdaShield)** |  ECCV'24   |           [link](https://arxiv.org/pdf/2403.09513)           |      [link](https://github.com/SaFoLab-WISC/AdaShield)       |
| 2023.12 | **Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations (LLaMA Guard)** |    Meta    |           [link](https://arxiv.org/pdf/2312.06674)           | [link](https://github.com/meta-llama/PurpleLlama/tree/main/Llama-Guard) |





#### Moderation API



| Time    | Title                                                        |      Venue      |                            Paper                             |                             Code                             |
| ------- | ------------------------------------------------------------ | :-------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| 2023.08 | **Using GPT-4 for content moderation (GPT-4)**               |     OpenAI      | [link](https://openai.com/index/using-gpt-4-for-content-moderation/) |                              -                               |
| 2023.02 | **A Holistic Approach to Undesired Content Detection in the Real World (OpenAI Moderation Endpoint)** |   AAAI OpenAI   |           [link](https://arxiv.org/pdf/2208.03274)           |   [link](https://github.com/openai/moderation-api-release)   |
| 2022.02 | **A New Generation of Perspective API: Efficient Multilingual Character-level Transformers (Perspective API)** |   KDD Google    |           [link](https://arxiv.org/pdf/2202.11176)           |             [link](https://perspectiveapi.com/)              |
| -       | **Azure AI Content Safety**                                  | Microsoft Azure |                              -                               | [link](https://azure.microsoft.com/en-us/products/ai-services/ai-content-safety/) |
| -       | **Detoxify**                                                 |   unitary.ai    |                              -                               |        [link](https://github.com/unitaryai/detoxify)         |








### Evaluation \& Analysis
| Time | Title                                                        |  Venue  |                            Paper                             |                             Code                             |
| ---- | ------------------------------------------------------------ | :-----: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| 2025.06 | **Activation Approximations Can Incur Safety Vulnerabilities Even in Aligned LLMs: Comprehensive Analysis and Defense** | USENIX Security'25 | [link](https://www.arxiv.org/pdf/2502.00840) |  [link](https://github.com/Kevin-Zh-CS/QuadA)  |
| 2025.05 | **PandaGuard: Systematic Evaluation of LLM Safety against Jailbreaking Attacks** | arXiv | [link](https://arxiv.org/pdf/2505.13862) |  [link](https://github.com/Beijing-AISI/panda-guard)  |
| 2025.05 | **Assessing Safety Risks and Quantization-aware Safety Patching for Quantized Large Language Models** | ICML'25 | [link](https://icml.cc/virtual/2025/poster/44278) |  [link](https://github.com/Thecommonirin/Qresafe)  |
| 2025.02 | **GuidedBench: Equipping Jailbreak Evaluation with Guidelines** | arXiv | [link](https://arxiv.org/pdf/2502.16903) |  [link](https://github.com/SproutNan/AI-Safety_Benchmark)  |
| 2024.12 | **Agent-SafetyBench: Evaluating the Safety of LLM Agents** | arXiv | [link](https://arxiv.org/pdf/2412.14470) |  [link](https://github.com/thu-coai/Agent-SafetyBench)  |
| 2024.11 | **Global Challenge for Safe and Secure LLMs Track 1** | arXiv | [link](https://arxiv.org/pdf/2411.14502v1) | -  |
| 2024.11 | **JailbreakLens: Interpreting Jailbreak Mechanism in the Lens of Representation and Circuit** | arXiv | [link](https://arxiv.org/pdf/2411.11114v1) | -  |
| 2024.11 | **The VLLM Safety Paradox: Dual Ease in Jailbreak Attack and Defense** | arXiv | [link](https://arxiv.org/pdf/2411.08410v1) | -  |
| 2024.11 | **HarmLevelBench: Evaluating Harm-Level Compliance and the Impact of Quantization on Model Alignment** | arXiv | [link](https://arxiv.org/pdf/2411.06835v1) | - |
| 2024.11 | **ChemSafetyBench: Benchmarking LLM Safety on Chemistry Domain** | arXiv | [link](https://arxiv.org/pdf/2411.16736) | [link](https://github.com/HaochenZhao/SafeAgent4Chem) |
| 2024.11 | **GuardBench: A Large-Scale Benchmark for Guardrail Models** | EMNLP'24 | [link](https://aclanthology.org/2024.emnlp-main.1022.pdf) | [link](https://github.com/AmenRa/guardbench) |
| 2024.11 | **What Features in Prompts Jailbreak LLMs? Investigating the Mechanisms Behind Attacks** | arXiv | [Link](https://arxiv.org/pdf/2411.03343v1) | [link](https://github.com/NLie2/what_features_jailbreak_LLMs) |
| 2024.11 | **Benchmarking LLM Guardrails in Handling Multilingual Toxicity** | arXiv | [link](https://arxiv.org/pdf/2410.22153v1) | [link](https://commoncrawl.github.io/cc-crawl-statistics/plots/languages.html) |
| 2024.10 | **JAILJUDGE: A Comprehensive Jailbreak Judge Benchmark with Multi-Agent Enhanced Explanation Evaluation Framework** | arXiv | [link](https://arxiv.org/pdf/2410.12855) | [link](https://github.com/usail-hkust/Jailjudge) |
| 2024.10 | **Do LLMs Have Political Correctness? Analyzing Ethical Biases and Jailbreak Vulnerabilities in AI Systems** | arXiv | [link](https://arxiv.org/pdf/2410.13334v1) | [link](https://anonymous.4open.science/r/PCJailbreak-F2B0/README.md) |
| 2024.10 | **A Realistic Threat Model for Large Language Model Jailbreaks** | arXiv | [link](https://arxiv.org/pdf/2410.16222v1) | [link](https://github.com/valentyn1boreiko/llm-threat-model) |
| 2024.10 | **ADVERSARIAL SUFFIXES MAY BE FEATURES TOO!** | arXiv | [link](https://arxiv.org/pdf/2410.00451) | [link](https://github.com/suffix-maybe-feature/adver-suffix-maybe-features) |
| 2024.09 | **JAILJUDGE: A COMPREHENSIVE JAILBREAK** | arXiv | [Link](https://openreview.net/pdf?id=cLYvhd0pDY) | [Link](https://anonymous.4open.science/r/public_multiagents_judge-66CB/README.md) |
| 2024.09 | **Multimodal Pragmatic Jailbreak on Text-to-image Models** | arXiv | [link](https://arxiv.org/pdf/2409.19149) | [link](https://github.com/multimodalpragmatic/multimodalpragmatic/tree/main) |
| 2024.08 | **ShieldGemma: Generative AI Content Moderation Based on Gemma (ShieldGemma)** |    arXiv   | [link](https://arxiv.org/pdf/2407.21772) | [link](https://huggingface.co/google/shieldgemma-2b) |
| 2024.08 | **MMJ-Bench: A Comprehensive Study on Jailbreak Attacks and Defenses for Vision Language Models (MMJ-Bench)** |    arXiv   | [link](https://arxiv.org/pdf/2408.08464) | [link](https://github.com/thunxxx/MLLM-Jailbreak-evaluation-MMJ-bench) |
| 2024.08 | **Mission Impossible: A Statistical Perspective on Jailbreaking LLMs** | NeurIPS'24 | [Link](https://arxiv.org/pdf/2408.01420) | - |
| 2024.07 | **Operationalizing a Threat Model for Red-Teaming Large Language Models (LLMs)** |    arXiv   | [link](https://arxiv.org/abs/2407.14937) | [link](https://github.com/dapurv5/awesome-llm-red-teaming) |
| 2024.07 | **JailBreakV-28K: A Benchmark for Assessing the Robustness of MultiModal Large Language Models against Jailbreak Attacks** |    arXiv   | [link](https://arxiv.org/abs/2404.03027) | [link](https://github.com/EddyLuo1232/JailBreakV_28K) |
| 2024.07 | **Jailbreak Attacks and Defenses Against Large Language Models: A Survey** |   arXiv    | [link](https://arxiv.org/abs/2407.04295)  |                     -                         |
| 2024.06 | **"Not Aligned" is Not "Malicious": Being Careful about Hallucinations of Large Language Models' Jailbreak** |   arXiv    | [link](https://arxiv.org/pdf/2406.11668) |                              [link](https://github.com/Meirtz/BabyBLUE-llm)                               |
| 2024.06 | **WildTeaming at Scale: From In-the-Wild Jailbreaks to (Adversarially) Safer Language Models (WildTeaming)** |   NeurIPS'24   | [link](https://arxiv.org/pdf/2406.18510) |                              [link](https://github.com/allenai/wildteaming)                               |
| 2024.06 | **From LLMs to MLLMs: Exploring the Landscape of Multimodal Jailbreaking** |   arXiv    | [link](https://arxiv.org/pdf/2406.14859) |                              -                               |
| 2024.06 | **AI Agents Under Threat: A Survey of Key Security Challenges and Future Pathways** |   arXiv    | [link](https://arxiv.org/pdf/2406.02630) |                              -                               |
| 2024.06 | **MM-SafetyBench: A Benchmark for Safety Evaluation of Multimodal Large Language Models (MM-SafetyBench)** |   arXiv    | [link](https://arxiv.org/pdf/2311.17600) |                              -                              |
| 2024.06 | **ArtPrompt: ASCII Art-based Jailbreak Attacks against Aligned LLMs (VITC)** |   ACL'24    | [link](https://arxiv.org/pdf/2402.11753) |                              [link](https://github.com/uw-nsl/ArtPrompt)                               |
| 2024.06 | **Bag of Tricks: Benchmarking of Jailbreak Attacks on LLMs** |   NeurIPS'24   | [link](https://arxiv.org/pdf/2406.09324) |                             [link](https://github.com/usail-hkust/Bag_of_Tricks_for_LLM_Jailbreaking)                              |
| 2024.06 | **JailbreakZoo: Survey, Landscapes, and Horizons in Jailbreaking Large Language and Vision-Language Models (JailbreakZoo)** |   arXiv    | [link](https://arxiv.org/pdf/2407.01599) |                             [link](https://github.com/Allen-piexl/JailbreakZoo)                              |
| 2024.06 | **Fundamental limitations of alignment in large language models** |   arXiv    | [link](https://arxiv.org/pdf/2304.11082) |                             -                              |
| 2024.06 | **JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models (JailbreakBench)** |   NeurIPS'24   | [link](https://arxiv.org/pdf/2404.01318) |                             [link](https://github.com/JailbreakBench/jailbreakbench)                             |
| 2024.06 | **Towards Understanding Jailbreak Attacks in LLMs: A Representation Space Analysis** |   arXiv    | [link](https://arxiv.org/pdf/2406.10794) |                             [link](https://github.com/yuplin2333/representation-space-jailbreak)                              |
| 2024.06 | **JailbreakEval: An Integrated Toolkit for Evaluating Jailbreak Attempts Against Large Language Models (JailbreakEval)** |   arXiv    | [link](https://arxiv.org/pdf/2406.09321) |                              [link](https://github.com/ThuCCSLab/JailbreakEval)                               |
| 2024.05 | **Rethinking How to Evaluate Language Model Jailbreak** |   arXiv    | [link](https://arxiv.org/pdf/2404.06407) |                             [link](https://github.com/controllability/jailbreak-evaluation)                               |
| 2024.05 | **Enhancing Large Language Models Against Inductive Instructions with Dual-critique Prompting (INDust)** |   arXiv    | [link](https://arxiv.org/pdf/2305.13733) |                             [link](https://github.com/DevoAllen/INDust)                               |
| 2024.05 | **Prompt Injection attack against LLM-integrated Applications** |   arXiv    | [link](https://arxiv.org/pdf/2306.05499) |                             -                               |
| 2024.05 | **Tricking LLMs into Disobedience: Formalizing, Analyzing, and Detecting Jailbreaks** |   LREC-COLING'24    | [link](https://arxiv.org/pdf/2305.14965) |                             [link](https://github.com/AetherPrior/TrickLLM)                               |
| 2024.05 | **LLM Jailbreak Attack versus Defense Techniques--A Comprehensive Study** |   NDSS'24    | [link](https://arxiv.org/pdf/2402.13457) |                             -                               |
| 2024.05 | **Jailbreaking ChatGPT via Prompt Engineering: An Empirical Study** |   arXiv    | [link](https://arxiv.org/pdf/2305.13860) |                             -                               |
| 2024.05 | **Detoxifying Large Language Models via Knowledge Editing (SafeEdit)** |   ACL'24    | [link](https://arxiv.org/pdf/2403.14472) |                             [link](https://github.com/zjunlp/EasyEdit/blob/main/examples/SafeEdit.md)                               |
| 2024.04 | **JailbreakLens: Visual Analysis of Jailbreak Attacks Against Large Language Models (JailbreakLens)** |   arXiv    | [link](https://arxiv.org/pdf/2404.08793) |                             -                               |
| 2024.03 | **How (un) ethical are instruction-centric responses of LLMs? Unveiling the vulnerabilities of safety guardrails to harmful queries (TECHHAZARDQA)** |   arXiv    | [link](https://arxiv.org/pdf/2402.15302) |                              [link](https://huggingface.co/datasets/SoftMINER-Group/TechHazardQA)                               |
| 2024.03 | **Don’t Listen To Me: Understanding and Exploring Jailbreak Prompts of Large Language Models** |   USENIX Security    | [link](https://arxiv.org/pdf/2403.17336) |                             -                               |
| 2024.03 | **EasyJailbreak: A Unified Framework for Jailbreaking Large Language Models (EasyJailbreak)** |   arXiv    | [link](https://arxiv.org/pdf/2403.12171) |                              [link](https://github.com/EasyJailbreak/EasyJailbreak)                               |
| 2024.02 | **Comprehensive Assessment of Jailbreak Attacks Against LLMs** |   arXiv    | [link](https://arxiv.org/abs/2402.05668) |                             -                     |
| 2024.02 | **SPML: A DSL for Defending Language Models Against Prompt Attacks** |   arXiv    | [link](https://arxiv.org/pdf/2402.11755) |                             -                     |
| 2024.02 | **Coercing LLMs to do and reveal (almost) anything** |   arXiv    | [link](https://arxiv.org/pdf/2402.14020) |                             -                     |
| 2024.02 | **A STRONGREJECT for Empty Jailbreaks (StrongREJECT)** |   NeurIPS'24    | [link](https://arxiv.org/pdf/2402.10260) |                             [link](https://github.com/alexandrasouly/strongreject)                      |
| 2024.02 | **ToolSword: Unveiling Safety Issues of Large Language Models in Tool Learning Across Three Stages** |   ACL'24    | [link](https://arxiv.org/pdf/2402.10753) |                             [link](https://github.com/Junjie-Ye/ToolSword)                      |
| 2024.02 | **HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal (HarmBench)** |   arXiv    | [link](https://arxiv.org/pdf/2402.04249) |                              [link](https://github.com/centerforaisafety/HarmBench)                               |
| 2023.12 | **Goal-Oriented Prompt Attack and Safety Evaluation for LLMs** |   arXiv    | [link](https://arxiv.org/pdf/2309.11830)  |                             [link](https://github.com/liuchengyuan123/CPAD)                              |
| 2023.12 | **The Art of Defending: A Systematic Evaluation and Analysis of LLM Defense Strategies on Safety and Over-Defensiveness** |   arXiv    | [link](https://arxiv.org/pdf/2401.00287)  |                            -                              |
| 2023.12 | **A Comprehensive Survey of Attack Techniques, Implementation, and Mitigation Strategies in Large Language Models** |   UbiSec'23    | [link](https://arxiv.org/pdf/2312.10982)  |                            -                              |
| 2023.11 | **Summon a Demon and Bind it: A Grounded Theory of LLM Red Teaming in the Wild** |   arXiv    | [link](https://arxiv.org/pdf/2311.06237)  |                             -                               |
| 2023.11 | **How many unicorns are in this image? a safety evaluation benchmark for vision llms** |   arXiv    | [link](https://arxiv.org/pdf/2311.16101)  |                             [link](https://github.com/UCSC-VLAA/vllm-safety-benchmark)                               |
| 2023.11 | **Exploiting Large Language Models (LLMs) through Deception Techniques and Persuasion Principles** |   arXiv    | [link](https://arxiv.org/pdf/2311.14876)  |                             -                               |
| 2023.10 | **Explore, establish, exploit: Red teaming language models from scratch** |   arXiv    | [link](https://arxiv.org/pdf/2306.09442)  |                             -                               |
| 2023.10 | **Survey of Vulnerabilities in Large Language Models Revealed by Adversarial Attacks** |   arXiv    | [link](https://arxiv.org/pdf/2310.10844)  |                             -                               |
| 2023.10 | **Fine-tuning aligned language models compromises safety, even when users do not intend to! (HEx-PHI)** |   ICLR'24 (oral)    | [link](https://arxiv.org/pdf/2310.03693)  |                             [link](https://github.com/LLM-Tuning-Safety/LLMs-Finetuning-Safety)                               |
| 2023.08 | **Red-Teaming Large Language Models using Chain of Utterances for Safety-Alignment (RED-EVAL)** |   arXiv    | [link](https://arxiv.org/pdf/2308.09662) |                              [link](https://github.com/declare-lab/red-instruct)                                |
| 2023.08 | **Use of LLMs for Illicit Purposes: Threats, Prevention Measures, and Vulnerabilities** |   arXiv    | [link](https://arxiv.org/pdf/2308.12833) |                              -                               |
| 2023.07 | **Jailbroken: How Does LLM Safety Training Fail? (Jailbroken)** |   NeurIPS'23    | [link](https://arxiv.org/pdf/2307.02483#page=1.01)  |                             -                               |
| 2023.08 | **Use of LLMs for Illicit Purposes: Threats, Prevention Measures, and Vulnerabilities** |   arXiv    | [link](https://arxiv.org/pdf/2308.12833)  |                              -                               |
| 2023.08 | **From chatgpt to threatgpt: Impact of generative ai in cybersecurity and privacy** |   IEEE Access    | [link](https://ieeexplore.ieee.org/document/10198233?denied=)  |                              -                               |
| 2023.07 | **Llm censorship: A machine learning challenge or a computer security problem?** |   arXiv    | [link](https://arxiv.org/pdf/2307.10719)  |                              -                               |
| 2023.07 | **Universal and Transferable Adversarial Attacks on Aligned Language Models (AdvBench)** |   arXiv    | [link](https://arxiv.org/pdf/2307.15043)  |                              [link](https://github.com/llm-attacks/llm-attacks)                               |
| 2023.06 | **DecodingTrust: A Comprehensive Assessment of Trustworthiness in GPT Models** |   NeurIPS'23    | [link](https://blogs.qub.ac.uk/digitallearning/wp-content/uploads/sites/332/2024/01/A-comprehensive-Assessment-of-Trustworthiness-in-GPT-Models.pdf)  |                              [link](https://decodingtrust.github.io/)                               |
| 2023.04 | **Safety Assessment of Chinese Large Language Models** |   arXiv    | [link](https://arxiv.org/pdf/2304.10436)  |                              [link](https://github.com/thu-coai/Safety-Prompts)                              |
| 2023.02 | **Exploiting Programmatic Behavior of LLMs: Dual-Use Through Standard Security Attacks** |   arXiv    | [link](https://arxiv.org/pdf/2302.05733)  |                              -                               |
| 2022.11 | **Red Teaming Language Models to Reduce Harms: Methods, Scaling Behaviors, and Lessons Learned** |   arXiv    | [link](https://arxiv.org/pdf/2209.07858)  |                              -                               |
| 2022.02 | **Red Teaming Language Models with Language Models** |   arXiv    | [link](https://arxiv.org/pdf/2202.03286)  |                              -                               |

### Application


| Time | Title                                                        |  Venue  |                            Paper                             |                             Code                             |
| ---- | ------------------------------------------------------------ | :-----: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| 2024.11 | **Attacking Vision-Language Computer Agents via Pop-ups** | arXiv | [link](https://arxiv.org/pdf/2411.02391) | [link](https://github.com/SALT-NLP/PopupAttack) |
| 2024.10 | **Jailbreaking LLM-Controlled Robots (ROBOPAIR)** |   arXiv    | [link](https://arxiv.org/pdf/2410.13691v1) |        [link](https://robopair.org/)      |
| 2024.10 | **SMILES-Prompting: A Novel Approach to LLM Jailbreak Attacks in Chemical Synthesis** |   arXiv    | [link](https://arxiv.org/pdf/2410.15641v1) |        [link](https://github.com/IDEA-XL/ChemSafety)      |
| 2024.10 | **Cheating Automatic LLM Benchmarks: Null Models Achieve High Win Rates** |   arXiv    | [link](https://arxiv.org/pdf/2410.07137) |        [link](https://github.com/sail-sg/Cheating-LLM-Benchmarks)      |
| 2024.09 | **RoleBreak: Character Hallucination as a Jailbreak Attack in Role-Playing Systems** |   arXiv    | [link](https://arxiv.org/pdf/2409.16727) |                              -  |
| 2024.08 | **A Jailbroken GenAI Model Can Cause Substantial Harm: GenAI-powered Applications are Vulnerable to PromptWares (APwT)** |   arXiv    | [link](https://arxiv.org/pdf/2408.05061) |                              -  |



## Other Related Awesome Repository

- [Awesome-LM-SSP](https://github.com/ThuCCSLab/Awesome-LM-SSP)
- [llm-sp](https://github.com/chawins/llm-sp)
- [awesome-llm-security](https://github.com/corca-ai/awesome-llm-security)
- [Awesome-LLM-Safety](https://github.com/ydyjya/Awesome-LLM-Safety)
- [Awesome-LRMs-Safety](https://github.com/WangCheng0116/Awesome-LRMs-Safety)
- [Awesome-LALMs-Jailbreak](https://github.com/WangCheng0116/Awesome_LALMs_Jailbreak)






## Contributors

<a href="https://github.com/yueliu1999" target="_blank"><img src="https://avatars.githubusercontent.com/u/41297969?s=64&v=4" alt="yueliu1999" width="96" height="96"/></a> 
<a href="https://github.com/bhooi" target="_blank"><img src="https://avatars.githubusercontent.com/u/733939?v=4" alt="bhooi" width="96" height="96"/></a> 
<a href="https://github.com/zqypku" target="_blank"><img src="https://avatars.githubusercontent.com/u/71053864?v=4" alt="zqypku" width="96" height="96"/></a> 
<a href="https://github.com/jiaxiaojunQAQ" target="_blank"><img src="https://avatars.githubusercontent.com/u/23453472?v=4" alt="jiaxiaojunQAQ" width="96" height="96"/></a> 
<a href="https://github.com/Huang-yihao" target="_blank"><img src="https://avatars.githubusercontent.com/u/16575311?v=4" alt="Huang-yihao" width="96" height="96"/></a> 
<a href="https://github.com/csyuhao" target="_blank"><img src="https://avatars.githubusercontent.com/u/24415219?v=4" alt="csyuhao" width="96" height="96"/></a> 
<a href="https://github.com/xszheng2020" target="_blank"><img src="https://avatars.githubusercontent.com/u/101038474?v=4" alt="xszheng2020" width="96" height="96"/></a> 
<a href="https://github.com/dapurv5" target="_blank"><img src="https://avatars.githubusercontent.com/u/654346?v=4" alt="dapurv5" width="96" height="96"/></a> 
<a href="https://github.com/ZYQ-Zoey77" target="_blank"><img src="https://avatars.githubusercontent.com/u/72439991?v=4" alt="ZYQ-Zoey77" width="96" height="96"/></a> 
<a href="https://github.com/mdoumbouya" target="_blank"><img src="https://avatars.githubusercontent.com/u/8138248?v=4" alt="mdoumbouya" width="96" height="96"/></a> 
<a href="https://github.com/xyliugo" target="_blank"><img src="https://avatars.githubusercontent.com/u/118035138?v=4" alt="xyliugo" width="96" height="96"/></a> 
<a href="https://github.com/zky001" target="_blank"><img src="https://avatars.githubusercontent.com/u/9131265?v=4" alt="zky001" width="96" height="96"/></a> 






<p align="right">(<a href="#top">back to top</a>)</p>























