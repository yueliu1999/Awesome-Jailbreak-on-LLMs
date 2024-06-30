# Awesome-Jailbreak-on-LLMs

> Awesome-Jailbreak-on-LLMs is a collection of state-of-the-art, novel, exciting jailbreak methods on LLMs. It contains papers, codes, datasets, and evaluations. Any additional things regarding jailbreak are welcome. Any problems, please contact yueliu19990731@163.com. If you find this repository useful to your research or work, it is really appreciated to star this repository. :sparkles:






## Papers


### Jailbreak Attack



#### White-box Attack
| Year | Title                                                        |  Venue  |                            Paper                             |                             Code                             |
| ---- | ------------------------------------------------------------ | :-----: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| 2024.05 | **AutoDAN: Generating Stealthy Jailbreak Prompts on Aligned Large Language Models (AutoDAN)** |   ICLR    | [link](https://arxiv.org/pdf/2310.04451)  |                              [link](https://github.com/SheltonLiu-N/AutoDAN)                               |
| 2023.07 | **Universal and Transferable Adversarial Attacks on Aligned Language Models (GCG)** |   arXiv    | [link](https://arxiv.org/pdf/2307.15043)  |                              [link](https://github.com/llm-attacks/llm-attacks)                               |


Attacking large language models with projected gradient descent ()
Jailbreaking leading safety-aligned llms with simple adaptive attacks


#### Black-box Attack

| Time | Title                                                        |  Venue  |                            Paper                             |                             Code                             |
| ---- | ------------------------------------------------------------ | :-----: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| 2024.06 | **When LLM Meets DRL: Advancing Jailbreaking Efficiency via DRL-guided Search (RLbreaker)** |   arXiv    | [link](https://arxiv.org/pdf/2406.08705) |                              -                               |
| 2024.06 | **A Wolf in Sheep’s Clothing: Generalized Nested Jailbreak Prompts can Fool Large Language Models Easily (ReNeLLM)** |   NAACL    | [link](https://arxiv.org/abs/2311.08268) |                              [link](https://github.com/NJUNLP/ReNeLLM)                               |
| 2024.06 | **QROA: A Black-Box Query-Response Optimization Attack on LLMs (QROA)** |   arXiv    | [link](https://arxiv.org/abs/2406.02044) |                              [link](https://github.com/qroa/qroa)                               |
| 2024.05 | **Multilingual Jailbreak Challenges in Large Language Models** |   ICLR    | [link](https://arxiv.org/pdf/2310.06474)  |                              [link](https://github.com/DAMO-NLP-SG/multilingual-safety-for-LLMs)                               |
| 2024.05 | **GPT-4 Jailbreaks Itself with Near-Perfect Success Using Self-Explanation (IRIS)** |   ACL    | [link](https://arxiv.org/abs/2405.13077) |                              -                               |
| 2024.02 | **PAL: Proxy-Guided Black-Box Attack on Large Language Models (PAL)** |   arXiv    | [link](https://arxiv.org/abs/2402.09674) |                              [link](https://github.com/chawins/pal)                               |
| 2024.01 | **How Johnny Can Persuade LLMs to Jailbreak Them: Rethinking Persuasion to Challenge AI Safety by Humanizing LLMs (PAP)** |   arXiv    | [link](https://arxiv.org/pdf/2401.06373) |                              [link](https://github.com/CHATS-lab/persuasive_jailbreaker)                               |
| 2023.12 | **Tree of Attacks: Jailbreaking Black-Box LLMs Automatically (TAP)** |   arXiv    | [link](https://arxiv.org/abs/2312.02119) |                              [link](https://github.com/RICommunity/TAP)                               |
| 2023.10 | **Jailbreaking Black Box Large Language Models in Twenty Queries (PAIR)** |   arXiv    | [link](https://arxiv.org/pdf/2310.08419) |                              [link](https://github.com/patrickrchao/JailbreakingLLMs)                               |
| 2023.09 | **Open Sesame! Universal Black Box Jailbreaking of Large Language Models (Open Sesame)** |   arXiv    | [link](https://arxiv.org/abs/2309.01446) |                              -                               |
| 2023.09 | **GPTFUZZER: Red Teaming Large Language Models with Auto-Generated Jailbreak Prompts (GPTFuzz)** |   arXiv    | [link](https://arxiv.org/abs/2309.10253) |                              [link](https://github.com/sherdencooper/GPTFuzz)                               |






### Jailbreak Defense

| Time | Title                                                        |  Venue  |                            Paper                             |                             Code                             |
| ---- | ------------------------------------------------------------ | :-----: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| 2024.06 | **A Wolf in Sheep’s Clothing: Generalized Nested Jailbreak Prompts can Fool Large Language Models Easily (ReNeLLM)** |   NAACL    | [link](https://arxiv.org/abs/2311.08268) |                              [link](https://github.com/NJUNLP/ReNeLLM)                               |
| 2024.05 | **Multilingual Jailbreak Challenges in Large Language Models** |   ICLR    | [link](https://arxiv.org/pdf/2310.06474)  |                              [link](https://github.com/DAMO-NLP-SG/multilingual-safety-for-LLMs)                               |
| 2024.01 | **How Johnny Can Persuade LLMs to Jailbreak Them: Rethinking Persuasion to Challenge AI Safety by Humanizing LLMs (PAP)** |   arXiv    | [link](https://arxiv.org/pdf/2401.06373) |                              [link](https://github.com/CHATS-lab/persuasive_jailbreaker)                               |


<!-- Smoothllm: Defending large language models against jailbreaking attacks




Many-shot jailbreaking



Jailbreaking chatgpt via prompt engineering: An empirical study

Tricking LLMs into Disobedience: Formalizing, Analyzing, and Detecting Jailbreaks

" do anything now": Characterizing and evaluating in-the-wild jailbreak prompts on large language models

JailbreakEval: An Integrated Toolkit for Evaluating Jailbreak Attempts Against Large Language Models

Multi-step jailbreaking privacy attacks on chatgpt

Red teaming language models with language models

Jailbreaking gpt-4v via self-adversarial attacks with system prompts

Prompt Injection attack against LLM-integrated Applications


Jailbreak Vision Language Models via Bi-Modal Adversarial Prompt

Cold-attack: Jailbreaking llms with stealthiness and controllability

Autodefense: Multi-agent llm defense against jailbreak attacks

Voice Jailbreak Attacks Against GPT-4o

AutoJailbreak: Exploring Jailbreak Attacks and Defenses through a Dependency Lens


Towards Understanding Jailbreak Attacks in LLMs: A Representation Space Analysis


--
Gpt-4 is too smart to be safe: Stealthy chat with llms via cipher (读)

Low-resource languages jailbreak gpt-4
：
low-resource language
--

---
Use of llms for illicit purposes: Threats, prevention measures, and vulnerabilities
Exploiting programmatic behavior of llms: Dual-use through standard security attacks

：

use programmatic behaviors, such as code injection
and virtualization, to expose LLM vulnerabilities.
---

Jailbreaker: Automated jailbreak across multiple large language model chatbots (读)

--
Jailbreak and guard aligned language models with only few in-context demonstrations
Adversarial demonstration attacks on large language models
:
in-context persona
--

Automatically auditing large language models via discrete optimization (ARCA)
maybe gradient
 -->
