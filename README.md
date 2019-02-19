# Deep NLP 2 (2019.3-5)
기계학습 기반 자연어 처리(Natural Language Processing) 방법론 스터디입니다. 널리 사용되는 Word2Vec (2012)부터, 2019년도 연구 동향까지 다룹니다. 자세한 일정, 인원, 장소 및 내용은 아래와 같습니다.

- 일정: 2019.3.9~ (토) / 매주 토요일 오후 2시-5시 (주 1회, 3시간)
- 인원: 10명 내외 (현재 6명 참여)
- 장소 : 스테이지나인 삼성점 [네이버 지도](https://map.naver.com/local/siteview.nhn?code=1042227506)
- 참여가능한 배경지식 : 기계학습 기초 [모두를 위한 딥러닝: 기본](https://hunkim.github.io/ml/)

## Part 1 : Natural Language Processing Basics (Week 1-4)

### **Week 1 : Introduction / Overview**
- Welcome!
- Introduction to Machine Learning & Deep Learning
- Introduction to Natural Language Processing
- Course Overview / Logistics

### Week 2 : Word, Sentence, and Document Embedding
- **Word2Vec** Distributed Representations of Words and Phrases and their Compositionality (NIPS 2012) [Link](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)
- **Word2Vec 2** Efficient Estimation of Word Representations in Vector Space (ICLR 2013) [Link](https://arxiv.org/pdf/1301.3781.pdf)
- **Glove** GloVe: Global Vectors for Word Representation (EMNLP 2014) [Link](https://nlp.stanford.edu/pubs/glove.pdf)
- **FastText** Enriching Word Vectors with Subword Information (TACL 2017) [Link](https://arxiv.org/pdf/1607.04606.pdf)
- **FastText for Korean** Subword-level Word Vector Representations for Korean (ACL 2018) [Link](http://aclweb.org/anthology/P18-1226)
- **Skip-thought** Skip-Thought Vectors (NIPS 2015) [Link](https://papers.nips.cc/paper/5950-skip-thought-vectors.pdf)
- **Doc2Vec** Distributed Representations of Sentences and Documents (ICML 2014)  [Link](https://cs.stanford.edu/~quocle/paragraph_vector.pdf)

### Week 3 : Document Classification
- **Document Classification Basics**
- **Recursive Neural Networks** Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank (EMNLP 2013) [Link](https://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf)
- **CNN** Convolutional Neural Networks for Sentence Classification (EMNLP 2014) [Link](https://arxiv.org/pdf/1408.5882.pdf)
- **CNN 2** Character-level Convolutional Networks for Text Classification (NIPS 2015) [Link](https://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classification.pdf)
- **Attention** Hierarchical Attention Networks for Document Classification (NAACL 2016) [Link](https://www.cs.cmu.edu/~hovy/papers/16HLT-hierarchical-attention-networks.pdf)

### Week 4 : Language Modeling & Transfer Learning
- **Language Modeling Basics** A Neural Probabilistic Language Model (JMLR 2003) [Link](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
- **CNN** Character-Aware Neural Language Models (AAAI 2016) [Link](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/viewFile/12489/12017)
- **RNN** Regularizing and Optimizing LSTM Language Models (ICLR 2018) [Link](https://openreview.net/pdf?id=SyyGPP0TZ)
- **ELMo** Deep Contextualized Word Representations (NAACL 2018) [Link](https://arxiv.org/pdf/1802.05365.pdf)
- **ULMFiT** Universal Language Model Fine-tuning for Text Classification (ACL 2018) [Link](https://arxiv.org/pdf/1801.06146.pdf)

## Part 2 : Advanced Topics in Natural Language Processing (Week 5-8)

### Week 5 : Machine Translation
- **Seq2Seq Basics**
- **GRU** Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation (EMNLP 2014) [Link](https://www.aclweb.org/anthology/D14-1179)
- **CNN** Convolutional Sequence to Sequence Learning (ICML 2017) [Link](http://proceedings.mlr.press/v70/gehring17a/gehring17a.pdf)
- **Transformer** Attention Is All You Need (NIPS 2017) [Link](https://arxiv.org/pdf/1706.03762.pdf)
- **BERT** BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (EMNLP 2018) [Link](https://arxiv.org/pdf/1810.04805.pdf)
- **Unsupervised MT** Word Translation Without Parallel Data (ICLR 2018) [Link](https://arxiv.org/pdf/1710.04087.pdf)

### Week 6 : Conversation Modeling & Response Generation
- **HRED** Building End-To-End Dialogue Systems Using Generative Hierarchical Neural Network Models (AAAI 2016) [Link](https://arxiv.org/pdf/1507.04808.pdf)
- **Variational AutoEncoder (VAE) Basics** Auto-Encoding Variational Bayes [Link(tutorial)](https://arxiv.org/pdf/1606.05908.pdf)
- **VHRED** A Hierarchical Latent Variable Encoder-Decoder Model for Generating Dialogues (AAAI 2017) [Link](http://www.cs.toronto.edu/~lcharlin/papers/vhred_aaai17.pdf)
- **VHCR** A Hierarchical Latent Structure for Variational Conversation Modeling (NAACL 2018) [Link](https://arxiv.org/pdf/1804.03424.pdf)
- **Generative Adversarial Network (GAN) Basics** Generative Adversarial Nets (NIPS 2014) [Link](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf)
- **DialogueWAE** DialogWAE: Multimodal Response Generation with Conditional Wasserstein Auto-Encoder (ICLR 2019) [Link](https://openreview.net/pdf?id=BkgBvsC9FQ)

### Week 7 : Question & Answering / Summarization
- **Memory Networks** Memory Networks (ICLR 2015) [Link](https://arxiv.org/pdf/1410.3916.pdf)
- **End-To-End Memory Networks** End-To-End Memory Networks (NIPS 2015) [Link](https://papers.nips.cc/paper/5846-end-to-end-memory-networks.pdf)
- **SQUAD 2.0** Know What You Don’t Know: Unanswerable Questions for SQuAD (ACL 2018) [Link](http://www.aclweb.org/anthology/P18-2124)
- **Pointer Networks** Get To The Point: Summarization with Pointer-Generator Networks (ACL 2017) [Link](http://aclweb.org/anthology/P17-1099)

### Week 8 : Recent (2019) Trends in NLP / Other topics
- **Hyperbolic Embeddings** Poincaré GloVe: Hyperbolic Word Embeddings (ICLR 2019, accepted) [Link](https://openreview.net/pdf?id=Ske5r3AqK7)
- **SOTA Language Models: GPT-2** Language Models are Unsupervised Multitask Learners [Link](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- **Computational Psychotherapy** Large-scale Analysis of Counseling Conversations: An Application of Natural Language Processing to Mental Health (TACL 2016) [Link](http://www.aclweb.org/anthology/Q16-1033)
- **Computational Psychotherapy 2** Conversation Model Fine-Tuning for Classifying Client Utterances in Counseling Dialogues [Link](https://naacl2019.org/)
- **Closing Remarks**

## Part 3 : Auto-correcting of Sentences via Out-of-vocubulary Generation for Korean Documents (Collaborative Project)
- **TBD**
