# A Curated Guide to the Awesome LLMOps Projects

## Introduction to LLMOps

LLMOps (Large Language Model Operations) is a specialized discipline of MLOps tailored to the unique challenges of managing the entire lifecycle of LLM-powered applications. As organizations move from experimenting with LLMs to deploying them in production, they face distinct hurdles that traditional MLOps practices do not fully address. These challenges include complex prompt engineering, continuous fine-tuning, managing Retrieval-Augmented Generation (RAG) pipelines, handling high computational costs for inference, and monitoring for specific failure modes like hallucinations, toxicity, and data privacy leakage.

LLMOps provides the principles, practices, and tools necessary to build, deploy, and maintain these applications in a reliable, scalable, and efficient manner. This guide organizes a curated list of high-relevance, open-source tools according to the core stages of the LLMOps lifecycle, providing a top-down workflow from initial concept to production monitoring.

## Table of Contents

  - [Phase 1: Development & Experimentation](https://www.google.com/search?q=%23phase-1-development--experimentation)
      - [1.1 Data & Knowledge Management](https://www.google.com/search?q=%2311-data--knowledge-management)
      - [1.2 Workflow & Agent Development](https://www.google.com/search?q=%2312-workflow--agent-development)
      - [1.3 Experiment Tracking & Evaluation](https://www.google.com/search?q=%2313-experiment-tracking--evaluation)
  - [Phase 2: Model Fine-Tuning & Optimization](https://www.google.com/search?q=%23phase-2-model-fine-tuning--optimization)
  - [Phase 3: Deployment & Serving](https://www.google.com/search?q=%23phase-3-deployment--serving)
      - [3.1 High-Performance Inference & Serving](https://www.google.com/search?q=%2331-high-performance-inference--serving)
      - [3.2 Model Deployment & Packaging](https://www.google.com/search?q=%2332-model-deployment--packaging)
  - [Phase 4: In-Production Operations](https://www.google.com/search?q=%23phase-4-in-production-operations)
      - [4.1 Observability, Monitoring & Cost Management](https://www.google.com/search?q=%2341-observability-monitoring--cost-management)
      - [4.2 Security & Guardrails](https://www.google.com/search?q=%2342-security--guardrails)
  - [Comprehensive Solutions: End-to-End Platforms](https://www.google.com/search?q=%23comprehensive-solutions-end-to-end-platforms)

-----

### Phase 1: Development & Experimentation

**Goal**: To build, iterate, and optimize the core logic of LLM applications, serving as the foundation for all innovation.

#### 1.1 Data & Knowledge Management

*Description*: Preparing, managing, and versioning data for RAG and fine-tuning, which is the cornerstone of building intelligent applications.

| Project | Details | Repository |
| :--- | :--- | :--- |
| [RagFlow](https://github.com/infiniflow/ragflow) | An open-source RAG application that provides a streamlined workflow based on deep document understanding. |  |
| [FastGPT](https://github.com/labring/FastGPT) | A platform that based on LLM, allows you to create your own knowledge base QA model with out-of-the-box capabilities. |  |
| [Chroma](https://github.com/chroma-core/chroma) | the open source embedding database |  |
| [Milvus](https://github.com/milvus-io/milvus) | Vector database for scalable similarity search and AI applications. |  |
| [Pinecone-Client](https://github.com/pinecone-io/pinecone-python-client) | The Pinecone vector database makes it easy to build high-performance vector search applications. |  |
| [Qdrant](https://github.com/qdrant/qdrant) | Vector Search Engine and Database for the next generation of AI applications. |  |
| [Weaviate](https://github.com/semi-technologies/weaviate) | Weaviate is an open source vector search engine that stores both objects and vectors. |  |
| [Lancedb](https://github.com/lancedb/lancedb) | Developer-friendly, serverless vector database for AI applications. |  |
| [DVC](https://github.com/iterative/dvc) | Data Version Control - Git for Data & Models - ML Experiments Management. |  |
| [deeplake](https://github.com/activeloopai/deeplake) | Data Lake for Deep Learning. Build, manage, query, version, & visualize datasets. Stream data in real-time to PyTorch/TensorFlow. |  |
| [LakeFS](https://github.com/treeverse/lakeFS) | Git-like capabilities for your object storage. |  |

#### 1.2 Workflow & Agent Development

*Description*: Orchestrating application logic to build intelligent agents capable of executing complex tasks.

| Project | Details | Repository |
| :--- | :--- | :--- |
| [Flowise](https://github.com/FlowiseAI/Flowise) | Drag & drop UI to build your customized LLM flow using LangchainJS. |  |
| [LangFlow](https://github.com/logspace-ai/langflow) | An effortless way to experiment and prototype LangChain flows with a chat interface. |  |
| [DB-GPT](https://github.com/eosphoros-ai/DB-GPT) | Revolutionizing Data Interactions with Private LLM Technology and a data-driven agent framework. |  |
| [langchain](https://github.com/hwchase17/langchain) | Building applications with LLMs through composability |  |
| [LlamaIndex](https://github.com/jerryjliu/llama_index) | Provides a central interface to connect your LLMs with external data. |  |
| [Hamilton](https://github.com/dagworks-inc/hamilton) | A lightweight framework to represent ML/language model pipelines as a series of python functions. |  |

#### 1.3 Experiment Tracking & Evaluation

*Description*: Recording and comparing experimental results, and systematically evaluating application performance to support data-driven decisions.

| Project | Details | Repository |
| :--- | :--- | :--- |
| [MLflow](https://github.com/mlflow/mlflow) | An open-source framework for the end-to-end machine learning lifecycle, helping developers track experiments, evaluate models/prompts, and more. |  |
| [Weights & Biases](https://github.com/wandb/wandb) | A developer first MLOps platform for experiment tracking, dataset versioning, and model management. Featuring W\&B Prompts for LLM execution flow visualization. |  |
| [promptfoo](https://github.com/typpo/promptfoo) | Open-source tool for testing & evaluating prompt quality. |  |
| [Vellum-Python](https://www.google.com/search?q=https://github.com/vellum-ai/vellum-python) | An AI product development platform to experiment with, evaluate, and deploy advanced LLM apps. |  |
| [LangWatch](https://github.com/langwatch/langwatch) | Visualize LLM evaluations experiments and DSPy pipeline optimizations |  |
| [Aim](https://github.com/aimhubio/aim) | an easy-to-use and performant open-source experiment tracker. |  |

-----

### Phase 2: Model Fine-Tuning & Optimization

**Goal**: To specialize a general-purpose large model for a specific domain or task by training it on a targeted dataset, achieving superior performance.

| Project | Details | Repository |
| :--- | :--- | :--- |
| [peft](https://github.com/huggingface/peft) | State-of-the-art Parameter-Efficient Fine-Tuning. |  |
| [QLoRA](https://github.com/artidoro/qlora) | Finetune a 65B parameter model on a single 48GB GPU while preserving full 16-bit finetuning task performance. |  |
| [TRL](https://github.com/huggingface/trl) | Train transformer language models with reinforcement learning. |  |
| [axolotl](https://github.com/OpenAccess-AI-Collective/axolotl) | A tool designed to streamline the fine-tuning of various AI models. |  |
| [FastEdit](https://github.com/hiyouga/FastEdit) | FastEdit aims to assist developers with injecting fresh and customized knowledge into large language models efficiently. |  |

-----

### Phase 3: Deployment & Serving

**Goal**: To deploy developed and optimized model applications into a production environment efficiently, reliably, and scalably.

#### 3.1 High-Performance Inference & Serving

| Project | Details | Repository |
| :--- | :--- | :--- |
| [vllm](https://github.com/vllm-project/vllm) | A high-throughput and memory-efficient inference and serving engine for LLMs. |  |
| [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | Inference engine for TensorRT on Nvidia GPUs |  |
| [Ollama](https://github.com/jmorganca/ollama) | Serve Llama 2 and other large language models locally from command line or through a browser interface. |  |

#### 3.2 Model Deployment & Packaging

| Project | Details | Repository |
| :--- | :--- | :--- |
| [BentoML](https://github.com/bentoml/BentoML) | The Unified Model Serving Framework |  |
| [OpenLLM](https://github.com/bentoml/OpenLLM) | An open platform for operating large language models (LLMs) in production. |  |
| [Kserve](https://github.com/kserve/kserve) | Standardized Serverless ML Inference Platform on Kubernetes |  |
| [Triton Server](https://github.com/triton-inference-server/server) | The Triton Inference Server provides an optimized cloud and edge inferencing solution. |  |

-----

### Phase 4: In-Production Operations

**Goal**: To ensure the stability, security, and quality of online services through continuous monitoring and iteration.

#### 4.1 Observability, Monitoring & Cost Management

| Project | Details | Repository |
| :--- | :--- | :--- |
| [Helicone](https://github.com/Helicone/helicone) | Open source LLM observability platform for logging, monitoring, and debugging. |  |
| [Portkey-SDK](https://github.com/Portkey-AI/portkey-python-sdk) | Control Panel with an observability suite & an AI gateway — to ship fast, reliable, and cost-efficient apps. |  |
| [Langfuse](https://github.com/langfuse/langfuse) | Open Source LLM Engineering Platform: Traces, evals, prompt management and metrics to debug and improve your LLM application. |  |
| [Arize-Phoenix](https://github.com/Arize-ai/phoenix) | ML observability for LLMs, vision, language, and tabular models. |  |
| [Evidently](https://github.com/evidentlyai/evidently) | An open-source framework to evaluate, test and monitor ML and LLM-powered systems. |  |

#### 4.2 Security & Guardrails

| Project | Details | Repository |
| :--- | :--- | :--- |
| [Plexiglass](https://github.com/kortex-labs/plexiglass) | A Python Machine Learning Pentesting Toolbox for Adversarial Attacks. Works with LLMs. |  |
| [Giskard](https://github.com/Giskard-AI/giskard) | Testing framework dedicated to ML models, from tabular to LLMs. Detect risks of biases, performance issues and errors. |  |
| [Deepchecks](https://github.com/deepchecks/deepchecks) | Tests for Continuous Validation of ML Models & Data. |  |

-----

### Comprehensive Solutions: End-to-End Platforms

**Goal**: To provide comprehensive solutions covering multiple stages of the lifecycle, simplifying the overall workflow for teams that need to quickly establish a complete system.

| Project | Details | Repository |
| :--- | :--- | :--- |
| [Dify](https://github.com/langgenius/dify) | An open-source LLM app development platform for building and operating generative AI-native applications. |  |
| [TrueFoundry-Py](https://www.google.com/search?q=https://github.com/truefoundry/truefoundry-py) | A PaaS to deploy, Fine-tune and serve LLM Models on your own Infrastructure with Data Security and Optimal GPU Management. |  |
| [ZenML](https://github.com/zenml-io/zenml) | MLOps framework to create reproducible pipelines. |  |
| [Kubeflow](https://github.com/kubeflow/kubeflow) | Machine Learning Toolkit for Kubernetes. |  |
