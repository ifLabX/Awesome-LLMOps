# A Curated Guide to the Awesome LLMOps Projects

## Introduction to LLMOps

LLMOps (Large Language Model Operations) is a specialized discipline of MLOps tailored to the unique challenges of managing the entire lifecycle of LLM-powered applications. As organizations move from experimenting with LLMs to deploying them in production, they face distinct hurdles that traditional MLOps practices do not fully address. These challenges include complex prompt engineering, continuous fine-tuning, managing Retrieval-Augmented Generation (RAG) pipelines, handling high computational costs for inference, and monitoring for specific failure modes like hallucinations, toxicity, and data privacy leakage.

LLMOps provides the principles, practices, and tools necessary to build, deploy, and maintain these applications in a reliable, scalable, and efficient manner. This guide organizes a curated list of high-relevance, open-source tools according to the core stages of the LLMOps lifecycle, providing a top-down workflow from initial concept to production monitoring.

## Table of Contents

- [Phase 1: Development & Experimentation](#phase-1-development--experimentation)
  - [1.1 Data & Knowledge Management](#11-data--knowledge-management)
  - [1.2 Prompt Engineering & Optimization](#12-prompt-engineering--optimization)
  - [1.3 Workflow & Agent Development](#13-workflow--agent-development)
  - [1.4 Experiment Tracking & Evaluation](#14-experiment-tracking--evaluation)
- [Phase 2: Model Fine-Tuning & Optimization](#phase-2-model-fine-tuning--optimization)
- [Phase 3: Deployment & Serving](#phase-3-deployment--serving)
  - [3.1 High-Performance Inference & Serving](#31-high-performance-inference--serving)
  - [3.2 Model Deployment & Packaging](#32-model-deployment--packaging)
- [Phase 4: In-Production Operations](#phase-4-in-production-operations)
  - [4.1 Observability, Monitoring & Cost Management](#41-observability-monitoring--cost-management)
  - [4.2 Security & Guardrails](#42-security--guardrails)


---

### Phase 1: Development & Experimentation
**Goal**: To build, iterate, and optimize the core logic of LLM applications, serving as the foundation for all innovation.

#### 1.1 Data & Knowledge Management
*Description*: Preparing, managing, and versioning data for RAG and fine-tuning, which is the cornerstone of building intelligent applications.

| Project | Details | Repository |
| :--- | :--- | :--- |
| [DVC](https://github.com/iterative/dvc) | Data Version Control - Git for Data & Models - ML Experiments Management. | ![GitHub Badge](https://img.shields.io/github/stars/iterative/dvc.svg?style=flat-square) |
| [deeplake](https://github.com/activeloopai/deeplake) | Data Lake for Deep Learning. Build, manage, query, version, & visualize datasets. Stream data in real-time to PyTorch/TensorFlow. | ![GitHub Badge](https://img.shields.io/github/stars/activeloopai/Hub.svg?style=flat-square) |
| [LakeFS](https://github.com/treeverse/lakeFS) | Git-like capabilities for your object storage. | ![GitHub Badge](https://img.shields.io/github/stars/treeverse/lakeFS.svg?style=flat-square) |

#### 1.2 Prompt Engineering & Optimization
*Description*: Systematically designing, testing, versioning, and optimizing prompts to ensure stable and high-quality model outputs.

| Project | Details | Repository |
| :--- | :--- | :--- |
| [promptfoo](https://github.com/typpo/promptfoo) | Open-source tool for testing & evaluating prompt quality. | ![GitHub Badge](https://img.shields.io/github/stars/typpo/promptfoo.svg?style=flat-square) |
| [Agenta](https://github.com/agenta-ai/agenta) | An open-source LLMOps platform with tools for prompt management, evaluation, and deployment. | ![GitHub Badge](https://img.shields.io/github/stars/agenta-ai/agenta.svg?style=flat-square) |
| [Vellum-Python](https://github.com/vellum-ai/vellum-python) | An AI product development platform to experiment with, evaluate, and deploy advanced LLM apps. | ![GitHub Badge](https://img.shields.io/github/stars/vellum-ai/vellum-python.svg?style=flat-square) |
| [Promptimizer](https://github.com/shobrook/promptimal) | A very fast, very minimal prompt optimizer using genetic algorithms. | ![GitHub Badge](https://img.shields.io/github/stars/shobrook/promptimal.svg?style=flat-square) |
| [DSPy](https://github.com/stanford-oval/dspy) | A framework for programming—not just prompting—language models. It allows you to optimize prompts and weights. | ![GitHub Badge](https://img.shields.io/github/stars/stanford-oval/dspy.svg?style=flat-square) |

#### 1.3 Workflow & Agent Development
*Description*: Orchestrating application logic to build intelligent agents capable of executing complex tasks.

| Project | Details | Repository |
| :--- | :--- | :--- |
| [Dify](https://github.com/langgenius/dify) | An open-source LLM app development platform for building and operating generative AI-native applications. | ![GitHub Badge](https://img.shields.io/github/stars/langgenius/dify.svg?style=flat-square) |
| [RagFlow](https://github.com/infiniflow/ragflow) | An open-source RAG application that provides a streamlined workflow based on deep document understanding. | ![GitHub Badge](https://img.shields.io/github/stars/infiniflow/ragflow.svg?style=flat-square) |
| [FastGPT](https://github.com/labring/FastGPT) | A platform that based on LLM, allows you to create your own knowledge base QA model with out-of-the-box capabilities. | ![GitHub Badge](https://img.shields.io/github/stars/labring/FastGPT.svg?style=flat-square) |
| [Flowise](https://github.com/FlowiseAI/Flowise) | Drag & drop UI to build your customized LLM flow using LangchainJS. | ![GitHub Badge](https://img.shields.io/github/stars/FlowiseAI/Flowise.svg?style=flat-square) |
| [LangFlow](https://github.com/logspace-ai/langflow) | An effortless way to experiment and prototype LangChain flows with a chat interface. | ![GitHub Badge](https://img.shields.io/github/stars/logspace-ai/langflow.svg?style=flat-square) |
| [DB-GPT](https://github.com/eosphoros-ai/DB-GPT) | Revolutionizing Data Interactions with Private LLM Technology and a data-driven agent framework. | ![GitHub Badge](https://img.shields.io/github/stars/eosphoros-ai/DB-GPT.svg?style=flat-square) |
| [ZenML](https://github.com/zenml-io/zenml) | MLOps framework to create reproducible pipelines for ML and LLM workflows. | ![GitHub Badge](https://img.shields.io/github/stars/zenml-io/zenml.svg?style=flat-square) |
| [langchain](https://github.com/hwchase17/langchain) | Building applications with LLMs through composability | ![GitHub Badge](https://img.shields.io/github/stars/hwchase17/langchain.svg?style=flat-square) |
| [LlamaIndex](https://github.com/jerryjliu/llama_index) | Provides a central interface to connect your LLMs with external data. | ![GitHub Badge](https://img.shields.io/github/stars/jerryjliu/llama_index.svg?style=flat-square) |
| [Hamilton](https://github.com/dagworks-inc/hamilton) | A lightweight framework to represent ML/language model pipelines as a series of python functions. | ![GitHub Badge](https://img.shields.io/github/stars/dagworks-inc/hamilton.svg?style=flat-square) |

#### 1.4 Experiment Tracking & Evaluation
*Description*: Recording and comparing experimental results, and systematically evaluating application performance to support data-driven decisions.

| Project | Details | Repository |
| :--- | :--- | :--- |
| [MLflow](https://github.com/mlflow/mlflow) | An open-source framework for the end-to-end machine learning lifecycle, helping developers track experiments, evaluate models/prompts, and more. | ![GitHub Badge](https://img.shields.io/github/stars/mlflow/mlflow.svg?style=flat-square) |
| [Weights & Biases](https://github.com/wandb/wandb) | A developer first MLOps platform for experiment tracking, dataset versioning, and model management. Featuring W&B Prompts for LLM execution flow visualization. | ![GitHub Badge](https://img.shields.io/github/stars/wandb/wandb.svg?style=flat-square) |
| [LangWatch](https://github.com/langwatch/langwatch) | Visualize LLM evaluations experiments and DSPy pipeline optimizations | ![GitHub Badge](https://img.shields.io/github/stars/langwatch/langwatch.svg?style=flat-square) |
| [Aim](https://github.com/aimhubio/aim) | an easy-to-use and performant open-source experiment tracker. | ![GitHub Badge](https://img.shields.io/github/stars/aimhubio/aim.svg?style=flat-square) |
| [Arize-Phoenix](https://github.com/Arize-ai/phoenix) | ML observability for LLMs, vision, language, and tabular models. Also offers powerful local evaluation capabilities. | ![GitHub Badge](https://img.shields.io/github/stars/Arize-ai/phoenix.svg?style=flat-square) |
| [Evidently](https://github.com/evidentlyai/evidently) | An open-source framework to evaluate, test and monitor ML and LLM-powered systems. | ![GitHub Badge](https://img.shields.io/github/stars/evidentlyai/evidently.svg?style=flat-square) |

---
### Phase 2: Model Fine-Tuning & Optimization
**Goal**: To specialize a general-purpose large model for a specific domain or task by training it on a targeted dataset, achieving superior performance.

| Project | Details | Repository |
| :--- | :--- | :--- |
| [peft](https://github.com/huggingface/peft) | State-of-the-art Parameter-Efficient Fine-Tuning. | ![GitHub Badge](https://img.shields.io/github/stars/huggingface/peft.svg?style=flat-square) |
| [QLoRA](https://github.com/artidoro/qlora) | Finetune a 65B parameter model on a single 48GB GPU while preserving full 16-bit finetuning task performance. | ![GitHub Badge](https://img.shields.io/github/stars/artidoro/qlora.svg?style=flat-square) |
| [TRL](https://github.com/huggingface/trl) | Train transformer language models with reinforcement learning. | ![GitHub Badge](https://img.shields.io/github/stars/huggingface/trl.svg?style=flat-square) |
| [axolotl](https://github.com/OpenAccess-AI-Collective/axolotl) | A tool designed to streamline the fine-tuning of various AI models. | ![GitHub Badge](https://img.shields.io/github/stars/OpenAccess-AI-Collective/axolotl.svg?style=flat-square) |
| [FastEdit](https://github.com/hiyouga/FastEdit) | FastEdit aims to assist developers with injecting fresh and customized knowledge into large language models efficiently. | ![GitHub Badge](https://img.shields.io/github/stars/hiyouga/FastEdit.svg?style=flat-square) |

---
### Phase 3: Deployment & Serving
**Goal**: To deploy developed and optimized model applications into a production environment efficiently, reliably, and scalably.

#### 3.1 High-Performance Inference & Serving

| Project | Details | Repository |
| :--- | :--- | :--- |
| [vllm](https://github.com/vllm-project/vllm) | A high-throughput and memory-efficient inference and serving engine for LLMs. | ![GitHub Badge](https://img.shields.io/github/stars/vllm-project/vllm.svg?style=flat-square) |
| [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | Inference engine for TensorRT on Nvidia GPUs | ![GitHub Badge](https://img.shields.io/github/stars/NVIDIA/TensorRT-LLM.svg?style=flat-square) |
| [Ollama](https://github.com/jmorganca/ollama) | Serve Llama 2 and other large language models locally from command line or through a browser interface. | ![GitHub Badge](https://img.shields.io/github/stars/jmorganca/ollama.svg?style=flat-square) |

#### 3.2 Model Deployment & Packaging

| Project | Details | Repository |
| :--- | :--- | :--- |
| [TrueFoundry-Py](https://github.com/truefoundry/truefoundry-py) | A PaaS to deploy, Fine-tune and serve LLM Models on your own Infrastructure with Data Security and Optimal GPU Management. | ![GitHub Badge](https://img.shields.io/github/stars/truefoundry/truefoundry-py.svg?style=flat-square) |
| [BentoML](https://github.com/bentoml/BentoML) | The Unified Model Serving Framework | ![GitHub Badge](https://img.shields.io/github/stars/bentoml/BentoML.svg?style=flat-square) |
| [OpenLLM](https://github.com/bentoml/OpenLLM) | An open platform for operating large language models (LLMs) in production. | ![GitHub Badge](https://img.shields.io/github/stars/bentoml/OpenLLM.svg?style=flat-square) |
| [Kserve](https://github.com/kserve/kserve) | Standardized Serverless ML Inference Platform on Kubernetes | ![GitHub Badge](https://img.shields.io/github/stars/kserve/kserve.svg?style=flat-square) |
| [Triton Server](https://github.com/triton-inference-server/server) | The Triton Inference Server provides an optimized cloud and edge inferencing solution. | ![GitHub Badge](https://img.shields.io/github/stars/triton-inference-server/server.svg?style=flat-square) |
| [Kubeflow](https://github.com/kubeflow/kubeflow) | Machine Learning Toolkit for Kubernetes, often used for orchestrating deployment pipelines. | ![GitHub Badge](https://img.shields.io/github/stars/kubeflow/kubeflow.svg?style=flat-square) |

---
### Phase 4: In-Production Operations
**Goal**: To ensure the stability, security, and quality of online services through continuous monitoring and iteration.

#### 4.1 Observability, Monitoring & Cost Management

| Project | Details | Repository |
| :--- | :--- | :--- |
| [Helicone](https://github.com/Helicone/helicone) | Open source LLM observability platform for logging, monitoring, and debugging. | ![GitHub Badge](https://img.shields.io/github/stars/Helicone/helicone.svg?style=flat-square) |
| [Portkey-SDK](https://github.com/Portkey-AI/portkey-python-sdk) | Control Panel with an observability suite & an AI gateway — to ship fast, reliable, and cost-efficient apps. | ![GitHub Badge](https://img.shields.io/github/stars/Portkey-AI/portkey-python-sdk.svg?style=flat-square) |
| [Langfuse](https://github.com/langfuse/langfuse) | Open Source LLM Engineering Platform: Traces, evals, prompt management and metrics to debug and improve your LLM application. | ![GitHub Badge](https://img.shields.io/github/stars/langfuse/langfuse.svg?style=flat-square) |

#### 4.2 Security & Guardrails

| Project | Details | Repository |
| :--- | :--- | :--- |
| [Plexiglass](https://github.com/kortex-labs/plexiglass) | A Python Machine Learning Pentesting Toolbox for Adversarial Attacks. Works with LLMs. | ![GitHub Badge](https://img.shields.io/github/stars/kortex-labs/plexiglass?style=flat-square) |
| [Giskard](https://github.com/Giskard-AI/giskard) | Testing framework dedicated to ML models, from tabular to LLMs. Detect risks of biases, performance issues and errors. | ![GitHub Badge](https://img.shields.io/github/stars/Giskard-AI/giskard.svg?style=flat-square) |
| [Deepchecks](https://github.com/deepchecks/deepchecks) | Tests for Continuous Validation of ML Models & Data. | ![GitHub Badge](https://img.shields.io/github/stars/deepchecks/deepchecks.svg?style=flat-square) |
