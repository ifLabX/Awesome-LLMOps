# A Curated Guide to **Awesome LLMOps** Projects (2025 Edition)

> **Maintenance policy**  
> This list is reviewed **quarterly** (January – April – July – October).  
> Projects must be (1) actively maintained within the last 6 months, (2) released under a permissive open-source license, and (3) either have ≥ 300 GitHub stars **or** demonstrable industry adoption.  
> Items that drop below these thresholds move to a watch-list and may be removed in the next cycle.

---

## Introduction to LLMOps

LLMOps (Large Language Model Operations) is a specialized discipline of MLOps tailored to the unique challenges of managing the entire lifecycle of LLM-powered applications. As organizations move from experimenting with LLMs to deploying them in production, they face distinct hurdles that traditional MLOps practices do not fully address. These challenges include complex prompt engineering, continuous fine-tuning, managing Retrieval-Augmented Generation (RAG) pipelines, handling high computational costs for inference, and monitoring for specific failure modes like hallucinations, toxicity, and data-privacy leakage.

LLMOps provides the principles, practices, and tools necessary to build, deploy, and maintain these applications in a reliable, scalable, and efficient manner. This guide organizes a curated list of high-relevance, open-source tools according to the core stages of the LLMOps lifecycle, providing a top-down workflow from initial concept to production monitoring.

---

## Table of Contents

- [Phase 1 – Development & Experimentation](#phase-1--development--experimentation)  
  - [1.1 Data Versioning & Governance](#11-data-versioning--governance)  
  - [1.2 Vector Stores & RAG Tooling](#12-vector-stores--rag-tooling)  
  - [1.3 Document Processing & Data Cleaning](#13-document-processing--data-cleaning)  
  - [1.4 Prompt Engineering & Optimization](#14-prompt-engineering--optimization)  
  - [1.5 Experiment Tracking](#15-experiment-tracking)  
  - [1.6 LLM Evaluation](#16-llm-evaluation)  
  - [1.7 Agent / App Frameworks](#17-agent--app-frameworks)  
  - [1.8 Pipeline Orchestration](#18-pipeline-orchestration)  
  - [1.9 Text-to-SQL & Database Agents](#19-text-to-sql--database-agents)  
- [Phase 2 – Model Adaptation](#phase-2--model-adaptation)  
  - [2.1 PEFT & LoRA](#21-peft--lora)  
  - [2.2 Model Editing](#22-model-editing)  
- [Phase 3 – Deployment & Serving](#phase-3--deployment--serving)  
  - [3.1 High-Performance Inference & Serving](#31-high-performance-inference--serving)  
  - [3.2 Model Deployment & Packaging](#32-model-deployment--packaging)  
  - [3.3 Edge / Local Runtime](#33-edge--local-runtime)  
- [Phase 4 – Operations](#phase-4--operations)  
  - [4.1 Observability & Cost Management](#41-observability--cost-management)  
  - [4.2 Security & Guardrails](#42-security--guardrails)  
- [Phase 5 – Privacy / Governance / Compliance](#phase-5--privacy--governance--compliance)

---

## Phase 1 – Development & Experimentation  <a id="phase-1--development--experimentation"></a>

> **Goal**: Rapidly iterate on ideas, data, and prompts to prove technical feasibility.  
> **Description**: These tools help collect, clean, version, and explore data; craft and test prompts; prototype agents; and keep experiments reproducible.

### 1.1 Data Versioning & Governance  <a id="11-data-versioning--governance"></a>

> **Goal**: Make datasets reproducible and auditable across the project’s lifetime.  
> **Description**: Git-style version control and labeling frameworks ensure data integrity and provenance.

| Project | Details | Repository |
| :--- | :--- | :--- |
| [DVC](https://github.com/iterative/dvc) | Data Version Control – Git for Data & Models – ML Experiments Management. | ![GitHub Badge](https://img.shields.io/github/stars/iterative/dvc.svg?style=flat-square) |
| [deeplake](https://github.com/activeloopai/deeplake) | Data Lake for Deep Learning. Build, manage, query, version, & visualize datasets. Stream data in real-time to PyTorch/TensorFlow. | ![GitHub Badge](https://img.shields.io/github/stars/activeloopai/Hub.svg?style=flat-square) |
| [LakeFS](https://github.com/treeverse/lakeFS) | Git-like capabilities for your object storage. | ![GitHub Badge](https://img.shields.io/github/stars/treeverse/lakeFS.svg?style=flat-square) |
| [Cleanlab](https://github.com/cleanlab/cleanlab) | The standard data-centric AI package for data quality and machine learning with messy, real-world data and labels. | ![GitHub Badge](https://img.shields.io/github/stars/cleanlab/cleanlab.svg?style=flat-square) |
| [Label Studio](https://github.com/HumanSignal/label-studio) | A multi-type data labeling and annotation tool with a standardized output format. Essential for creating high-quality datasets. | ![GitHub Badge](https://img.shields.io/github/stars/HumanSignal/label-studio.svg?style=flat-square) |

### 1.2 Vector Stores & RAG Tooling  <a id="12-vector-stores--rag-tooling"></a>

> **Goal**: Store and retrieve embeddings efficiently for Retrieval-Augmented Generation.  
> **Description**: RAG platforms and vector databases manage unstructured knowledge and power hybrid search.

| Project | Details | Repository |
| :--- | :--- | :--- |
| [RagFlow](https://github.com/infiniflow/ragflow) | An open-source RAG application that provides a streamlined workflow based on deep document understanding. | ![GitHub Badge](https://img.shields.io/github/stars/infiniflow/ragflow.svg?style=flat-square) |
| [FastGPT](https://github.com/labring/FastGPT) | A platform that based on LLM, allows you to create your own knowledge-base QA model with out-of-the-box capabilities. | ![GitHub Badge](https://img.shields.io/github/stars/labring/FastGPT.svg?style=flat-square) |

### 1.3 Document Processing & Data Cleaning  <a id="13-document-processing--data-cleaning"></a>

> **Goal**: Convert raw files and web sources into high-quality, LLM-ready text.  
> **Description**: ETL, parsing, and adversarial augmentation frameworks enhance data variety and robustness.

| Project | Details | Repository |
| :--- | :--- | :--- |
| [Data-Juicer](https://github.com/modelscope/data-juicer) | A one-stop data processing system for LLMs. Used to build diverse, high-quality data recipes for pre-training and fine-tuning. | ![GitHub Badge](https://img.shields.io/github/stars/modelscope/data-juicer.svg?style=flat-square) |
| [Firecrawl](https://github.com/mendableai/firecrawl) | An API service that crawls any URL and converts it into clean, LLM-ready Markdown or structured data. | ![GitHub Badge](https://img.shields.io/github/stars/mendableai/firecrawl.svg?style=flat-square) |
| [OneFileLLM](https://github.com/jimmc414/onefilellm) | A CLI tool to aggregate and preprocess data from multiple sources (files, GitHub, web) into a single text file for LLM use. | ![GitHub Badge](https://img.shields.io/github/stars/jimmc414/onefilellm.svg?style=flat-square) |
| [Apache Tika](https://github.com/apache/tika) | A content detection and analysis framework that extracts text and metadata from a huge variety of file formats. | ![GitHub Badge](https://img.shields.io/github/stars/apache/tika.svg?style=flat-square) |
| [Unstructured](https://github.com/Unstructured-IO/unstructured) | Open-source libraries and APIs to build custom data transformation pipelines for ETL, LLMs, and data analysis. | ![GitHub Badge](https://img.shields.io/github/stars/Unstructured-IO/unstructured.svg?style=flat-square) |
| [DeepKE](https://github.com/zjunlp/DeepKE) | A deep learning based knowledge extraction toolkit, supporting named entity, relation, and attribute extraction. | ![GitHub Badge](https://img.shields.io/github/stars/zjunlp/DeepKE.svg?style=flat-square) |
| [Lilac](https://github.com/databricks/lilac) | An open-source tool that helps you see and understand your unstructured text data. Explore, cluster, clean, and enrich datasets for LLMs. | ![GitHub Badge](https://img.shields.io/github/stars/databricks/lilac.svg?style=flat-square) |
| [TextAttack](https://github.com/QData/TextAttack) | A Python framework for adversarial attacks, data augmentation, and hard-negative generation to improve robustness. | ![GitHub Badge](https://img.shields.io/github/stars/QData/TextAttack.svg?style=flat-square) |

### 1.4 Prompt Engineering & Optimization  <a id="14-prompt-engineering--optimization"></a>

> **Goal**: Design, test, and version prompts for consistent, high-quality outputs.  
> **Description**: These tools provide A/B testing, genetic search, and interactive sandboxes for rapid iteration.

| Project | Details | Repository |
| :--- | :--- | :--- |
| [promptfoo](https://github.com/typpo/promptfoo) | Open-source tool for testing & evaluating prompt quality. | ![GitHub Badge](https://img.shields.io/github/stars/typpo/promptfoo.svg?style=flat-square) |
| [Agenta](https://github.com/agenta-ai/agenta) | An open-source LLMOps platform with tools for prompt management, evaluation, and deployment. | ![GitHub Badge](https://img.shields.io/github/stars/agenta-ai/agenta.svg?style=flat-square) |
| [DSPy](https://github.com/stanfordnlp/dspy) | A framework for programming—not just prompting—language models. It allows you to optimize prompts and weights. | ![GitHub Badge](https://img.shields.io/github/stars/stanfordnlp/dspy.svg?style=flat-square) |
| [Chainlit](https://github.com/Chainlit/chainlit) | Build and share conversational UIs in seconds; perfect for interactive prompt sandboxing and demos. | ![GitHub Badge](https://img.shields.io/github/stars/Chainlit/chainlit.svg?style=flat-square) |

### 1.5 Experiment Tracking  <a id="15-experiment-tracking"></a>

> **Goal**: Record, compare, and reproduce experiments across data, prompts, and models.  
> **Description**: Track metrics, parameters, and artifacts; integrate with CI to enable data-driven decisions.

| Project | Details | Repository |
| :--- | :--- | :--- |
| [MLflow](https://github.com/mlflow/mlflow) | An open-source framework for the end-to-end machine learning lifecycle, helping developers track experiments, evaluate models/prompts, and more. | ![GitHub Badge](https://img.shields.io/github/stars/mlflow/mlflow.svg?style=flat-square) |
| [Weights & Biases](https://github.com/wandb/wandb) | A developer-first MLOps platform for experiment tracking, dataset versioning, and model management. Featuring W&B Prompts for LLM execution flow visualization. | ![GitHub Badge](https://img.shields.io/github/stars/wandb/wandb.svg?style=flat-square) |
| [Aim](https://github.com/aimhubio/aim) | An easy-to-use and performant open-source experiment tracker. | ![GitHub Badge](https://img.shields.io/github/stars/aimhubio/aim.svg?style=flat-square) |

### 1.6 LLM Evaluation  <a id="16-llm-evaluation"></a>

> **Goal**: Quantify performance, robustness, and safety of prompts and models.  
> **Description**: Local and cloud frameworks automate scoring for RAG, summarization, Q&A, and more.

| Project | Details | Repository |
| :--- | :--- | :--- |
| [LangWatch](https://github.com/langwatch/langwatch) | Visualize LLM evaluations experiments and DSPy pipeline optimizations. | ![GitHub Badge](https://img.shields.io/github/stars/langwatch/langwatch.svg?style=flat-square) |
| [Arize-Phoenix](https://github.com/Arize-ai/phoenix) | ML observability for LLMs, vision, language, and tabular models. Also offers powerful local evaluation capabilities. | ![GitHub Badge](https://img.shields.io/github/stars/Arize-ai/phoenix.svg?style=flat-square) |
| [Evidently](https://github.com/evidentlyai/evidently) | An open-source framework to evaluate, test and monitor ML and LLM-powered systems. | ![GitHub Badge](https://img.shields.io/github/stars/evidentlyai/evidently.svg?style=flat-square) |
| [Ragas](https://github.com/explodinggradients/ragas) | RAG evaluation metrics and pipelines for faithfulness and answer relevancy. | ![GitHub Badge](https://img.shields.io/github/stars/explodinggradients/ragas.svg?style=flat-square) |
| [OpenAI Evals](https://github.com/openai/evals) | Reference harness for benchmarking GPT-style models across tasks. | ![GitHub Badge](https://img.shields.io/github/stars/openai/evals.svg?style=flat-square) |

### 1.7 Agent / App Frameworks  <a id="17-agent--app-frameworks"></a>

> **Goal**: Compose prompts, tools, and workflows into full-stack LLM applications.  
> **Description**: High-level SDKs and low-code builders accelerate agent development and experimentation.

| Project | Details | Repository |
| :--- | :--- | :--- |
| [LangChain](https://github.com/hwchase17/langchain) | Building applications with LLMs through composability. | ![GitHub Badge](https://img.shields.io/github/stars/hwchase17/langchain.svg?style=flat-square) |
| [LlamaIndex](https://github.com/jerryjliu/llama_index) | Provides a central interface to connect your LLMs with external data. | ![GitHub Badge](https://img.shields.io/github/stars/jerryjliu/llama_index.svg?style=flat-square) |
| [Dify](https://github.com/langgenius/dify) | An open-source LLM app development platform for building and operating generative AI-native applications. | ![GitHub Badge](https://img.shields.io/github/stars/langgenius/dify.svg?style=flat-square) |
| [Flowise](https://github.com/FlowiseAI/Flowise) | Drag & drop UI to build your customized LLM flow using LangchainJS. | ![GitHub Badge](https://img.shields.io/github/stars/FlowiseAI/Flowise.svg?style=flat-square) |

### 1.8 Pipeline Orchestration  <a id="18-pipeline-orchestration"></a>

> **Goal**: Automate batch and streaming workflows for data ingestion, fine-tuning, and evaluation.  
> **Description**: DAG-based schedulers and function-graph frameworks ensure reproducible, modular pipelines.

| Project | Details | Repository |
| :--- | :--- | :--- |
| [Apache Airflow](https://github.com/apache/airflow) | A platform to programmatically author, schedule, and monitor workflows. Ideal for orchestrating batch jobs like fine-tuning or RAG indexing. | ![GitHub Badge](https://img.shields.io/github/stars/apache/airflow.svg?style=flat-square) |
| [Apache NiFi](https://github.com/apache/nifi) | An easy-to-use, powerful, and reliable system to process and distribute data. Well-suited for real-time, streaming data pipelines for RAG. | ![GitHub Badge](https://img.shields.io/github/stars/apache/nifi.svg?style=flat-square) |
| [ZenML](https://github.com/zenml-io/zenml) | MLOps framework to create reproducible pipelines for ML and LLM workflows. | ![GitHub Badge](https://img.shields.io/github/stars/zenml-io/zenml.svg?style=flat-square) |
| [Hamilton](https://github.com/dagworks-inc/hamilton) | A lightweight framework to represent ML/language model pipelines as a series of Python functions. | ![GitHub Badge](https://img.shields.io/github/stars/dagworks-inc/hamilton.svg?style=flat-square) |

### 1.9 Text-to-SQL & Database Agents  <a id="19-text-to-sql--database-agents"></a>

> **Goal**: Translate natural-language queries to SQL and unlock structured data for business users.  
> **Description**: These tools combine LLMs with schema discovery and query execution to generate accurate, safe SQL across diverse databases.

| Project | Details | Repository |
| :--- | :--- | :--- |
| [Chat2DB](https://github.com/chat2db/Chat2DB) | AI-augmented SQL client: natural-language to SQL, visualization, and reporting. | ![GitHub Badge](https://img.shields.io/github/stars/chat2db/Chat2DB.svg?style=flat-square) |
| [Vanna.ai](https://github.com/vanna-ai/vanna) | Python-based framework for schema-aware text-to-SQL and RAG-enhanced analytics. | ![GitHub Badge](https://img.shields.io/github/stars/vanna-ai/vanna.svg?style=flat-square) |
| [DB-GPT](https://github.com/eosphoros-ai/DB-GPT) | Private, self-hosted text-to-SQL agent framework with RAG support. | ![GitHub Badge](https://img.shields.io/github/stars/eosphoros-ai/DB-GPT.svg?style=flat-square) |

---

## Phase 2 – Model Adaptation  <a id="phase-2--model-adaptation"></a>

> **Goal**: Specialize general-purpose LLMs to domain-specific tasks while controlling compute and data cost.  
> **Description**: Parameter-efficient fine-tuning and editing techniques inject new knowledge and correct errors without full retraining.

### 2.1 PEFT & LoRA  <a id="21-peft--lora"></a>

| Project | Details | Repository |
| :--- | :--- | :--- |
| [LlamaFactory](https://github.com/hiyouga/LLaMA-Factory) | A unified, efficient fine-tuning framework for over 100 LLMs and VLMs. | ![GitHub Badge](https://img.shields.io/github/stars/hiyouga/LLaMA-Factory.svg?style=flat-square) |
| [Swift (modelscope)](https://github.com/modelscope/swift) | A framework for fine-tuning and deploying 500+ LLMs and 200+ MLLMs, with extensive support for PEFT techniques. | ![GitHub Badge](https://img.shields.io/github/stars/modelscope/swift.svg?style=flat-square) |
| [peft](https://github.com/huggingface/peft) | State-of-the-art Parameter-Efficient Fine-Tuning. | ![GitHub Badge](https://img.shields.io/github/stars/huggingface/peft.svg?style=flat-square) |
| [QLoRA](https://github.com/artidoro/qlora) | Finetune a 65 B parameter model on a single 48 GB GPU while preserving full 16-bit finetuning task performance. | ![GitHub Badge](https://img.shields.io/github/stars/artidoro/qlora.svg?style=flat-square) |
| [axolotl](https://github.com/OpenAccess-AI-Collective/axolotl) | A tool designed to streamline the fine-tuning of various AI models. | ![GitHub Badge](https://img.shields.io/github/stars/OpenAccess-AI-Collective/axolotl.svg?style=flat-square) |
| [LoRA-Hub](https://github.com/sail-sg/lorahub) | Community marketplace and registry for sharing and discovering LoRA weight adapters. | ![GitHub Badge](https://img.shields.io/github/stars/sail-sg/lorahub.svg?style=flat-square) |

### 2.2 Model Editing  <a id="22-model-editing"></a>

| Project | Details | Repository |
| :--- | :--- | :--- |
| [FastEdit](https://github.com/hiyouga/FastEdit) | FastEdit aims to assist developers with injecting fresh and customized knowledge into large language models efficiently. | ![GitHub Badge](https://img.shields.io/github/stars/hiyouga/FastEdit.svg?style=flat-square) |

---

## Phase 3 – Deployment & Serving  <a id="phase-3--deployment--serving"></a>

> **Goal**: Deliver low-latency, scalable inference to end users across cloud and edge environments.  
> **Description**: Engines, packaging frameworks, and local runtimes optimize throughput, cost, and portability.

### 3.1 High-Performance Inference & Serving  <a id="31-high-performance-inference--serving"></a>

| Project | Details | Repository |
| :--- | :--- | :--- |
| [vllm](https://github.com/vllm-project/vllm) | A high-throughput and memory-efficient inference and serving engine for LLMs. | ![GitHub Badge](https://img.shields.io/github/stars/vllm-project/vllm.svg?style=flat-square) |
| [SGLang](https://github.com/sgl-project/sglang) | A fast serving framework for LLMs and VLMs, designed for high throughput and controllable, structured generation. | ![GitHub Badge](https://img.shields.io/github/stars/sgl-project/sglang.svg?style=flat-square) |
| [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | Inference engine for TensorRT on Nvidia GPUs. | ![GitHub Badge](https://img.shields.io/github/stars/NVIDIA/TensorRT-LLM.svg?style=flat-square) |
| [Ollama](https://github.com/jmorganca/ollama) | Serve LLMs locally. A user-friendly application often powered by llama.cpp underneath. | ![GitHub Badge](https://img.shields.io/github/stars/jmorganca/ollama.svg?style=flat-square) |
| [llama.cpp](https://github.com/ggerganov/llama.cpp) | A foundational library for LLM inference in pure C/C++, enabling efficient performance on CPUs and consumer hardware. | ![GitHub Badge](https://img.shields.io/github/stars/ggerganov/llama.cpp.svg?style=flat-square) |

### 3.2 Model Deployment & Packaging  <a id="32-model-deployment--packaging"></a>

| Project | Details | Repository |
| :--- | :--- | :--- |
| [Xinference](https://github.com/xorbitsai/inference) | A versatile platform to serve language, speech, and multimodal models with a unified, OpenAI-compatible API. | ![GitHub Badge](https://img.shields.io/github/stars/xorbitsai/inference.svg?style=flat-square) |
| [BentoML](https://github.com/bentoml/BentoML) | The Unified Model Serving Framework. | ![GitHub Badge](https://img.shields.io/github/stars/bentoml/BentoML.svg?style=flat-square) |
| [OpenLLM](https://github.com/bentoml/OpenLLM) | An open platform for operating large language models (LLMs) in production. | ![GitHub Badge](https://img.shields.io/github/stars/bentoml/OpenLLM.svg?style=flat-square) |
| [Kserve](https://github.com/kserve/kserve) | Standardized Serverless ML Inference Platform on Kubernetes. | ![GitHub Badge](https://img.shields.io/github/stars/kserve/kserve.svg?style=flat-square) |
| [Triton Server](https://github.com/triton-inference-server/server) | The Triton Inference Server provides an optimized cloud and edge inferencing solution. | ![GitHub Badge](https://img.shields.io/github/stars/triton-inference-server/server.svg?style=flat-square) |
| [Kubeflow](https://github.com/kubeflow/kubeflow) | Machine Learning Toolkit for Kubernetes, often used for orchestrating deployment pipelines. | ![GitHub Badge](https://img.shields.io/github/stars/kubeflow/kubeflow.svg?style=flat-square) |

### 3.3 Edge / Local Runtime  <a id="33-edge--local-runtime"></a>

| Project | Details | Repository |
| :--- | :--- | :--- |
| [llama.cpp](https://github.com/ggerganov/llama.cpp) | A foundational library for LLM inference in pure C/C++, enabling efficient performance on CPUs and consumer hardware. | ![GitHub Badge](https://img.shields.io/github/stars/ggerganov/llama.cpp.svg?style=flat-square) |
| [Ollama](https://github.com/jmorganca/ollama) | Serve LLMs locally. A user-friendly application often powered by llama.cpp underneath. | ![GitHub Badge](https://img.shields.io/github/stars/jmorganca/ollama.svg?style=flat-square) |

---

## Phase 4 – Operations  <a id="phase-4--operations"></a>

> **Goal**: Maintain reliability, cost efficiency, and user safety for live systems.  
> **Description**: Observability, guardrails, and policy frameworks provide continuous feedback and protection.

### 4.1 Observability & Cost Management  <a id="41-observability--cost-management"></a>

| Project | Details | Repository |
| :--- | :--- | :--- |
| [Helicone](https://github.com/Helicone/helicone) | Open source LLM observability platform for logging, monitoring, and debugging. | ![GitHub Badge](https://img.shields.io/github/stars/Helicone/helicone.svg?style=flat-square) |
| [Portkey-SDK](https://github.com/Portkey-AI/portkey-python-sdk) | Control Panel with an observability suite & an AI gateway — to ship fast, reliable, and cost-efficient apps. | ![GitHub Badge](https://img.shields.io/github/stars/Portkey-AI/portkey-python-sdk.svg?style=flat-square) |
| [Langfuse](https://github.com/langfuse/langfuse) | Open Source LLM Engineering Platform: Traces, evals, prompt management and metrics to debug and improve your LLM application. | ![GitHub Badge](https://img.shields.io/github/stars/langfuse/langfuse.svg?style=flat-square) |

### 4.2 Security & Guardrails  <a id="42-security--guardrails"></a>

| Project | Details | Repository |
| :--- | :--- | :--- |
| [Guardrails-AI](https://github.com/guardrails-ai/guardrails) | Declarative, schema-driven validation and content moderation for LLM outputs. | ![GitHub Badge](https://img.shields.io/github/stars/guardrails-ai/guardrails.svg?style=flat-square) |

---

## Phase 5 – Privacy / Governance / Compliance  <a id="phase-5--privacy--governance--compliance"></a>

> **Goal**: Ensure AI systems meet legal, ethical, and organizational standards.  
> **Description**: Policy-as-code, bias detection, and continuous validation frameworks enable trustworthy deployment.

| Project | Details | Repository |
| :--- | :--- | :--- |
| [Giskard](https://github.com/Giskard-AI/giskard) | Testing framework dedicated to ML models, from tabular to LLMs. Detect risks of biases, performance issues and errors. | ![GitHub Badge](https://img.shields.io/github/stars/Giskard-AI/giskard.svg?style=flat-square) |
| [Deepchecks](https://github.com/deepchecks/deepchecks) | Tests for Continuous Validation of ML Models & Data. | ![GitHub Badge](https://img.shields.io/github/stars/deepchecks/deepchecks.svg?style=flat-square) |

---
