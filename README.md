# Prompt Injection Detection Engine (PIDE)

<div align="center">

[![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-success.svg)](https://fastapi.tiangolo.com/)
[![Gradio UI](https://img.shields.io/badge/UI-Gradio-orange.svg)](https://gradio.app/)

*An enterprise-grade, multi-layered security gateway designed to intercept, analyze, and mitigate prompt-injection attacks against Large Language Models (LLMs).*

</div>

---

## Overview

**Prompt Injection Detection Engine (PIDE)** is a high-performance security layer built to safeguard LLM applications from malicious prompt manipulation. Moving beyond single-point validation, PIDE executes a robust, multi-tier pipeline incorporating a lightning-fast rule-based Layer 1 regex engine, embedding similarity checks, heuristic analysis, and dynamic risk scoring.

---

## Core Architecture & Detection Layers

| Layer / Module | Mechanism & Logic | Objective |
| :--- | :--- | :--- |
| **Layer 1** | Rule-Based Regex Engine | Instant pattern matching for known injection signatures ($O(N)$). |
| **Layer 2** | Embedding Similarity & FAISS | Vector-space semantic distance analysis against known malicious exemplars. |
| **Layer 3** | Heuristic & Risk Scoring | Behavioral weighting and dynamic multi-variable scoring thresholds. |
| **API Gateway** | FastAPI REST Gateway | Low-latency asynchronous request handling and secure wrapper integration. |
| **Playground UI** | Gradio Interactive Interface | Real-time monitoring, live auditing, and payload testing dashboard. |

---

## Application Interface & System Preview

### 1. PIDE Main Interface
<div align="center">
  <img src="PIDE MAIN INTERFACE.png" alt="PIDE Main Interface" width="85%"/>
</div>

### 2. Interactive Playground & Detection Engine
<div align="center">
  <img src="pide interface2.png" alt="PIDE Interface Playground" width="85%"/>
</div>

### 3. Execution Logs & Detailed Threat Output
<div align="center">
  <img src="output details.png" alt="Output Details and Threat Analysis" width="85%"/>
</div>

### 4. Live Pipeline Simulation Sample
<div align="center">
  <img src="example1.png" alt="Example Simulation View" width="85%"/>
</div>

---

## Quick Start (No Extra Files Needed)

### 1. Clone the Repository
```bash
git clone https://github.com/AyeshaaRafaqat/Prompt-Injection-Detection-Engine.git
cd Prompt-Injection-Detection-Engine
