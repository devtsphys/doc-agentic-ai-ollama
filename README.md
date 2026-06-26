# Agentic AI & Open Source Models — Comprehensive Reference Card
> Focus: Ollama Desktop · llama3.2 · Open-Weight LLMs · MCP · Agents · Skills · Design Patterns

---

## Table of Contents

1. [Core Concepts & Mental Models](#1-core-concepts--mental-models)
2. [Ollama Desktop — Setup & Model Management](#2-ollama-desktop--setup--model-management)
3. [Model Taxonomy & Architecture](#3-model-taxonomy--architecture)
4. [Agentic AI Architecture](#4-agentic-ai-architecture)
5. [LLM Interaction Patterns](#5-llm-interaction-patterns)
6. [Prompt Engineering for Agents](#6-prompt-engineering-for-agents)
7. [Tool Use & Function Calling](#7-tool-use--function-calling)
8. [Model Context Protocol (MCP)](#8-model-context-protocol-mcp)
9. [Agent Skills & Capabilities](#9-agent-skills--capabilities)
10. [Multi-Agent Systems](#10-multi-agent-systems)
11. [Memory Systems](#11-memory-systems)
12. [RAG — Retrieval-Augmented Generation](#12-rag--retrieval-augmented-generation)
13. [Agentic Workflows & Pipelines](#13-agentic-workflows--pipelines)
14. [Design Patterns Catalog](#14-design-patterns-catalog)
15. [Project Structure Templates](#15-project-structure-templates)
16. [Implementation Code Library](#16-implementation-code-library)
17. [Evaluation & Observability](#17-evaluation--observability)
18. [Best Practices](#18-best-practices)
19. [Method Reference Table](#19-method-reference-table)
20. [Quick-Reference Cheat Sheet](#20-quick-reference-cheat-sheet)

---

## 1. Core Concepts & Mental Models

### 1.1 What Is Agentic AI?

An **AI agent** is an LLM-powered system that can:
- **Perceive** an environment (via context, tool results, memory)
- **Reason** about a goal across multiple steps
- **Act** by calling tools, writing code, or triggering workflows
- **Reflect** on outcomes and self-correct

```
┌─────────────────────────────────────────────────────────────┐
│                       AGENT LOOP                            │
│                                                             │
│  Observe → Think → Plan → Act → Observe → Think → ...      │
│     ↑                              │                        │
│     └──────────── Feedback ────────┘                        │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 The Spectrum of Autonomy

| Level | Name | Description | Example |
|---|---|---|---|
| 0 | **Chatbot** | Single-turn Q&A | `ollama run llama3.2` |
| 1 | **Prompted Chain** | Fixed multi-step pipeline | Summarize → Translate |
| 2 | **Tool-Augmented** | LLM selects tools from set | Web search + calculator |
| 3 | **ReAct Agent** | Reason + Act loop, dynamic | Coding agent, research |
| 4 | **Multi-Agent** | Agents collaborate/delegate | Orchestrator + workers |
| 5 | **Self-Improving** | Agents modify their own prompts/skills | AutoGPT-style |

### 1.3 Key Terminology

| Term | Definition |
|---|---|
| **LLM** | Large Language Model — the core reasoning engine |
| **Agent** | LLM + tools + memory + loop controller |
| **Skill** | A composable, reusable capability unit (tool + prompt + schema) |
| **MCP** | Model Context Protocol — standard for tool/resource exposure |
| **RAG** | Retrieval-Augmented Generation — ground LLM with external docs |
| **Context Window** | Max tokens the model processes at once |
| **Temperature** | Sampling randomness (0 = deterministic, 1+ = creative) |
| **Tool Call** | Structured output requesting external function execution |
| **Grounding** | Anchoring responses to verifiable external knowledge |
| **Hallucination** | Confident but factually wrong generation |
| **ReAct** | Reasoning + Acting — interleaved thought and action trace |
| **Chain-of-Thought** | Step-by-step reasoning in the prompt/response |
| **Orchestrator** | Agent that plans and delegates to sub-agents |
| **Scratchpad** | Temporary reasoning space within context |

---

## 2. Ollama Desktop — Setup & Model Management

### 2.1 Installation

```bash
# macOS
brew install ollama

# Linux (one-liner)
curl -fsSL https://ollama.com/install.sh | sh

# Windows
# Download installer from https://ollama.com/download
# OR via winget:
winget install Ollama.Ollama

# Start the server (macOS/Linux)
ollama serve          # runs on http://localhost:11434
```

### 2.2 Core CLI Commands

```bash
# Pull / download models
ollama pull llama3.2           # 3B default
ollama pull llama3.2:1b        # 1B variant
ollama pull llama3.2:3b-instruct-q8_0   # specific quant
ollama pull llama3.1:8b        # 8B model
ollama pull llama3.1:70b       # 70B model (needs ~40GB RAM)
ollama pull mistral            # Mistral 7B
ollama pull mistral-nemo       # Mistral Nemo 12B
ollama pull qwen2.5:7b         # Qwen2.5 7B
ollama pull deepseek-r1:7b     # DeepSeek R1 7B (reasoning)
ollama pull phi4               # Microsoft Phi-4
ollama pull gemma3:4b          # Google Gemma3 4B
ollama pull nomic-embed-text   # Embedding model

# Run interactively
ollama run llama3.2
ollama run llama3.2 "Explain transformers in one paragraph"

# List local models
ollama list

# Model info
ollama show llama3.2
ollama show llama3.2 --modelfile   # show Modelfile
ollama show llama3.2 --parameters  # context size, temperature, etc.

# Remove model
ollama rm llama3.2

# Copy/rename
ollama cp llama3.2 my-custom-llama

# Push to registry (Ollama Hub)
ollama push myuser/mymodel

# Pull running processes
ollama ps
```

### 2.3 Modelfile — Custom Models

```dockerfile
# Modelfile — custom system prompt + parameters
FROM llama3.2

# System prompt (agent persona)
SYSTEM """
You are an expert Python data engineer specializing in Apache Spark and
Databricks. You respond concisely, always provide runnable code examples,
and prefer PySpark over pandas for large datasets.
"""

# Model parameters
PARAMETER temperature 0.2
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 8192          # context window size
PARAMETER num_predict 2048      # max output tokens
PARAMETER repeat_penalty 1.1   # penalize repetition
PARAMETER stop "<|eot_id|>"    # stop token

# Template override (llama3.2 chat format)
TEMPLATE """{{ if .System }}<|start_header_id|>system<|end_header_id|>
{{ .System }}<|eot_id|>{{ end }}
{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>
{{ .Prompt }}<|eot_id|>{{ end }}
<|start_header_id|>assistant<|end_header_id|>
{{ .Response }}<|eot_id|>"""
```

```bash
# Build and use custom model
ollama create spark-engineer -f Modelfile
ollama run spark-engineer "Write a PySpark UDF for string normalization"
```

### 2.4 REST API Reference

```bash
# Base URL
http://localhost:11434

# Generate (single-turn)
curl http://localhost:11434/api/generate \
  -d '{"model":"llama3.2","prompt":"Hello","stream":false}'

# Chat (multi-turn)
curl http://localhost:11434/api/chat \
  -d '{
    "model": "llama3.2",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is 2+2?"}
    ],
    "stream": false
  }'

# Embeddings
curl http://localhost:11434/api/embed \
  -d '{"model":"nomic-embed-text","input":"Hello world"}'

# List models
curl http://localhost:11434/api/tags

# Model info
curl http://localhost:11434/api/show \
  -d '{"name":"llama3.2"}'

# Pull model (programmatic)
curl http://localhost:11434/api/pull \
  -d '{"name":"llama3.2","stream":false}'

# Delete model
curl -X DELETE http://localhost:11434/api/delete \
  -d '{"name":"llama3.2"}'
```

### 2.5 Python Client (ollama-python)

```python
pip install ollama
```

```python
import ollama

# Simple generation
response = ollama.generate(model="llama3.2", prompt="Why is the sky blue?")
print(response["response"])

# Chat completion
response = ollama.chat(
    model="llama3.2",
    messages=[
        {"role": "system", "content": "You are a concise technical assistant."},
        {"role": "user", "content": "Explain Docker in 3 bullet points."},
    ],
    options={"temperature": 0.1, "num_ctx": 4096},
)
print(response["message"]["content"])

# Streaming
for chunk in ollama.chat(
    model="llama3.2",
    messages=[{"role": "user", "content": "Write a haiku about Python."}],
    stream=True,
):
    print(chunk["message"]["content"], end="", flush=True)

# Embeddings
embedding = ollama.embed(model="nomic-embed-text", input="Hello world")
print(embedding["embeddings"][0][:5])  # first 5 dims

# Async client
import asyncio
from ollama import AsyncClient

async def main():
    client = AsyncClient()
    response = await client.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": "Hi"}],
    )
    print(response["message"]["content"])

asyncio.run(main())
```

---

## 3. Model Taxonomy & Architecture

### 3.1 LLaMA 3.2 Architecture Deep-Dive

```
LLaMA 3.2 — Meta's Open-Weight Model Family (Sept 2024)

Sizes:          1B  │  3B  │  11B (vision)  │  90B (vision)
Context:        128K tokens (all variants)
Architecture:   Grouped Query Attention (GQA)
                Rotary Positional Embeddings (RoPE)
                SwiGLU activation
                RMSNorm (no bias)
                Tied embeddings (1B, 3B)
Training:       ~9 trillion tokens
Instruction:    Meta's supervised fine-tuning (SFT) + RLHF
License:        LLaMA 3 Community License
```

**Transformer Block (conceptual)**:
```
Input Tokens
     │
  Embedding Layer  (vocab_size → d_model)
     │
  ┌──┴──────────────────────────────────┐
  │  N × Transformer Blocks             │
  │  ┌─────────────────────────────┐    │
  │  │ RMSNorm                     │    │
  │  │ Multi-Head GQA Attention    │    │
  │  │   (Q heads > KV heads)      │    │
  │  │ Residual                    │    │
  │  │ RMSNorm                     │    │
  │  │ SwiGLU Feed-Forward         │    │
  │  │   (gate proj × up proj)     │    │
  │  │ Residual                    │    │
  │  └─────────────────────────────┘    │
  └──────────────────────────────────┬──┘
                                     │
  RMSNorm → Linear (d_model → vocab) → Softmax
     │
  Output Token Probabilities
```

### 3.2 Model Comparison Table

| Model | Params | Context | VRAM (Q4) | Strengths | Best For |
|---|---|---|---|---|---|
| llama3.2:1b | 1B | 128K | ~0.8GB | Fastest, tiny | Edge, classification |
| llama3.2:3b | 3B | 128K | ~2GB | Great balance | Agents, coding |
| llama3.1:8b | 8B | 128K | ~5GB | Strong reasoning | Complex tasks |
| llama3.1:70b | 70B | 128K | ~40GB | Near-GPT4 quality | Research |
| mistral:7b | 7B | 32K | ~4GB | Fast, efficient | General purpose |
| mistral-nemo | 12B | 128K | ~7GB | Multilingual | EU/GDPR use cases |
| qwen2.5:7b | 7B | 128K | ~4.5GB | Coding, math | Data engineering |
| deepseek-r1:7b | 7B | 64K | ~4.5GB | Chain-of-thought | Reasoning tasks |
| phi4 | 14B | 16K | ~8GB | Dense knowledge | STEM, analysis |
| gemma3:4b | 4B | 128K | ~2.5GB | Instruction-follow | Chat, summaries |
| codellama:7b | 7B | 16K | ~4GB | Code generation | Dev assistants |
| nomic-embed-text | — | 8K | ~0.3GB | Embeddings | RAG, similarity |

### 3.3 Quantization Levels

| Quantization | Bits | Size Reduction | Quality Loss | Use Case |
|---|---|---|---|---|
| f16 | 16-bit | 1× (baseline) | None | Training, max quality |
| q8_0 | 8-bit | ~2× | Negligible | High quality inference |
| q6_K | 6-bit | ~2.5× | Very low | Quality/size balance |
| q5_K_M | 5-bit | ~3× | Low | Recommended sweet spot |
| q4_K_M | 4-bit | ~4× | Low-Medium | Default Ollama builds |
| q4_0 | 4-bit | ~4× | Medium | Faster, slightly lower |
| q3_K_M | 3-bit | ~5× | Medium | Resource-constrained |
| q2_K | 2-bit | ~7× | High | Extreme compression |

```bash
# Pull specific quantization
ollama pull llama3.1:8b-instruct-q5_K_M
ollama pull llama3.1:8b-instruct-q8_0
```

### 3.4 Chat Template Formats

```python
# LLaMA 3.x format
LLAMA3_TEMPLATE = """<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
{system}<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{user}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

# Mistral / Mixtral format
MISTRAL_TEMPLATE = """<s>[INST] {system}

{user} [/INST]"""

# Gemma format  
GEMMA_TEMPLATE = """<start_of_turn>user
{user}<end_of_turn>
<start_of_turn>model
"""

# ChatML (OpenAI-compatible, used by many models)
CHATML_TEMPLATE = """<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{user}<|im_end|>
<|im_start|>assistant
"""
```

---

## 4. Agentic AI Architecture

### 4.1 Agent Component Model

```
┌────────────────────────────────────────────────────────────────┐
│                          AGENT                                 │
│                                                                │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────┐   │
│  │   PERCEIVE   │   │    REASON    │   │      ACT         │   │
│  │              │   │              │   │                  │   │
│  │ • Context    │──▶│ • LLM Core   │──▶│ • Tool Executor  │   │
│  │ • Tool Res.  │   │ • CoT/ReAct  │   │ • Code Runner    │   │
│  │ • Memory     │   │ • Planning   │   │ • API Calls      │   │
│  │ • Environment│   │ • Reflection │   │ • File I/O       │   │
│  └──────────────┘   └──────────────┘   └──────────────────┘   │
│                              │                    │            │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                        MEMORY                            │  │
│  │  Working Memory  │  Episodic  │  Semantic  │  Procedural │  │
│  │  (Context Window)│  (History) │  (KB/RAG)  │  (Skills)   │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
```

### 4.2 ReAct Loop Architecture

```python
# ReAct: Reasoning + Acting
# Pattern: Thought → Action → Observation → Thought → ...

REACT_SYSTEM_PROMPT = """You are an intelligent assistant with access to tools.
Use the following format strictly:

Thought: [your reasoning about what to do next]
Action: [tool_name]
Action Input: [JSON input for the tool]
Observation: [result from tool — filled by system]
... (repeat Thought/Action/Observation as needed)
Thought: I now have enough information to answer.
Final Answer: [your complete answer to the user]

Available tools:
{tool_descriptions}
"""

class ReactAgent:
    def __init__(self, model: str, tools: list):
        self.model = model
        self.tools = {t.name: t for t in tools}
        self.max_iterations = 10
    
    def run(self, task: str) -> str:
        messages = [
            {"role": "system", "content": REACT_SYSTEM_PROMPT.format(
                tool_descriptions=self._format_tools()
            )},
            {"role": "user", "content": task},
        ]
        
        for i in range(self.max_iterations):
            response = ollama.chat(model=self.model, messages=messages)
            content = response["message"]["content"]
            messages.append({"role": "assistant", "content": content})
            
            # Parse action from response
            action, action_input = self._parse_action(content)
            
            if action == "FINAL_ANSWER":
                return self._extract_final_answer(content)
            
            # Execute tool
            observation = self._execute_tool(action, action_input)
            messages.append({
                "role": "user",
                "content": f"Observation: {observation}"
            })
        
        return "Max iterations reached."
```

### 4.3 Plan-and-Execute Architecture

```python
# Two-phase: first plan, then execute each step
class PlanExecuteAgent:
    def plan(self, task: str) -> list[str]:
        """Phase 1: Generate ordered step list"""
        response = ollama.chat(
            model="llama3.2",
            messages=[{
                "role": "user",
                "content": f"""Break this task into concrete steps (JSON array of strings):
Task: {task}
Return ONLY a JSON array, no explanation."""
            }],
        )
        return json.loads(response["message"]["content"])
    
    def execute(self, steps: list[str], context: dict) -> list[str]:
        """Phase 2: Execute each step sequentially"""
        results = []
        for step in steps:
            result = self._execute_step(step, context, results)
            results.append(result)
            context["history"] = results
        return results
    
    def run(self, task: str) -> str:
        steps = self.plan(task)
        results = self.execute(steps, {})
        return self._synthesize(task, steps, results)
```

---

## 5. LLM Interaction Patterns

### 5.1 Completion Modes

```python
import ollama
from typing import Generator

# ── 1. Single Generation ──────────────────────────────────────
def generate(prompt: str, model: str = "llama3.2") -> str:
    r = ollama.generate(model=model, prompt=prompt, options={"temperature": 0})
    return r["response"]

# ── 2. Chat Completion ────────────────────────────────────────
def chat(messages: list[dict], model: str = "llama3.2") -> str:
    r = ollama.chat(model=model, messages=messages)
    return r["message"]["content"]

# ── 3. Streaming ──────────────────────────────────────────────
def stream_chat(messages: list[dict]) -> Generator[str, None, None]:
    for chunk in ollama.chat(model="llama3.2", messages=messages, stream=True):
        yield chunk["message"]["content"]

# ── 4. Structured Output (JSON mode) ─────────────────────────
def structured_output(prompt: str, schema_hint: str) -> dict:
    r = ollama.chat(
        model="llama3.2",
        messages=[{
            "role": "user",
            "content": f"{prompt}\n\nRespond ONLY with valid JSON matching: {schema_hint}"
        }],
        format="json",   # Ollama JSON mode
    )
    return json.loads(r["message"]["content"])

# ── 5. Batch Processing ───────────────────────────────────────
async def batch_generate(prompts: list[str], model: str = "llama3.2"):
    client = ollama.AsyncClient()
    tasks = [
        client.generate(model=model, prompt=p)
        for p in prompts
    ]
    return await asyncio.gather(*tasks)
```

### 5.2 Conversation State Management

```python
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class Message:
    role: str      # "system" | "user" | "assistant" | "tool"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)

class ConversationManager:
    def __init__(self, system_prompt: str, max_tokens: int = 4096):
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.history: list[Message] = []
    
    def add(self, role: str, content: str, **meta):
        self.history.append(Message(role=role, content=content, metadata=meta))
    
    def to_ollama_messages(self) -> list[dict]:
        messages = [{"role": "system", "content": self.system_prompt}]
        messages += [{"role": m.role, "content": m.content} for m in self.history]
        return messages
    
    def trim_to_fit(self):
        """Remove oldest messages when approaching context limit"""
        # Rough token estimate: 1 token ≈ 4 chars
        while self._estimated_tokens() > self.max_tokens * 0.8:
            if len(self.history) > 1:
                self.history.pop(0)
    
    def _estimated_tokens(self) -> int:
        total_chars = sum(len(m.content) for m in self.history)
        return total_chars // 4
    
    def chat(self, user_input: str) -> str:
        self.add("user", user_input)
        self.trim_to_fit()
        
        response = ollama.chat(
            model="llama3.2",
            messages=self.to_ollama_messages()
        )
        content = response["message"]["content"]
        self.add("assistant", content)
        return content
```

---

## 6. Prompt Engineering for Agents

### 6.1 Prompt Taxonomy

```python
# ── System Prompts ────────────────────────────────────────────
SYSTEM_PERSONA = """You are {name}, a {role} expert.
Your capabilities: {capabilities}
Your constraints: {constraints}
Output format: {format}"""

# ── Zero-Shot ─────────────────────────────────────────────────
ZERO_SHOT = "Classify the sentiment of this text: {text}"

# ── Few-Shot ──────────────────────────────────────────────────
FEW_SHOT = """Classify sentiment as POSITIVE, NEGATIVE, or NEUTRAL.

Text: "I love this product!" → POSITIVE
Text: "It's okay I guess." → NEUTRAL
Text: "Terrible experience." → NEGATIVE

Text: "{text}" →"""

# ── Chain-of-Thought ──────────────────────────────────────────
COT_PROMPT = """Solve this step by step:

Problem: {problem}

Let me think through this carefully:
Step 1:"""

# ── Self-Consistency ──────────────────────────────────────────
# Run same prompt 3-5× with temperature>0, majority-vote answer

# ── Tree-of-Thought ───────────────────────────────────────────
TOT_PROMPT = """Generate 3 different approaches to solve this problem.
For each approach, evaluate its pros and cons.
Then select the best approach and implement it.

Problem: {problem}"""

# ── ReAct ─────────────────────────────────────────────────────
REACT_PROMPT = """Solve using this format:
Thought: [reason about next step]
Action: [tool_name(args)]
Observation: [tool result]
... repeat ...
Final Answer: [answer]

Task: {task}"""

# ── Role-play / Persona ───────────────────────────────────────
PERSONA_PROMPT = """You are a senior {role} at a Fortune 500 company.
You have 20 years of experience in {domain}.
You are reviewing: {artifact}
Provide specific, actionable feedback."""
```

### 6.2 Structured Output Prompts

```python
# JSON extraction
JSON_EXTRACT = """Extract the following information as JSON.
Return ONLY valid JSON, no markdown fences, no explanation.

Schema:
{{
  "name": "string",
  "date": "YYYY-MM-DD",
  "items": ["string"],
  "total": number
}}

Text to parse:
{text}"""

# Pydantic-guided extraction
from pydantic import BaseModel, Field

class ExtractedEntity(BaseModel):
    name: str = Field(description="Full name of the person")
    role: str = Field(description="Job title or role")
    company: str | None = Field(default=None)
    
def extract_entity(text: str) -> ExtractedEntity:
    schema = ExtractedEntity.model_json_schema()
    prompt = f"""Extract entity info as JSON matching this schema:
{json.dumps(schema, indent=2)}

Text: {text}

Return ONLY valid JSON:"""
    
    r = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}],
        format="json",
    )
    return ExtractedEntity.model_validate_json(r["message"]["content"])
```

### 6.3 Prompt Template Engine

```python
from string import Template

class PromptTemplate:
    def __init__(self, template: str):
        self.template = Template(template)
    
    def render(self, **kwargs) -> str:
        return self.template.safe_substitute(**kwargs)

# Usage
t = PromptTemplate("""
You are a $role.
Task: $task
Context: $context
Output format: $output_format
""")

prompt = t.render(
    role="Python expert",
    task="Write unit tests",
    context="FastAPI endpoint",
    output_format="pytest code only"
)
```

---

## 7. Tool Use & Function Calling

### 7.1 Tool Definition Schema

```python
from dataclasses import dataclass
from typing import Callable, Any
import inspect

@dataclass
class ToolParameter:
    name: str
    type: str            # "string" | "integer" | "number" | "boolean" | "array" | "object"
    description: str
    required: bool = True
    enum: list | None = None

@dataclass  
class Tool:
    name: str
    description: str
    parameters: list[ToolParameter]
    fn: Callable
    
    def to_schema(self) -> dict:
        """Convert to Ollama/OpenAI-compatible tool schema"""
        props = {}
        required = []
        for p in self.parameters:
            prop = {"type": p.type, "description": p.description}
            if p.enum:
                prop["enum"] = p.enum
            props[p.name] = prop
            if p.required:
                required.append(p.name)
        
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": props,
                    "required": required,
                }
            }
        }
    
    def execute(self, **kwargs) -> Any:
        return self.fn(**kwargs)
```

### 7.2 Ollama Native Tool Calling

```python
# Ollama ≥0.3 supports OpenAI-compatible tool calling

import ollama, json

def get_weather(city: str, unit: str = "celsius") -> str:
    # Mock implementation
    return json.dumps({"city": city, "temp": 22, "unit": unit, "condition": "sunny"})

def search_web(query: str, num_results: int = 5) -> str:
    # Mock implementation  
    return json.dumps({"results": [f"Result {i} for '{query}'" for i in range(num_results)]})

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "num_results": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
]

TOOL_REGISTRY = {"get_weather": get_weather, "search_web": search_web}

def tool_agent(user_input: str) -> str:
    messages = [{"role": "user", "content": user_input}]
    
    while True:
        response = ollama.chat(
            model="llama3.1:8b",   # llama3.1+ needed for native tool calls
            messages=messages,
            tools=TOOLS,
        )
        
        msg = response["message"]
        messages.append({"role": "assistant", "content": msg.get("content", ""), 
                          "tool_calls": msg.get("tool_calls", [])})
        
        # Check for tool calls
        if not msg.get("tool_calls"):
            return msg["content"]
        
        # Execute tool calls
        for tool_call in msg["tool_calls"]:
            fn_name = tool_call["function"]["name"]
            fn_args = tool_call["function"]["arguments"]
            
            result = TOOL_REGISTRY[fn_name](**fn_args)
            
            messages.append({
                "role": "tool",
                "content": result,
            })

# Usage
print(tool_agent("What's the weather in Berlin? And search for 'Bundesliga standings 2025'"))
```

### 7.3 Tool Library — Common Skills

```python
import subprocess, os, requests
from pathlib import Path

# ── File System Tools ─────────────────────────────────────────
def read_file(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")

def write_file(path: str, content: str) -> str:
    Path(path).write_text(content, encoding="utf-8")
    return f"Written {len(content)} chars to {path}"

def list_directory(path: str = ".", pattern: str = "*") -> list[str]:
    return [str(p) for p in Path(path).glob(pattern)]

# ── Code Execution Tools ──────────────────────────────────────
def run_python(code: str, timeout: int = 30) -> dict:
    try:
        result = subprocess.run(
            ["python3", "-c", code],
            capture_output=True, text=True, timeout=timeout
        )
        return {"stdout": result.stdout, "stderr": result.stderr, 
                "returncode": result.returncode}
    except subprocess.TimeoutExpired:
        return {"error": f"Timed out after {timeout}s"}

def run_shell(command: str, cwd: str | None = None) -> dict:
    result = subprocess.run(
        command, shell=True, capture_output=True, 
        text=True, cwd=cwd, timeout=60
    )
    return {"stdout": result.stdout, "stderr": result.stderr}

# ── HTTP Tools ────────────────────────────────────────────────
def http_get(url: str, headers: dict = {}) -> dict:
    r = requests.get(url, headers=headers, timeout=15)
    return {"status": r.status_code, "body": r.text[:4000]}

def http_post(url: str, data: dict, headers: dict = {}) -> dict:
    r = requests.post(url, json=data, headers=headers, timeout=15)
    return {"status": r.status_code, "body": r.json()}

# ── Math/Calc Tools ───────────────────────────────────────────
def calculator(expression: str) -> dict:
    try:
        # Safe eval with math module
        import math
        result = eval(expression, {"__builtins__": {}}, vars(math))
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}

# ── Datetime Tools ────────────────────────────────────────────
def get_current_datetime() -> dict:
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    return {
        "utc": now.isoformat(),
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "weekday": now.strftime("%A"),
    }
```

---

## 8. Model Context Protocol (MCP)

### 8.1 MCP Overview

```
Model Context Protocol (MCP) — Anthropic, 2024
An open standard for connecting LLM agents to external tools and data.

┌────────────────────────────────────────────────────────────────┐
│                          MCP Architecture                      │
│                                                                │
│  ┌──────────────┐   JSON-RPC 2.0    ┌──────────────────────┐  │
│  │  MCP CLIENT  │◄─────────────────▶│    MCP SERVER        │  │
│  │  (AI Agent)  │   stdio / SSE     │                      │  │
│  │              │                   │  ┌──────────────────┐ │  │
│  │  • Prompts   │                   │  │ Tools (actions)  │ │  │
│  │  • Tools     │                   │  │ Resources (data) │ │  │
│  │  • Resources │                   │  │ Prompts (templates│ │  │
│  └──────────────┘                   │  └──────────────────┘ │  │
│                                     └──────────────────────┘  │
└────────────────────────────────────────────────────────────────┘

Transport Options:
• stdio     — subprocess, local tools (most common)
• SSE       — HTTP Server-Sent Events, remote services
• WebSocket — real-time bidirectional (emerging)
```

### 8.2 MCP Server Implementation (Python)

```python
pip install mcp
```

```python
# mcp_server.py — Minimal MCP server with Ollama backend
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types
import ollama, json

app = Server("ollama-mcp-server")

# ── Register Tools ────────────────────────────────────────────
@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="ollama_generate",
            description="Generate text using a local Ollama model",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "The prompt"},
                    "model": {"type": "string", "default": "llama3.2"},
                    "temperature": {"type": "number", "default": 0.7},
                },
                "required": ["prompt"],
            },
        ),
        types.Tool(
            name="ollama_embed",
            description="Generate embeddings for text",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "model": {"type": "string", "default": "nomic-embed-text"},
                },
                "required": ["text"],
            },
        ),
        types.Tool(
            name="ollama_list_models",
            description="List available local Ollama models",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    if name == "ollama_generate":
        r = ollama.generate(
            model=arguments.get("model", "llama3.2"),
            prompt=arguments["prompt"],
            options={"temperature": arguments.get("temperature", 0.7)},
        )
        return [types.TextContent(type="text", text=r["response"])]
    
    elif name == "ollama_embed":
        r = ollama.embed(
            model=arguments.get("model", "nomic-embed-text"),
            input=arguments["text"],
        )
        return [types.TextContent(type="text", text=json.dumps(r["embeddings"][0][:10]))]
    
    elif name == "ollama_list_models":
        models = ollama.list()
        names = [m["name"] for m in models["models"]]
        return [types.TextContent(type="text", text="\n".join(names))]
    
    raise ValueError(f"Unknown tool: {name}")

# ── Register Resources ────────────────────────────────────────
@app.list_resources()
async def list_resources() -> list[types.Resource]:
    return [
        types.Resource(
            uri="ollama://models",
            name="Available Ollama Models",
            mimeType="application/json",
        )
    ]

@app.read_resource()
async def read_resource(uri: str) -> str:
    if uri == "ollama://models":
        models = ollama.list()
        return json.dumps(models["models"], indent=2)
    raise ValueError(f"Unknown resource: {uri}")

# ── Register Prompts ──────────────────────────────────────────
@app.list_prompts()
async def list_prompts() -> list[types.Prompt]:
    return [
        types.Prompt(
            name="code_review",
            description="Review code for bugs and improvements",
            arguments=[
                types.PromptArgument(name="code", description="Code to review", required=True),
                types.PromptArgument(name="language", description="Programming language"),
            ],
        )
    ]

@app.get_prompt()
async def get_prompt(name: str, arguments: dict | None) -> types.GetPromptResult:
    if name == "code_review":
        code = (arguments or {}).get("code", "")
        lang = (arguments or {}).get("language", "Python")
        return types.GetPromptResult(
            description="Code review prompt",
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(
                        type="text",
                        text=f"Review this {lang} code for bugs, style, and improvements:\n\n```{lang.lower()}\n{code}\n```"
                    )
                )
            ]
        )
    raise ValueError(f"Unknown prompt: {name}")

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

### 8.3 MCP Client Integration

```python
# Using MCP servers from a Python agent
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def use_mcp_server():
    server_params = StdioServerParameters(
        command="python",
        args=["mcp_server.py"],
        env=None,
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # List available tools
            tools = await session.list_tools()
            print([t.name for t in tools.tools])
            
            # Call a tool
            result = await session.call_tool(
                "ollama_generate",
                {"prompt": "What is the capital of France?", "model": "llama3.2"}
            )
            print(result.content[0].text)
            
            # Read a resource
            resource = await session.read_resource("ollama://models")
            print(resource.contents[0].text)
```

### 8.4 MCP Configuration (Claude Desktop / compatible clients)

```json
// ~/.config/claude/claude_desktop_config.json
{
  "mcpServers": {
    "ollama": {
      "command": "python",
      "args": ["/path/to/mcp_server.py"],
      "env": {
        "OLLAMA_HOST": "http://localhost:11434"
      }
    },
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/home/user/projects"]
    },
    "sqlite": {
      "command": "uvx",
      "args": ["mcp-server-sqlite", "--db-path", "/path/to/db.sqlite"]
    },
    "brave-search": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-brave-search"],
      "env": {
        "BRAVE_API_KEY": "YOUR_API_KEY"
      }
    }
  }
}
```

---

## 9. Agent Skills & Capabilities

### 9.1 Skill Architecture

```python
# A "Skill" = prompt template + tools + validation + output parser
from abc import ABC, abstractmethod
from pydantic import BaseModel

class SkillInput(BaseModel):
    pass

class SkillOutput(BaseModel):
    success: bool
    result: str
    metadata: dict = {}

class Skill(ABC):
    name: str
    description: str
    version: str = "1.0.0"
    
    @abstractmethod
    def get_system_prompt(self) -> str: ...
    
    @abstractmethod
    def validate_input(self, inp: SkillInput) -> bool: ...
    
    @abstractmethod
    async def execute(self, inp: SkillInput) -> SkillOutput: ...
    
    def to_tool_schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.input_schema(),
        }
```

### 9.2 Built-in Skill Library

```python
# ── Summarization Skill ───────────────────────────────────────
class SummarizationSkill(Skill):
    name = "summarize"
    description = "Summarize text at specified length and style"
    
    PROMPT = """Summarize the following text.
Style: {style}   (bullet_points | paragraph | executive | technical)
Max length: {max_words} words

Text:
{text}

Summary:"""
    
    async def execute(self, text: str, style: str = "paragraph", 
                      max_words: int = 100) -> str:
        r = ollama.chat(model="llama3.2", messages=[{
            "role": "user",
            "content": self.PROMPT.format(text=text, style=style, max_words=max_words)
        }])
        return r["message"]["content"]

# ── Code Generation Skill ─────────────────────────────────────
class CodeGenSkill(Skill):
    name = "generate_code"
    description = "Generate code from natural language description"
    
    PROMPT = """Write {language} code for the following requirement.
Requirements: {requirement}
Additional context: {context}

Rules:
- Return ONLY code, no explanation
- Include docstrings
- Handle edge cases
- Add type hints (if applicable)

```{language.lower()}"""
    
    async def execute(self, requirement: str, language: str = "Python",
                      context: str = "") -> str:
        r = ollama.chat(model="llama3.2", messages=[{
            "role": "user",
            "content": self.PROMPT.format(
                language=language, requirement=requirement, context=context
            )
        }])
        return self._extract_code(r["message"]["content"])
    
    def _extract_code(self, text: str) -> str:
        import re
        pattern = r"```(?:\w+)?\n(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)
        return matches[0].strip() if matches else text.strip()

# ── Classification Skill ──────────────────────────────────────
class ClassificationSkill(Skill):
    name = "classify"
    description = "Classify text into predefined categories"
    
    async def execute(self, text: str, categories: list[str],
                      examples: list[dict] = []) -> dict:
        few_shot = "\n".join([
            f'Text: "{e["text"]}" → {e["label"]}' for e in examples
        ])
        
        prompt = f"""Classify the text into one of: {', '.join(categories)}
{few_shot}

Text: "{text}" →
Return ONLY the category name, nothing else."""
        
        r = ollama.chat(model="llama3.2", messages=[
            {"role": "user", "content": prompt}
        ], options={"temperature": 0})
        
        label = r["message"]["content"].strip()
        return {"label": label, "confidence": 0.9}  # mock confidence

# ── Translation Skill ─────────────────────────────────────────
class TranslationSkill(Skill):
    name = "translate"
    
    async def execute(self, text: str, target_lang: str, 
                      source_lang: str = "auto") -> str:
        r = ollama.chat(model="llama3.2", messages=[{
            "role": "user",
            "content": f"Translate to {target_lang}. Return ONLY the translation:\n\n{text}"
        }])
        return r["message"]["content"]

# ── Data Extraction Skill ─────────────────────────────────────
class DataExtractionSkill(Skill):
    name = "extract_data"
    
    async def execute(self, text: str, schema: dict) -> dict:
        r = ollama.chat(
            model="llama3.2",
            messages=[{
                "role": "user", 
                "content": f"""Extract data as JSON matching this schema:
{json.dumps(schema, indent=2)}

Text: {text}

Return ONLY valid JSON:"""
            }],
            format="json",
        )
        return json.loads(r["message"]["content"])
```

### 9.3 Skill Registry

```python
class SkillRegistry:
    def __init__(self):
        self._skills: dict[str, Skill] = {}
    
    def register(self, skill: Skill):
        self._skills[skill.name] = skill
    
    def get(self, name: str) -> Skill:
        return self._skills[name]
    
    def list(self) -> list[str]:
        return list(self._skills.keys())
    
    def to_tools_schema(self) -> list[dict]:
        return [s.to_tool_schema() for s in self._skills.values()]

# Global registry
registry = SkillRegistry()
registry.register(SummarizationSkill())
registry.register(CodeGenSkill())
registry.register(ClassificationSkill())
registry.register(TranslationSkill())
```

---

## 10. Multi-Agent Systems

### 10.1 Orchestrator-Worker Pattern

```python
# Orchestrator delegates tasks to specialized sub-agents
from dataclasses import dataclass
from enum import Enum

class AgentRole(Enum):
    ORCHESTRATOR = "orchestrator"
    RESEARCHER = "researcher"
    WRITER = "writer"
    CRITIC = "critic"
    CODER = "coder"

@dataclass
class AgentMessage:
    sender: str
    receiver: str
    content: str
    message_type: str  # "task" | "result" | "feedback"

class BaseAgent:
    def __init__(self, role: AgentRole, model: str = "llama3.2"):
        self.role = role
        self.model = model
        self.inbox: list[AgentMessage] = []
    
    def system_prompt(self) -> str:
        prompts = {
            AgentRole.RESEARCHER: "You are a research analyst. Find and synthesize information.",
            AgentRole.WRITER: "You are a technical writer. Create clear, structured content.",
            AgentRole.CRITIC: "You are a critical reviewer. Identify weaknesses and suggest improvements.",
            AgentRole.CODER: "You are a senior software engineer. Write clean, tested code.",
        }
        return prompts.get(self.role, "You are a helpful AI assistant.")
    
    def process(self, task: str) -> str:
        r = ollama.chat(model=self.model, messages=[
            {"role": "system", "content": self.system_prompt()},
            {"role": "user", "content": task},
        ])
        return r["message"]["content"]

class OrchestratorAgent(BaseAgent):
    def __init__(self, workers: dict[str, BaseAgent]):
        super().__init__(AgentRole.ORCHESTRATOR)
        self.workers = workers
    
    def decompose_task(self, task: str) -> list[dict]:
        r = ollama.chat(model=self.model, messages=[{
            "role": "user",
            "content": f"""Decompose this task into sub-tasks for specialized agents.
Available agents: {list(self.workers.keys())}
Task: {task}

Return JSON array:
[{{"agent": "researcher", "task": "...", "depends_on": []}}, ...]
Return ONLY JSON:"""
        }], format="json")
        return json.loads(r["message"]["content"])
    
    def run(self, task: str) -> str:
        subtasks = self.decompose_task(task)
        results = {}
        
        for subtask in subtasks:
            agent_name = subtask["agent"]
            if agent_name in self.workers:
                # Inject results from dependencies
                context = "\n".join([
                    f"{dep}: {results[dep]}" 
                    for dep in subtask.get("depends_on", [])
                    if dep in results
                ])
                full_task = f"{subtask['task']}\n\nContext:\n{context}" if context else subtask["task"]
                results[agent_name] = self.workers[agent_name].process(full_task)
        
        return self.synthesize(task, results)
    
    def synthesize(self, task: str, results: dict) -> str:
        context = "\n\n".join([f"=== {k} ===\n{v}" for k, v in results.items()])
        r = ollama.chat(model=self.model, messages=[{
            "role": "user",
            "content": f"Synthesize these agent outputs into a final answer for:\n{task}\n\n{context}"
        }])
        return r["message"]["content"]

# Usage
workers = {
    "researcher": BaseAgent(AgentRole.RESEARCHER),
    "writer": BaseAgent(AgentRole.WRITER),
    "critic": BaseAgent(AgentRole.CRITIC),
}
orchestrator = OrchestratorAgent(workers)
result = orchestrator.run("Write a technical blog post about transformer architectures")
```

### 10.2 Debate / Critique Pattern

```python
class DebateOrchestrator:
    """Two agents debate a proposition, third synthesizes verdict"""
    
    def run(self, proposition: str, rounds: int = 2) -> dict:
        pro_agent = BaseAgent(AgentRole.RESEARCHER, model="llama3.2")
        con_agent = BaseAgent(AgentRole.CRITIC, model="llama3.2")
        
        history = []
        
        # Initial positions
        pro = pro_agent.process(f"Argue FOR this proposition: {proposition}")
        con = con_agent.process(f"Argue AGAINST this proposition: {proposition}")
        history.append({"round": 0, "pro": pro, "con": con})
        
        # Debate rounds
        for i in range(rounds):
            pro_rebuttal = pro_agent.process(
                f"Proposition: {proposition}\nOpponent argued: {con}\nRefute and strengthen your position:"
            )
            con_rebuttal = con_agent.process(
                f"Proposition: {proposition}\nOpponent argued: {pro_rebuttal}\nRefute and strengthen your position:"
            )
            history.append({"round": i+1, "pro": pro_rebuttal, "con": con_rebuttal})
            pro, con = pro_rebuttal, con_rebuttal
        
        # Synthesis
        judge = BaseAgent(AgentRole.WRITER)
        verdict = judge.process(
            f"Evaluate this debate on: {proposition}\n"
            + "\n".join([f"Round {h['round']}: PRO: {h['pro'][:200]}... CON: {h['con'][:200]}..." 
                         for h in history])
            + "\nProvide a balanced verdict with strongest points from each side."
        )
        
        return {"proposition": proposition, "history": history, "verdict": verdict}
```

---

## 11. Memory Systems

### 11.1 Memory Taxonomy

```
┌─────────────────────────────────────────────────────────────┐
│                     AGENT MEMORY TYPES                      │
├─────────────────┬───────────────────────────────────────────┤
│ Working Memory  │ Current context window (token-limited)    │
│                 │ → Managed via context trimming/summarize   │
├─────────────────┼───────────────────────────────────────────┤
│ Episodic Memory │ Past conversation history                  │
│                 │ → SQLite / Redis / JSON files              │
├─────────────────┼───────────────────────────────────────────┤
│ Semantic Memory │ Facts, knowledge, documents                │
│                 │ → Vector DB (ChromaDB, Qdrant, FAISS)      │
├─────────────────┼───────────────────────────────────────────┤
│ Procedural      │ How to do things (skills, workflows)       │
│                 │ → Code/tool registry                       │
└─────────────────┴───────────────────────────────────────────┘
```

### 11.2 Vector Memory with ChromaDB

```python
pip install chromadb ollama
```

```python
import chromadb
import ollama
from chromadb.utils import embedding_functions

class VectorMemory:
    def __init__(self, collection_name: str = "agent_memory"):
        self.client = chromadb.PersistentClient(path="./memory_db")
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
    
    def _embed(self, text: str) -> list[float]:
        r = ollama.embed(model="nomic-embed-text", input=text)
        return r["embeddings"][0]
    
    def store(self, text: str, metadata: dict = {}, doc_id: str | None = None):
        import uuid
        doc_id = doc_id or str(uuid.uuid4())
        self.collection.add(
            ids=[doc_id],
            embeddings=[self._embed(text)],
            documents=[text],
            metadatas=[metadata],
        )
        return doc_id
    
    def search(self, query: str, n_results: int = 5, 
               where: dict | None = None) -> list[dict]:
        results = self.collection.query(
            query_embeddings=[self._embed(query)],
            n_results=n_results,
            where=where,
        )
        return [
            {"id": id_, "text": doc, "metadata": meta, "distance": dist}
            for id_, doc, meta, dist in zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]
    
    def update(self, doc_id: str, text: str, metadata: dict = {}):
        self.collection.update(
            ids=[doc_id],
            embeddings=[self._embed(text)],
            documents=[text],
            metadatas=[metadata],
        )
    
    def delete(self, doc_id: str):
        self.collection.delete(ids=[doc_id])
    
    def count(self) -> int:
        return self.collection.count()

# Usage
memory = VectorMemory("project_memory")

# Store facts
memory.store("The Bundesliga season runs from August to May.", 
             metadata={"topic": "football", "source": "manual"})
memory.store("PySpark DataFrames are lazily evaluated.",
             metadata={"topic": "engineering", "source": "docs"})

# Retrieve relevant context
results = memory.search("How does Spark handle data processing?", n_results=3)
context = "\n".join([r["text"] for r in results])
```

### 11.3 Episodic Memory (SQLite)

```python
import sqlite3
from datetime import datetime

class EpisodicMemory:
    def __init__(self, db_path: str = "episodes.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_schema()
    
    def _init_schema(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                created_at TIMESTAMP,
                metadata TEXT
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                role TEXT,
                content TEXT,
                timestamp TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );
            CREATE INDEX IF NOT EXISTS idx_session ON messages(session_id);
        """)
        self.conn.commit()
    
    def create_session(self, session_id: str, metadata: dict = {}):
        self.conn.execute(
            "INSERT INTO sessions VALUES (?, ?, ?)",
            (session_id, datetime.now(), json.dumps(metadata))
        )
        self.conn.commit()
    
    def add_message(self, session_id: str, role: str, content: str):
        self.conn.execute(
            "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
            (session_id, role, content, datetime.now())
        )
        self.conn.commit()
    
    def get_session_history(self, session_id: str) -> list[dict]:
        cursor = self.conn.execute(
            "SELECT role, content FROM messages WHERE session_id=? ORDER BY timestamp",
            (session_id,)
        )
        return [{"role": row[0], "content": row[1]} for row in cursor.fetchall()]
    
    def summarize_session(self, session_id: str) -> str:
        history = self.get_session_history(session_id)
        if not history:
            return ""
        text = "\n".join([f"{m['role']}: {m['content']}" for m in history])
        r = ollama.chat(model="llama3.2", messages=[{
            "role": "user",
            "content": f"Summarize this conversation in 3 sentences:\n\n{text}"
        }])
        return r["message"]["content"]
```

---

## 12. RAG — Retrieval-Augmented Generation

### 12.1 RAG Pipeline Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                        RAG PIPELINE                            │
│                                                                │
│  INDEXING PHASE (offline)                                      │
│  Documents → Chunking → Embedding → Vector Store               │
│                                                                │
│  RETRIEVAL PHASE (online)                                      │
│  Query → Embed → Similarity Search → Top-K Chunks             │
│                                                                │
│  GENERATION PHASE (online)                                     │
│  [System + Context Chunks + Query] → LLM → Answer             │
└────────────────────────────────────────────────────────────────┘
```

### 12.2 Full RAG Implementation

```python
from pathlib import Path
import re

class RAGSystem:
    def __init__(self, model: str = "llama3.2", 
                 embed_model: str = "nomic-embed-text"):
        self.model = model
        self.embed_model = embed_model
        self.memory = VectorMemory("rag_store")
    
    # ── Ingestion ─────────────────────────────────────────────
    def ingest_text(self, text: str, source: str = "unknown",
                    chunk_size: int = 512, overlap: int = 50):
        chunks = self._chunk_text(text, chunk_size, overlap)
        for i, chunk in enumerate(chunks):
            self.memory.store(chunk, metadata={"source": source, "chunk": i})
        return len(chunks)
    
    def ingest_file(self, path: str | Path):
        text = Path(path).read_text(encoding="utf-8")
        return self.ingest_text(text, source=str(path))
    
    def ingest_directory(self, dir_path: str, pattern: str = "*.txt"):
        total = 0
        for p in Path(dir_path).glob(pattern):
            total += self.ingest_file(p)
        return total
    
    def _chunk_text(self, text: str, size: int, overlap: int) -> list[str]:
        # Sentence-aware chunking
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks, current, current_len = [], [], 0
        
        for sentence in sentences:
            words = sentence.split()
            if current_len + len(words) > size and current:
                chunks.append(" ".join(current))
                # Overlap: keep last N words
                overlap_words = current[-overlap:] if len(current) > overlap else current
                current = overlap_words + words
                current_len = len(current)
            else:
                current.extend(words)
                current_len += len(words)
        
        if current:
            chunks.append(" ".join(current))
        return chunks
    
    # ── Retrieval ─────────────────────────────────────────────
    def retrieve(self, query: str, n_results: int = 5) -> list[dict]:
        return self.memory.search(query, n_results=n_results)
    
    # ── Generation ────────────────────────────────────────────
    def query(self, question: str, n_results: int = 5,
              temperature: float = 0.1) -> dict:
        chunks = self.retrieve(question, n_results)
        context = "\n\n---\n\n".join([
            f"[Source: {c['metadata'].get('source', 'unknown')}]\n{c['text']}"
            for c in chunks
        ])
        
        prompt = f"""Answer the question based ONLY on the provided context.
If the context doesn't contain enough information, say so clearly.

Context:
{context}

Question: {question}

Answer:"""
        
        r = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": temperature},
        )
        
        return {
            "question": question,
            "answer": r["message"]["content"],
            "sources": [c["metadata"].get("source") for c in chunks],
            "chunks_used": len(chunks),
        }
    
    # ── Hybrid Search ─────────────────────────────────────────
    def hybrid_query(self, question: str) -> dict:
        """Combine vector search with keyword matching"""
        vector_results = self.retrieve(question, n_results=5)
        
        # Simple keyword scoring as fallback
        keywords = set(question.lower().split())
        for r in vector_results:
            doc_words = set(r["text"].lower().split())
            r["keyword_score"] = len(keywords & doc_words) / len(keywords)
        
        # Blend scores
        for r in vector_results:
            r["combined_score"] = (1 - r["distance"]) * 0.7 + r["keyword_score"] * 0.3
        
        vector_results.sort(key=lambda x: x["combined_score"], reverse=True)
        return self.query(question)

# Usage
rag = RAGSystem()
rag.ingest_file("documentation.txt")
result = rag.query("How do I configure the connection pool?")
print(result["answer"])
```

### 12.3 Advanced RAG Patterns

```python
# ── Self-Query RAG ────────────────────────────────────────────
def self_query_rag(rag: RAGSystem, question: str) -> str:
    """LLM rewrites query for better retrieval"""
    rewritten = ollama.chat(model="llama3.2", messages=[{
        "role": "user",
        "content": f"""Rewrite this question to maximize retrieval from a technical document store.
Keep it specific, add synonyms, remove stop words.
Original: {question}
Rewritten:"""
    }])["message"]["content"]
    
    return rag.query(rewritten)["answer"]

# ── Hypothetical Document Embeddings (HyDE) ───────────────────
def hyde_rag(rag: RAGSystem, question: str) -> str:
    """Generate hypothetical answer, use it as search query"""
    hypothesis = ollama.chat(model="llama3.2", messages=[{
        "role": "user",
        "content": f"Write a plausible 2-sentence answer to: {question}"
    }])["message"]["content"]
    
    chunks = rag.retrieve(hypothesis)  # Embed the hypothesis
    context = "\n".join([c["text"] for c in chunks])
    
    return ollama.chat(model="llama3.2", messages=[{
        "role": "user",
        "content": f"Answer based on context:\n{context}\n\nQuestion: {question}"
    }])["message"]["content"]

# ── Multi-Query RAG ───────────────────────────────────────────
def multi_query_rag(rag: RAGSystem, question: str) -> str:
    """Generate multiple queries, merge unique results"""
    variants = ollama.chat(model="llama3.2", messages=[{
        "role": "user",
        "content": f"""Generate 3 different search queries to find info about:
{question}
Return JSON array of strings only."""
    }], format="json")["message"]["content"]
    
    queries = json.loads(variants)
    all_chunks = {}
    for q in queries:
        for chunk in rag.retrieve(q, n_results=3):
            all_chunks[chunk["id"]] = chunk  # deduplicate by ID
    
    context = "\n\n".join([c["text"] for c in all_chunks.values()])
    return ollama.chat(model="llama3.2", messages=[{
        "role": "user",
        "content": f"Answer: {question}\n\nContext:\n{context}"
    }])["message"]["content"]
```

---

## 13. Agentic Workflows & Pipelines

### 13.1 Sequential Pipeline

```python
from typing import TypeVar, Generic
T = TypeVar("T")

class PipelineStep:
    def __init__(self, name: str, fn: callable):
        self.name = name
        self.fn = fn
    
    def run(self, input_data, context: dict) -> tuple:
        result = self.fn(input_data, context)
        return result, context

class Pipeline:
    def __init__(self, steps: list[PipelineStep]):
        self.steps = steps
    
    def run(self, initial_input, context: dict = {}) -> dict:
        data = initial_input
        trace = []
        
        for step in self.steps:
            data, context = step.run(data, context)
            trace.append({"step": step.name, "output_preview": str(data)[:100]})
        
        return {"result": data, "trace": trace, "context": context}

# Example: document processing pipeline
def extract(raw_text, ctx):
    entities = DataExtractionSkill().execute(raw_text, {"name": "str", "date": "str"})
    return entities, ctx

def enrich(entities, ctx):
    # Add metadata from external source
    entities["enriched"] = True
    return entities, ctx

def summarize(entities, ctx):
    summary = SummarizationSkill().execute(str(entities), style="executive")
    return summary, ctx

pipeline = Pipeline([
    PipelineStep("extract", extract),
    PipelineStep("enrich", enrich),
    PipelineStep("summarize", summarize),
])
```

### 13.2 Event-Driven Agent Workflow

```python
import asyncio
from enum import Enum
from dataclasses import dataclass, field

class EventType(Enum):
    TASK_RECEIVED = "task_received"
    TOOL_CALLED = "tool_called"
    TOOL_RESULT = "tool_result"
    STEP_COMPLETE = "step_complete"
    AGENT_DONE = "agent_done"
    ERROR = "error"

@dataclass
class Event:
    type: EventType
    data: dict
    source: str = "system"
    timestamp: float = field(default_factory=lambda: __import__("time").time())

class EventBus:
    def __init__(self):
        self._handlers: dict[EventType, list[callable]] = {}
        self._queue: asyncio.Queue = asyncio.Queue()
    
    def subscribe(self, event_type: EventType, handler: callable):
        self._handlers.setdefault(event_type, []).append(handler)
    
    async def publish(self, event: Event):
        await self._queue.put(event)
    
    async def dispatch(self):
        while True:
            event = await self._queue.get()
            for handler in self._handlers.get(event.type, []):
                await handler(event)

# Usage
bus = EventBus()

async def on_task(event: Event):
    print(f"Task received: {event.data['task']}")

bus.subscribe(EventType.TASK_RECEIVED, on_task)
```

### 13.3 Retry & Fault Tolerance

```python
import time
from functools import wraps

def retry(max_attempts: int = 3, delay: float = 1.0, 
          backoff: float = 2.0, exceptions: tuple = (Exception,)):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            attempt, current_delay = 0, delay
            while attempt < max_attempts:
                try:
                    return fn(*args, **kwargs)
                except exceptions as e:
                    attempt += 1
                    if attempt >= max_attempts:
                        raise
                    time.sleep(current_delay)
                    current_delay *= backoff
        return wrapper
    return decorator

@retry(max_attempts=3, delay=1.0, backoff=2.0)
def call_ollama_with_retry(messages: list[dict], model: str = "llama3.2") -> str:
    r = ollama.chat(model=model, messages=messages)
    return r["message"]["content"]

# Fallback chain
def llm_with_fallback(prompt: str, models: list[str] = None) -> str:
    models = models or ["llama3.2:3b", "llama3.2:1b", "mistral:7b"]
    for model in models:
        try:
            r = ollama.generate(model=model, prompt=prompt)
            return r["response"]
        except Exception as e:
            print(f"Model {model} failed: {e}, trying next...")
    raise RuntimeError("All models failed")
```

---

## 14. Design Patterns Catalog

### 14.1 Pattern Reference Table

| Pattern | Category | When To Use | Key Components |
|---|---|---|---|
| **ReAct** | Agent Loop | Dynamic problem solving | Thought-Action-Observation |
| **Chain of Thought** | Reasoning | Complex multi-step problems | Step-by-step reasoning |
| **Plan & Execute** | Planning | Long-horizon tasks | Planner + Executor |
| **Reflection** | Quality | Self-improvement | Generator + Critic |
| **Tool Use** | Integration | External data/actions | Tool schema + parser |
| **RAG** | Knowledge | Grounding in docs | Embedder + Retriever + LLM |
| **Multi-Agent** | Scale | Task parallelism | Orchestrator + Workers |
| **Debate** | Accuracy | Controversial topics | Pro + Con + Judge |
| **Critic-in-the-Loop** | QA | High-stakes outputs | Generator + Validator |
| **Router** | Efficiency | Mixed task types | Classifier + Specialized LLMs |
| **Memory Distillation** | Memory | Long conversations | Summarize old into semantic |
| **HyDE** | Retrieval | Sparse queries | Hypothetical doc embeddings |
| **Self-Refine** | Quality | Iterative improvement | Generator + Evaluator loop |
| **Skeleton-of-Thought** | Speed | Long generation | Parallel skeleton + expansion |

### 14.2 Reflection / Self-Refine Pattern

```python
class SelfRefineAgent:
    def __init__(self, model: str = "llama3.2", max_rounds: int = 3):
        self.model = model
        self.max_rounds = max_rounds
    
    def generate(self, task: str) -> str:
        r = ollama.chat(model=self.model, messages=[
            {"role": "user", "content": task}
        ])
        return r["message"]["content"]
    
    def critique(self, task: str, output: str) -> dict:
        r = ollama.chat(model=self.model, messages=[{
            "role": "user",
            "content": f"""Critique this output for the given task.
Task: {task}
Output: {output}

Provide:
1. Score (1-10)
2. Specific issues
3. Concrete suggestions

Return JSON: {{"score": 8, "issues": [...], "suggestions": [...]}}"""
        }], format="json")
        return json.loads(r["message"]["content"])
    
    def refine(self, task: str, output: str, critique: dict) -> str:
        r = ollama.chat(model=self.model, messages=[{
            "role": "user",
            "content": f"""Improve this output based on the critique.
Task: {task}
Current output: {output}
Issues: {critique['issues']}
Suggestions: {critique['suggestions']}

Improved output:"""
        }])
        return r["message"]["content"]
    
    def run(self, task: str, min_score: float = 8.0) -> dict:
        output = self.generate(task)
        history = [{"round": 0, "output": output}]
        
        for round_ in range(self.max_rounds):
            critique = self.critique(task, output)
            
            if critique["score"] >= min_score:
                break
            
            output = self.refine(task, output, critique)
            history.append({"round": round_ + 1, "output": output, 
                             "critique": critique})
        
        return {"final_output": output, "rounds": len(history), "history": history}
```

### 14.3 Router Pattern

```python
class AgentRouter:
    """Route tasks to specialized agents based on intent"""
    
    ROUTING_PROMPT = """Classify this task into exactly ONE category.
Categories: {categories}
Task: {task}
Return ONLY the category name:"""
    
    def __init__(self, agents: dict[str, BaseAgent]):
        self.agents = agents
    
    def route(self, task: str) -> str:
        categories = list(self.agents.keys())
        r = ollama.chat(
            model="llama3.2:1b",  # Use fast small model for routing
            messages=[{
                "role": "user",
                "content": self.ROUTING_PROMPT.format(
                    categories=", ".join(categories), task=task
                )
            }],
            options={"temperature": 0},
        )
        label = r["message"]["content"].strip().lower()
        return label if label in self.agents else "general"
    
    def dispatch(self, task: str) -> str:
        agent_name = self.route(task)
        agent = self.agents.get(agent_name, self.agents.get("general"))
        return agent.process(task)

# Setup
router = AgentRouter({
    "coding": BaseAgent(AgentRole.CODER),
    "research": BaseAgent(AgentRole.RESEARCHER),
    "writing": BaseAgent(AgentRole.WRITER),
    "general": BaseAgent(AgentRole.ORCHESTRATOR),
})

result = router.dispatch("Write a Python script to parse CSV files")
# → Routed to 'coding' agent
```

---

## 15. Project Structure Templates

### 15.1 Simple Agent Project

```
my-agent/
├── .env                    # OLLAMA_HOST, API keys
├── .gitignore
├── requirements.txt
├── README.md
│
├── src/
│   ├── __init__.py
│   ├── agent.py            # Core agent logic
│   ├── tools.py            # Tool definitions
│   ├── prompts.py          # Prompt templates
│   └── config.py           # Configuration
│
├── data/
│   └── knowledge_base/     # Documents for RAG
│
├── tests/
│   ├── test_agent.py
│   └── test_tools.py
│
└── scripts/
    ├── ingest.py           # Load documents into vector store
    └── evaluate.py         # Run eval benchmarks
```

### 15.2 Multi-Agent System Project

```
multi-agent-system/
├── pyproject.toml
├── .env
│
├── agents/
│   ├── __init__.py
│   ├── base.py             # BaseAgent ABC
│   ├── orchestrator.py     # OrchestratorAgent
│   ├── researcher.py       # ResearchAgent
│   ├── coder.py            # CodingAgent
│   ├── writer.py           # WriterAgent
│   └── critic.py           # CriticAgent
│
├── skills/
│   ├── __init__.py
│   ├── registry.py         # SkillRegistry
│   ├── summarize.py
│   ├── code_gen.py
│   ├── extract.py
│   └── translate.py
│
├── memory/
│   ├── __init__.py
│   ├── vector.py           # ChromaDB wrapper
│   ├── episodic.py         # SQLite history
│   └── cache.py            # LRU response cache
│
├── tools/
│   ├── __init__.py
│   ├── filesystem.py
│   ├── web.py
│   ├── code_runner.py
│   └── calculator.py
│
├── mcp/
│   ├── server.py           # MCP server
│   └── client.py           # MCP client wrapper
│
├── rag/
│   ├── ingestor.py
│   ├── retriever.py
│   └── pipeline.py
│
├── api/
│   ├── main.py             # FastAPI app
│   ├── routes.py
│   └── schemas.py
│
├── tests/
│   ├── unit/
│   └── integration/
│
└── config/
    ├── models.yaml          # Model configs
    └── agents.yaml          # Agent configs
```

### 15.3 RAG Application Project

```
rag-app/
├── pyproject.toml
├── Makefile
│
├── src/
│   ├── ingest/
│   │   ├── loader.py       # File loaders (PDF, DOCX, TXT, HTML)
│   │   ├── chunker.py      # Text chunking strategies
│   │   └── pipeline.py     # End-to-end ingestion
│   │
│   ├── retrieval/
│   │   ├── embedder.py     # Embedding with nomic-embed-text
│   │   ├── store.py        # ChromaDB / FAISS
│   │   └── reranker.py     # Cross-encoder reranking
│   │
│   ├── generation/
│   │   ├── prompts.py      # RAG prompt templates
│   │   └── chain.py        # Retrieval + Generation chain
│   │
│   └── api/
│       └── app.py          # Streamlit / FastAPI interface
│
├── data/
│   ├── raw/                # Source documents
│   └── processed/          # Chunked + indexed
│
└── chroma_db/              # Persistent vector store
```

---

## 16. Implementation Code Library

### 16.1 Production-Ready Agent Class

```python
# agent.py — Production agent with full feature set
import ollama, json, logging, time
from typing import Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class ProductionAgent:
    DEFAULT_OPTIONS = {
        "temperature": 0.1,
        "num_ctx": 8192,
        "num_predict": 2048,
        "repeat_penalty": 1.1,
    }
    
    def __init__(
        self,
        model: str = "llama3.2",
        system_prompt: str = "You are a helpful assistant.",
        tools: list[dict] = None,
        memory: VectorMemory = None,
        max_iterations: int = 10,
        options: dict = None,
    ):
        self.model = model
        self.system_prompt = system_prompt
        self.tools = tools or []
        self.memory = memory
        self.max_iterations = max_iterations
        self.options = {**self.DEFAULT_OPTIONS, **(options or {})}
        self.tool_registry: dict[str, callable] = {}
        self._conversation: list[dict] = []
    
    def register_tool(self, name: str, fn: callable):
        self.tool_registry[name] = fn
    
    def _get_context_from_memory(self, query: str, n: int = 3) -> str:
        if not self.memory:
            return ""
        results = self.memory.search(query, n_results=n)
        return "\n".join([r["text"] for r in results])
    
    def _build_messages(self, user_input: str) -> list[dict]:
        mem_context = self._get_context_from_memory(user_input)
        system = self.system_prompt
        if mem_context:
            system += f"\n\nRelevant knowledge:\n{mem_context}"
        
        return [
            {"role": "system", "content": system},
            *self._conversation,
            {"role": "user", "content": user_input},
        ]
    
    def _execute_tool(self, tool_call: dict) -> str:
        fn_name = tool_call["function"]["name"]
        fn_args = tool_call["function"]["arguments"]
        
        if fn_name not in self.tool_registry:
            return json.dumps({"error": f"Unknown tool: {fn_name}"})
        
        try:
            logger.debug(f"Calling tool: {fn_name}({fn_args})")
            result = self.tool_registry[fn_name](**fn_args)
            return json.dumps(result) if not isinstance(result, str) else result
        except Exception as e:
            logger.error(f"Tool {fn_name} failed: {e}")
            return json.dumps({"error": str(e)})
    
    def run(self, user_input: str) -> dict:
        start_time = time.time()
        messages = self._build_messages(user_input)
        iterations = 0
        tool_calls_made = []
        
        while iterations < self.max_iterations:
            iterations += 1
            
            kwargs = {"model": self.model, "messages": messages, "options": self.options}
            if self.tools:
                kwargs["tools"] = self.tools
            
            response = ollama.chat(**kwargs)
            msg = response["message"]
            messages.append({"role": "assistant", "content": msg.get("content", ""),
                             "tool_calls": msg.get("tool_calls", [])})
            
            # No tool calls → final answer
            if not msg.get("tool_calls"):
                self._conversation.append({"role": "user", "content": user_input})
                self._conversation.append({"role": "assistant", "content": msg["content"]})
                
                # Store in memory
                if self.memory:
                    self.memory.store(
                        f"Q: {user_input}\nA: {msg['content']}",
                        metadata={"type": "interaction", "ts": datetime.now().isoformat()}
                    )
                
                return {
                    "response": msg["content"],
                    "iterations": iterations,
                    "tool_calls": tool_calls_made,
                    "latency_ms": round((time.time() - start_time) * 1000),
                }
            
            # Execute tool calls
            for tc in msg["tool_calls"]:
                result = self._execute_tool(tc)
                tool_calls_made.append({
                    "tool": tc["function"]["name"],
                    "args": tc["function"]["arguments"],
                    "result_preview": result[:100],
                })
                messages.append({"role": "tool", "content": result})
        
        return {"response": "Max iterations reached.", "iterations": iterations,
                "tool_calls": tool_calls_made,
                "latency_ms": round((time.time() - start_time) * 1000)}
    
    def reset(self):
        self._conversation = []
```

### 16.2 FastAPI Agent Server

```python
# api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import uvicorn

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    model: str = "llama3.2"
    temperature: float = 0.1
    stream: bool = False

class ChatResponse(BaseModel):
    response: str
    session_id: str
    iterations: int
    latency_ms: int
    tool_calls: list[dict] = []

# Global agent store (per session)
agents: dict[str, ProductionAgent] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Agent API starting...")
    yield
    print("🛑 Agent API shutting down...")

app = FastAPI(title="Ollama Agent API", version="1.0.0", lifespan=lifespan)

def get_or_create_agent(session_id: str, model: str) -> ProductionAgent:
    if session_id not in agents:
        agents[session_id] = ProductionAgent(
            model=model,
            tools=TOOLS,
            memory=VectorMemory(f"session_{session_id}"),
        )
        for name, fn in TOOL_REGISTRY.items():
            agents[session_id].register_tool(name, fn)
    return agents[session_id]

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    agent = get_or_create_agent(req.session_id, req.model)
    result = agent.run(req.message)
    return ChatResponse(session_id=req.session_id, **result)

@app.delete("/sessions/{session_id}")
async def clear_session(session_id: str):
    if session_id in agents:
        agents[session_id].reset()
    return {"cleared": session_id}

@app.get("/models")
async def list_models():
    return ollama.list()

@app.get("/health")
async def health():
    return {"status": "ok", "models": len(ollama.list()["models"])}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
```

### 16.3 Streamlit Agent UI

```python
# app.py — Streamlit chat interface
import streamlit as st
import ollama

st.set_page_config(page_title="Ollama Agent", page_icon="🤖", layout="wide")

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Configuration")
    
    available_models = [m["name"] for m in ollama.list()["models"]]
    model = st.selectbox("Model", available_models, index=0)
    temperature = st.slider("Temperature", 0.0, 2.0, 0.1, 0.05)
    max_tokens = st.slider("Max Tokens", 256, 8192, 2048, 256)
    
    st.divider()
    system_prompt = st.text_area(
        "System Prompt",
        value="You are a helpful AI assistant powered by a local Ollama model.",
        height=150
    )
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []

# ── Main Chat ─────────────────────────────────────────────────
st.title("🤖 Ollama Agent")
st.caption(f"Running: `{model}` locally")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Message the agent..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    messages_for_api = [
        {"role": "system", "content": system_prompt},
        *st.session_state.messages
    ]
    
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        
        for chunk in ollama.chat(
            model=model,
            messages=messages_for_api,
            stream=True,
            options={"temperature": temperature, "num_predict": max_tokens}
        ):
            full_response += chunk["message"]["content"]
            placeholder.markdown(full_response + "▌")
        
        placeholder.markdown(full_response)
    
    st.session_state.messages.append({
        "role": "assistant", "content": full_response
    })
```

---

## 17. Evaluation & Observability

### 17.1 Agent Evaluation Framework

```python
from dataclasses import dataclass
from typing import Callable

@dataclass
class EvalCase:
    input: str
    expected_output: str | None = None
    expected_contains: list[str] = None
    expected_tool_calls: list[str] = None
    max_iterations: int = 10
    min_score: float = 0.7

@dataclass
class EvalResult:
    case: EvalCase
    actual_output: str
    score: float
    passed: bool
    iterations: int
    latency_ms: int
    details: dict

class AgentEvaluator:
    def __init__(self, agent: ProductionAgent):
        self.agent = agent
        self.results: list[EvalResult] = []
    
    def evaluate_case(self, case: EvalCase) -> EvalResult:
        self.agent.reset()
        result = self.agent.run(case.input)
        
        score = self._score(case, result)
        
        return EvalResult(
            case=case,
            actual_output=result["response"],
            score=score,
            passed=score >= case.min_score,
            iterations=result["iterations"],
            latency_ms=result["latency_ms"],
            details={"tool_calls": result["tool_calls"]},
        )
    
    def _score(self, case: EvalCase, result: dict) -> float:
        scores = []
        output = result["response"].lower()
        
        # Substring containment check
        if case.expected_contains:
            hits = sum(1 for kw in case.expected_contains if kw.lower() in output)
            scores.append(hits / len(case.expected_contains))
        
        # Tool call check
        if case.expected_tool_calls:
            called = {tc["tool"] for tc in result["tool_calls"]}
            expected = set(case.expected_tool_calls)
            scores.append(len(called & expected) / len(expected))
        
        # LLM-as-judge
        if case.expected_output:
            judge_score = self._llm_judge(case.input, case.expected_output, result["response"])
            scores.append(judge_score)
        
        return sum(scores) / len(scores) if scores else 0.5
    
    def _llm_judge(self, question: str, expected: str, actual: str) -> float:
        r = ollama.chat(model="llama3.2", messages=[{
            "role": "user",
            "content": f"""Score how well the actual answer matches the expected answer.
Question: {question}
Expected: {expected}
Actual: {actual}
Score from 0.0 to 1.0. Return ONLY a number:"""
        }], options={"temperature": 0})
        try:
            return float(r["message"]["content"].strip())
        except:
            return 0.5
    
    def run_suite(self, cases: list[EvalCase]) -> dict:
        for case in cases:
            self.results.append(self.evaluate_case(case))
        
        passed = sum(1 for r in self.results if r.passed)
        avg_score = sum(r.score for r in self.results) / len(self.results)
        avg_latency = sum(r.latency_ms for r in self.results) / len(self.results)
        
        return {
            "total": len(cases),
            "passed": passed,
            "pass_rate": passed / len(cases),
            "avg_score": avg_score,
            "avg_latency_ms": avg_latency,
            "results": self.results,
        }
```

### 17.2 Logging & Tracing

```python
import logging, json
from datetime import datetime

# Structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)

class AgentTracer:
    def __init__(self, trace_file: str = "agent_traces.jsonl"):
        self.trace_file = trace_file
    
    def log_run(self, session_id: str, input_: str, output: dict):
        trace = {
            "ts": datetime.now().isoformat(),
            "session_id": session_id,
            "input": input_,
            "output": output["response"],
            "iterations": output["iterations"],
            "latency_ms": output["latency_ms"],
            "tool_calls": output["tool_calls"],
        }
        with open(self.trace_file, "a") as f:
            f.write(json.dumps(trace) + "\n")
    
    def load_traces(self, session_id: str | None = None) -> list[dict]:
        traces = []
        with open(self.trace_file) as f:
            for line in f:
                trace = json.loads(line)
                if session_id is None or trace["session_id"] == session_id:
                    traces.append(trace)
        return traces
```

---

## 18. Best Practices

### 18.1 Model Selection

| Scenario | Recommended Model | Reasoning |
|---|---|---|
| Quick chat / MVP | `llama3.2:3b` | Fast, small, good quality |
| Complex reasoning | `llama3.1:8b` | Strong reasoning, tool use |
| Coding tasks | `qwen2.5-coder:7b` or `codellama:7b` | Trained on code |
| German/EU compliance | `mistral-nemo` | EU-trained, multilingual |
| Routing/classification | `llama3.2:1b` | Ultra-fast for simple tasks |
| Embeddings | `nomic-embed-text` | Best open-source embedder |
| Reasoning chains | `deepseek-r1:7b` | Explicit CoT reasoning |
| Low-resource edge | `llama3.2:1b-q4_0` | <1GB, runs on CPU |

### 18.2 Context Management

```python
# 1. Always size your context window explicitly
options = {"num_ctx": 8192}  # Don't use model default

# 2. Summarize long histories
def compress_history(messages: list[dict], model: str = "llama3.2:1b") -> list[dict]:
    if len(messages) <= 10:
        return messages
    
    # Keep last 5 messages verbatim, summarize the rest
    to_summarize = messages[:-5]
    text = "\n".join([f"{m['role']}: {m['content']}" for m in to_summarize])
    
    r = ollama.chat(model=model, messages=[{
        "role": "user",
        "content": f"Summarize this conversation history in 3-4 sentences:\n{text}"
    }])
    
    summary_msg = {
        "role": "system",
        "content": f"[Previous conversation summary: {r['message']['content']}]"
    }
    return [summary_msg] + messages[-5:]

# 3. Token budget awareness
def estimate_tokens(text: str) -> int:
    return len(text) // 4  # rough estimate

def fits_in_context(messages: list[dict], max_tokens: int = 8192) -> bool:
    total = sum(estimate_tokens(m["content"]) for m in messages)
    return total < max_tokens * 0.8  # 80% safety margin
```

### 18.3 Prompt Best Practices

```python
# ✅ DO: Be explicit about output format
GOOD_PROMPT = """Extract the company name and founding year.
Return ONLY this JSON (no explanation):
{"company": "string", "year": number}

Text: {text}"""

# ❌ DON'T: Leave format ambiguous
BAD_PROMPT = "Find the company and when it was founded: {text}"

# ✅ DO: Use stop tokens to control generation
response = ollama.generate(
    model="llama3.2",
    prompt="Code:\n",
    options={"stop": ["```", "---", "###"]}
)

# ✅ DO: Separate reasoning from output
COT_JSON_PROMPT = """Think through this step by step, then output the answer.

Problem: {problem}

<thinking>
[your step-by-step reasoning]
</thinking>

<answer>
[final answer only]
</answer>"""

# ✅ DO: Constrain with enums when possible
CLASSIFICATION_PROMPT = """Reply with EXACTLY one of: positive, negative, neutral
No other words. Text: {text}"""
```

### 18.4 Error Handling Patterns

```python
from typing import Any
import json

def safe_json_parse(text: str, default: Any = None) -> Any:
    """Try multiple JSON extraction strategies"""
    # Strategy 1: Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Extract from markdown fences
    import re
    matches = re.findall(r"```(?:json)?\s*([\s\S]*?)```", text)
    if matches:
        try:
            return json.loads(matches[0])
        except json.JSONDecodeError:
            pass
    
    # Strategy 3: Find first {...} or [...]
    match = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    
    return default

def with_validation(fn: callable, validator: callable, 
                    max_retries: int = 3) -> callable:
    """Retry a generation until output passes validation"""
    def wrapper(*args, **kwargs):
        for attempt in range(max_retries):
            result = fn(*args, **kwargs)
            if validator(result):
                return result
            if attempt < max_retries - 1:
                kwargs["prompt"] = kwargs.get("prompt", "") + "\nIMPORTANT: Return ONLY valid JSON."
        return result
    return wrapper
```

### 18.5 Security & Privacy

```python
# Input sanitization
INJECTION_PATTERNS = [
    "ignore previous instructions",
    "you are now",
    "disregard your",
    "system prompt:",
    "new instructions:",
]

def sanitize_input(user_input: str) -> tuple[str, bool]:
    lower = user_input.lower()
    flagged = any(p in lower for p in INJECTION_PATTERNS)
    return user_input, flagged

# PII detection before sending to model
import re
PII_PATTERNS = {
    "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    "phone": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
    "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
}

def redact_pii(text: str) -> str:
    for pii_type, pattern in PII_PATTERNS.items():
        text = re.sub(pattern, f"[REDACTED_{pii_type.upper()}]", text)
    return text

# Local-only guarantee
# Ollama runs entirely locally — no data leaves your machine
# No API keys required for base usage
# GDPR-compliant by design for on-premise deployments
```

---

## 19. Method Reference Table

### 19.1 Ollama Python SDK

| Method | Parameters | Returns | Notes |
|---|---|---|---|
| `ollama.generate()` | `model, prompt, system, options, stream` | `dict` | Single-turn generation |
| `ollama.chat()` | `model, messages, tools, format, options, stream` | `dict` | Multi-turn chat |
| `ollama.embed()` | `model, input` | `dict` | Get embeddings |
| `ollama.list()` | — | `dict` | List local models |
| `ollama.show()` | `model` | `dict` | Model metadata |
| `ollama.pull()` | `model, stream` | `dict` | Download model |
| `ollama.push()` | `model, stream` | `dict` | Upload model |
| `ollama.copy()` | `source, destination` | `None` | Copy model |
| `ollama.delete()` | `model` | `None` | Remove model |
| `ollama.ps()` | — | `dict` | Running models |
| `ollama.create()` | `model, modelfile, stream` | `dict` | Build from Modelfile |
| `AsyncClient.chat()` | (same as sync) | `Coroutine` | Async version |
| `AsyncClient.generate()` | (same as sync) | `Coroutine` | Async version |

### 19.2 Generation Options

| Option | Type | Default | Description |
|---|---|---|---|
| `temperature` | float | 0.8 | Sampling randomness (0=det, 1+=creative) |
| `top_p` | float | 0.9 | Nucleus sampling threshold |
| `top_k` | int | 40 | Top-K token sampling |
| `num_ctx` | int | model-dependent | Context window size |
| `num_predict` | int | -1 (unlimited) | Max output tokens |
| `repeat_penalty` | float | 1.1 | Penalize token repetition |
| `repeat_last_n` | int | 64 | Tokens to check for repetition |
| `seed` | int | -1 (random) | Fixed seed for determinism |
| `stop` | list[str] | [] | Stop sequences |
| `tfs_z` | float | 1.0 | Tail-free sampling |
| `mirostat` | int | 0 | Perplexity control (0/1/2) |
| `mirostat_tau` | float | 5.0 | Mirostat target entropy |
| `mirostat_eta` | float | 0.1 | Mirostat learning rate |
| `num_gpu` | int | auto | GPU layers to offload |
| `num_thread` | int | auto | CPU threads |
| `low_vram` | bool | false | Reduce VRAM usage |
| `f16_kv` | bool | true | FP16 key-value cache |

### 19.3 ChromaDB Vector Store Methods

| Method | Parameters | Returns | Notes |
|---|---|---|---|
| `client.create_collection()` | `name, metadata` | `Collection` | Create collection |
| `client.get_collection()` | `name` | `Collection` | Get existing |
| `client.get_or_create_collection()` | `name, metadata` | `Collection` | Upsert collection |
| `client.list_collections()` | — | `list` | All collections |
| `client.delete_collection()` | `name` | `None` | Remove collection |
| `collection.add()` | `ids, embeddings, documents, metadatas` | `None` | Add documents |
| `collection.query()` | `query_embeddings, n_results, where` | `dict` | Similarity search |
| `collection.get()` | `ids, where, include` | `dict` | Fetch by IDs/filter |
| `collection.update()` | `ids, embeddings, documents, metadatas` | `None` | Update documents |
| `collection.upsert()` | `ids, embeddings, documents, metadatas` | `None` | Add or update |
| `collection.delete()` | `ids, where` | `None` | Remove documents |
| `collection.count()` | — | `int` | Number of documents |
| `collection.peek()` | `limit` | `dict` | Sample documents |

### 19.4 MCP Server Decorators

| Decorator | When To Use | Returns |
|---|---|---|
| `@app.list_tools()` | Enumerate available tools | `list[types.Tool]` |
| `@app.call_tool()` | Handle tool execution | `list[TextContent]` |
| `@app.list_resources()` | Enumerate data resources | `list[types.Resource]` |
| `@app.read_resource()` | Serve resource content | `str` |
| `@app.list_prompts()` | Enumerate prompt templates | `list[types.Prompt]` |
| `@app.get_prompt()` | Render prompt template | `GetPromptResult` |

---

## 20. Quick-Reference Cheat Sheet

### 20.1 One-Liners

```bash
# Quick inference
echo "What is 42?" | ollama run llama3.2

# Pipe file through model
cat document.txt | ollama run llama3.2 "Summarize this:"

# Generate code
ollama run codellama:7b "Write a Python quicksort"

# Check what's running  
ollama ps

# Pull latest model list
ollama list
```

```python
# Simplest possible agent (3 lines)
import ollama
response = ollama.chat(model="llama3.2", messages=[{"role":"user","content":"Hello"}])
print(response["message"]["content"])

# Streaming in 4 lines
for chunk in ollama.chat(model="llama3.2", messages=[{"role":"user","content":"Count to 5"}], stream=True):
    print(chunk["message"]["content"], end="", flush=True)

# Embeddings in 2 lines
r = ollama.embed(model="nomic-embed-text", input="Hello world")
vector = r["embeddings"][0]  # 768-dim list

# JSON structured output in 3 lines
r = ollama.chat(model="llama3.2", messages=[{"role":"user","content":"Return JSON: {\"name\":\"?\"}"}], format="json")
data = json.loads(r["message"]["content"])
```

### 20.2 Agent Decision Framework

```
Is the task single-turn and well-defined?
    YES → Use ollama.generate() or ollama.chat() directly
    NO  ↓

Does the task need external data or actions?
    NO  → Use Chain-of-Thought prompting
    YES ↓

Are the tools few and deterministic?
    YES → Use native Ollama tool calling (llama3.1+)
    NO  ↓

Is the task multi-step with unknown number of steps?
    YES → Use ReAct agent with max_iterations guard
    NO  ↓

Does the task need specialized expertise across domains?
    YES → Use Multi-Agent / Orchestrator-Worker pattern
    NO  → Use Plan-and-Execute with single agent
```

### 20.3 Model Parameters Quick-Select

```python
# Deterministic / factual tasks (code, extraction, classification)
FACTUAL = {"temperature": 0.0, "top_p": 1.0, "seed": 42}

# Balanced (general Q&A, summarization)
BALANCED = {"temperature": 0.1, "top_p": 0.9, "repeat_penalty": 1.1}

# Creative (writing, brainstorming, stories)
CREATIVE = {"temperature": 0.8, "top_p": 0.95, "top_k": 50}

# Diverse (multiple ideas, sampling variety)
DIVERSE = {"temperature": 1.2, "top_p": 0.95, "top_k": 100}
```

### 20.4 Environment Variables

```bash
# Ollama config
OLLAMA_HOST=http://localhost:11434      # API endpoint
OLLAMA_MODELS=/path/to/models          # Custom model storage
OLLAMA_NUM_PARALLEL=2                  # Concurrent requests
OLLAMA_MAX_LOADED_MODELS=1             # Models in memory
OLLAMA_KEEP_ALIVE=5m                   # Model unload timeout
CUDA_VISIBLE_DEVICES=0                 # GPU selection

# Set via export (Linux/macOS)
export OLLAMA_HOST=http://0.0.0.0:11434

# systemd override (Linux)
# /etc/systemd/system/ollama.service.d/override.conf
[Service]
Environment="OLLAMA_HOST=0.0.0.0"
Environment="OLLAMA_NUM_PARALLEL=4"
```

### 20.5 Requirements Files

```txt
# requirements.txt — Core agent stack
ollama>=0.3.0
chromadb>=0.5.0
pydantic>=2.0.0
fastapi>=0.110.0
uvicorn[standard]>=0.29.0
httpx>=0.27.0
mcp>=1.0.0
python-dotenv>=1.0.0

# Optional
streamlit>=1.35.0          # UI
requests>=2.31.0           # HTTP tools
pypdf>=4.0.0               # PDF ingestion
python-docx>=1.1.0         # DOCX ingestion
sentence-transformers>=3.0 # Alternative embeddings
```

```toml
# pyproject.toml
[tool.poetry.dependencies]
python = "^3.11"
ollama = "^0.3"
chromadb = "^0.5"
pydantic = "^2"
fastapi = "^0.110"
mcp = "^1.0"
```

---

## Appendix: Model Format & Token Special IDs

### LLaMA 3.x Special Tokens

| Token | ID | Role |
|---|---|---|
| `<\|begin_of_text\|>` | 128000 | Start of document |
| `<\|end_of_text\|>` | 128001 | End of document |
| `<\|start_header_id\|>` | 128006 | Start of role header |
| `<\|end_header_id\|>` | 128007 | End of role header |
| `<\|eot_id\|>` | 128009 | End of turn |
| `<\|python_tag\|>` | 128010 | Python code block |

### Performance Benchmarks (Approximate, M2 Pro / RTX 3080)

| Model | Tokens/s (CPU) | Tokens/s (GPU) | TTFT (ms) |
|---|---|---|---|
| llama3.2:1b-q4 | ~35 | ~120 | ~200 |
| llama3.2:3b-q4 | ~18 | ~80 | ~350 |
| llama3.1:8b-q4 | ~8 | ~50 | ~700 |
| llama3.1:70b-q4 | ~1.5 | ~12 | ~3000 |
| mistral:7b-q4 | ~9 | ~55 | ~600 |

> **TTFT** = Time-to-first-token · CPU benchmarks on 8-core Intel i7 · GPU on RTX 3080 (10GB VRAM)

---

*Reference card covers: Ollama ≥0.3 · llama3.2 · llama3.1 · MCP ≥1.0 · ChromaDB ≥0.5 · Python ≥3.11*
*Last updated: June 2026*
