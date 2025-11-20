# Deep Researcher

## Overview

Deep Researcher is a comprehensive research tool that integrates with the Brave Search API to perform multi-phase research on any topic. It uses a local LLM (LFM2-1.2B-Tool-Q4_K_M-cuda) to orchestrate research tasks through multiple phases of investigation, validation, and synthesis.

## Key Features

- **Real API Integration**: Uses the real Brave Search API (not mocks) with a provided API key
- **Multi-Phase Research**: Executes research in 4 distinct phases (Exploration, Deep Investigation, Cross-Validation, Synthesis)
- **Chain Depth Management**: Implements conservative chain depth limits to prevent exceeding model capabilities
- **Fact Extraction & Verification**: Extracts key facts from search results and cross-references them across multiple sources
- **Angle Identification**: Identifies key research angles and subtopics to investigate further
- **Source Tracking**: Maintains and reports on all sources used during research
- **Interactive Interface**: Supports both command-line and interactive research modes

## Architecture

### Core Components

- `BraveSearchAPI`: Handles real Brave Search API requests and response formatting
- `ResearchState`: Manages research state across multiple phases
- `ResearchToolExecutor`: Executes research-specific tools with real API calls
- `DeepResearchOrchestrator`: Coordinates the multi-phase research process

### Research Tools

The system implements these research-specific tools:
- `brave_search`: Performs web searches using the Brave API
- `extract_key_facts`: Extracts key facts and themes from search results
- `identify_angles`: Identifies key research angles/subtopics to investigate
- `cross_reference`: Verifies if facts appear in multiple sources
- `generate_follow_up`: Generates follow-up search queries

### Research Phases

1. **Exploration Phase**: Performs broad search to get an overview of the topic
2. **Deep Investigation Phase**: Targets specific angles identified in phase 1
3. **Cross-Validation Phase**: Verifies key findings across multiple sources
4. **Synthesis Phase**: Creates a comprehensive research report

## Configuration

- `BRAVE_API_KEY`: API key for Brave Search (pre-configured)
- `LLM_BASE_URL`: Local LLM endpoint (default: "http://localhost:8080/v1")
- `MODEL_NAME`: Model to use for research (default: "LFM2-1.2B-Tool-Q4_K_M-cuda")
- `MAX_CHAIN_DEPTH`: Maximum tool chain depth (default: 6)
- `MAX_PHASES`: Maximum research phases (default: 4)

## Usage

The tool supports two main commands:
- `research <topic>`: Start quick research (2 phases)
- `deep-research <topic>`: Start thorough research (3 phases)

Example usage:
```
> research artificial intelligence
> deep-research quantum computing
```

The tool will output:
- Research phases execution
- Search results and findings
- Cross-validation of key facts
- Final synthesized report
- Top sources used
- Full results saved to a timestamped JSON file

## Output

Results include:
- Comprehensive research report
- List of sources used
- Key findings and facts
- Identified research angles
- Number of searches executed
- Time elapsed for research

## Limitations

- Requires a valid Brave Search API key to function
- Interactive command-line interface (not batch processing)
- Chain depth limits may prevent very complex research
- Results depend on Brave Search API availability and quality
- Local LLM must be running at the specified endpoint

## Dependencies

- `requests` for API communication
- `json` for data serialization
- `typing` for type hints
- `datetime` for timestamping
- `re` for pattern matching