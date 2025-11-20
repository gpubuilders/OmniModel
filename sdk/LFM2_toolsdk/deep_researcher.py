"""
Deep Researcher - Brave Search API Integration
Fully working implementation with no mocks
Handles 8-step chain limit with multi-phase approach
"""

import requests
import json
import time
from typing import List, Dict, Optional
from datetime import datetime
import re

# ============================================================
# CONFIGURATION
# ============================================================
BRAVE_API_KEY = "BSAYiuqbDL8nsXhDb8e-bxVk1GPrR92"
BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"

LLM_BASE_URL = "http://localhost:8080/v1"
MODEL_NAME = "LFM2-1.2B-Tool-Q4_K_M-cuda"

# Research limits
MAX_CHAIN_DEPTH = 6  # Conservative limit
MAX_SEARCH_RESULTS_PER_QUERY = 8
MAX_PHASES = 4


# ============================================================
# BRAVE SEARCH API
# ============================================================
class BraveSearchAPI:
    """Real Brave Search API integration"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": api_key
        }
    
    def search(self, query: str, count: int = 10, freshness: Optional[str] = None) -> Dict:
        """
        Execute web search
        freshness: None, 'pd' (past day), 'pw' (past week), 'pm' (past month), 'py' (past year)
        """
        params = {
            "q": query,
            "count": min(count, 20)  # Brave max is 20
        }
        
        if freshness:
            params["freshness"] = freshness
        
        try:
            response = requests.get(
                BRAVE_SEARCH_URL,
                headers=self.headers,
                params=params,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.RequestException as e:
            print(f"✗ Brave API Error: {str(e)}")
            return {"web": {"results": []}}
    
    def format_results(self, search_response: Dict) -> List[Dict]:
        """Format Brave results into clean structure"""
        results = []
        
        web_results = search_response.get("web", {}).get("results", [])
        
        for idx, result in enumerate(web_results):
            results.append({
                "id": idx,
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "description": result.get("description", ""),
                "age": result.get("age", "")
            })
        
        return results


# ============================================================
# LLM API
# ============================================================
def call_llm(messages: List[Dict], tools: Optional[List[Dict]] = None, 
             temperature: float = 0.1) -> str:
    """Call local LLM API"""
    
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 1024
    }
    
    if tools:
        payload["tools"] = tools
    
    try:
        response = requests.post(
            f"{LLM_BASE_URL}/chat/completions",
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        
        if "error" in data:
            raise Exception(f"LLM Error: {data['error']['message']}")
        
        return data["choices"][0]["message"]["content"]
    
    except requests.exceptions.RequestException as e:
        raise Exception(f"LLM API Error: {str(e)}")


def extract_tool_calls(content: str) -> List[str]:
    """Extract tool calls from LLM response"""
    if "<|tool_call_start|>" not in content:
        return []
    
    start = content.find("<|tool_call_start|>") + len("<|tool_call_start|>")
    end = content.find("<|tool_call_end|>")
    
    if end == -1:
        return []
    
    calls_str = content[start:end].strip().strip("[]")
    if not calls_str:
        return []
    
    return re.findall(r'\w+\([^)]*\)', calls_str)


# ============================================================
# RESEARCH STATE MANAGER
# ============================================================
class ResearchState:
    """External state management for multi-phase research"""
    
    def __init__(self, topic: str):
        self.topic = topic
        self.phases_completed = 0
        self.findings = []
        self.sources = []
        self.key_angles = []
        self.searches_executed = []
        self.current_phase_data = {}
    
    def add_finding(self, finding: Dict):
        """Add a research finding"""
        self.findings.append({
            **finding,
            "timestamp": datetime.now().isoformat(),
            "phase": self.phases_completed
        })
    
    def add_sources(self, sources: List[Dict]):
        """Add sources"""
        for source in sources:
            if source['url'] not in [s['url'] for s in self.sources]:
                self.sources.append(source)
    
    def get_summary(self) -> str:
        """Get current state summary"""
        return json.dumps({
            "topic": self.topic,
            "phases_completed": self.phases_completed,
            "findings_count": len(self.findings),
            "sources_count": len(self.sources),
            "key_angles": self.key_angles,
            "searches_executed": len(self.searches_executed)
        }, indent=2)
    
    def complete_phase(self):
        """Mark phase as complete"""
        self.phases_completed += 1
        self.current_phase_data = {}


# ============================================================
# TOOL EXECUTOR
# ============================================================
class ResearchToolExecutor:
    """Execute research tools with real Brave API"""
    
    def __init__(self, brave_api: BraveSearchAPI, research_state: ResearchState):
        self.brave = brave_api
        self.state = research_state
    
    def execute(self, tool_call: str) -> str:
        """Execute a tool call"""
        
        # Parse tool name and params
        tool_name = tool_call.split("(")[0]
        params = self._extract_params(tool_call)
        
        print(f"  → Executing: {tool_name}")
        print(f"    Params: {params}")
        
        # Route to appropriate handler
        handlers = {
            "brave_search": self._brave_search,
            "extract_key_facts": self._extract_key_facts,
            "identify_angles": self._identify_angles,
            "cross_reference": self._cross_reference,
            "generate_follow_up": self._generate_follow_up,
        }
        
        handler = handlers.get(tool_name)
        if not handler:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})
        
        result = handler(params)
        
        # Track search
        if tool_name == "brave_search":
            self.state.searches_executed.append({
                "query": params.get("query", ""),
                "timestamp": datetime.now().isoformat(),
                "results_count": len(result.get("results", []))
            })
        
        return json.dumps(result)
    
    def _extract_params(self, tool_call: str) -> Dict:
        """Extract parameters from tool call string"""
        params = {}
        
        # Simple regex-based extraction
        param_pattern = r'(\w+)=(?:"([^"]*)"|\'([^\']*)\'|(\d+))'
        matches = re.findall(param_pattern, tool_call)
        
        for match in matches:
            key = match[0]
            value = match[1] or match[2] or match[3]
            
            # Try to convert to int if it's a number
            try:
                value = int(value)
            except (ValueError, TypeError):
                pass
            
            params[key] = value
        
        return params
    
    def _brave_search(self, params: Dict) -> Dict:
        """Execute Brave search"""
        query = params.get("query", "")
        count = params.get("count", MAX_SEARCH_RESULTS_PER_QUERY)
        freshness = params.get("freshness")
        
        print(f"    Searching Brave: '{query}' (count={count})")
        
        raw_results = self.brave.search(query, count, freshness)
        formatted_results = self.brave.format_results(raw_results)
        
        # Add to state
        self.state.add_sources(formatted_results)
        
        print(f"    ✓ Found {len(formatted_results)} results")
        
        return {
            "query": query,
            "results": formatted_results,
            "count": len(formatted_results)
        }
    
    def _extract_key_facts(self, params: Dict) -> Dict:
        """Extract key facts from search results"""
        result_ids = params.get("result_ids", [])
        
        # Get results from state
        results = [s for s in self.state.sources if s['id'] in result_ids]
        
        # Simple extraction - in real system, this could use LLM
        facts = []
        themes = set()
        
        for result in results:
            # Extract sentences from description
            desc = result.get("description", "")
            sentences = [s.strip() for s in desc.split(".") if len(s.strip()) > 20]
            
            for sentence in sentences[:2]:  # Top 2 sentences per result
                facts.append({
                    "fact": sentence,
                    "source_url": result['url'],
                    "source_title": result['title']
                })
            
            # Simple theme extraction from title
            title_words = result['title'].lower().split()
            themes.update([w for w in title_words if len(w) > 5])
        
        return {
            "facts": facts[:10],  # Top 10 facts
            "themes": list(themes)[:5],  # Top 5 themes
            "sources_analyzed": len(results)
        }
    
    def _identify_angles(self, params: Dict) -> Dict:
        """Identify research angles from current findings"""
        context = params.get("context", "")
        max_angles = params.get("max_angles", 3)
        
        # Analyze current sources for angles
        all_titles = [s['title'] for s in self.state.sources]
        all_descriptions = [s['description'] for s in self.state.sources]
        
        # Simple keyword extraction
        words = {}
        for text in all_titles + all_descriptions:
            for word in text.lower().split():
                if len(word) > 5 and word.isalpha():
                    words[word] = words.get(word, 0) + 1
        
        # Top keywords become angles
        top_keywords = sorted(words.items(), key=lambda x: x[1], reverse=True)[:max_angles]
        angles = [kw[0] for kw in top_keywords]
        
        self.state.key_angles = angles
        
        return {
            "angles": angles,
            "rationale": "Based on frequency analysis of search results"
        }
    
    def _cross_reference(self, params: Dict) -> Dict:
        """Cross-reference a fact across sources"""
        fact = params.get("fact", "")
        min_sources = params.get("min_sources", 2)
        
        # Check how many sources mention similar content
        mentions = 0
        mentioning_sources = []
        
        for source in self.state.sources:
            text = (source['title'] + " " + source['description']).lower()
            
            # Simple substring check (could be more sophisticated)
            fact_words = set(fact.lower().split())
            if len(fact_words) > 0:
                text_words = set(text.split())
                overlap = len(fact_words & text_words) / len(fact_words)
                
                if overlap > 0.5:  # 50% word overlap
                    mentions += 1
                    mentioning_sources.append({
                        "url": source['url'],
                        "title": source['title']
                    })
        
        verified = mentions >= min_sources
        
        return {
            "fact": fact,
            "verified": verified,
            "mentions": mentions,
            "required_mentions": min_sources,
            "sources": mentioning_sources[:5]
        }
    
    def _generate_follow_up(self, params: Dict) -> Dict:
        """Generate follow-up search queries"""
        context = params.get("context", "")
        max_queries = params.get("max_queries", 3)
        
        # Based on current angles and gaps
        follow_ups = []
        
        for angle in self.state.key_angles[:max_queries]:
            follow_ups.append(f"{self.state.topic} {angle}")
        
        # Add some specific queries
        if len(follow_ups) < max_queries:
            follow_ups.append(f"{self.state.topic} recent developments 2024")
        
        if len(follow_ups) < max_queries:
            follow_ups.append(f"{self.state.topic} research studies")
        
        return {
            "follow_up_queries": follow_ups[:max_queries],
            "rationale": "Based on identified angles and research gaps"
        }


# ============================================================
# TOOL DEFINITIONS
# ============================================================
RESEARCH_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "brave_search",
            "description": "Search the web using Brave Search API. Returns real-time web results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "count": {
                        "type": "integer",
                        "description": "Number of results (1-20)",
                        "default": 8
                    },
                    "freshness": {
                        "type": "string",
                        "description": "Time filter: 'pd' (past day), 'pw' (past week), 'pm' (past month)",
                        "enum": ["pd", "pw", "pm", "py"]
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "extract_key_facts",
            "description": "Extract key facts and themes from search results",
            "parameters": {
                "type": "object",
                "properties": {
                    "result_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "List of result IDs to analyze (from previous search)"
                    }
                },
                "required": ["result_ids"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "identify_angles",
            "description": "Identify key research angles/subtopics to investigate",
            "parameters": {
                "type": "object",
                "properties": {
                    "context": {
                        "type": "string",
                        "description": "Current research context"
                    },
                    "max_angles": {
                        "type": "integer",
                        "description": "Maximum number of angles to identify",
                        "default": 3
                    }
                },
                "required": ["context"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "cross_reference",
            "description": "Verify if a fact appears in multiple sources",
            "parameters": {
                "type": "object",
                "properties": {
                    "fact": {
                        "type": "string",
                        "description": "Fact to verify"
                    },
                    "min_sources": {
                        "type": "integer",
                        "description": "Minimum number of sources required for verification",
                        "default": 2
                    }
                },
                "required": ["fact"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_follow_up",
            "description": "Generate follow-up search queries based on current findings",
            "parameters": {
                "type": "object",
                "properties": {
                    "context": {
                        "type": "string",
                        "description": "Current research context"
                    },
                    "max_queries": {
                        "type": "integer",
                        "description": "Maximum number of follow-up queries",
                        "default": 3
                    }
                },
                "required": ["context"]
            }
        }
    }
]


# ============================================================
# RESEARCH ORCHESTRATOR
# ============================================================
class DeepResearchOrchestrator:
    """Multi-phase research orchestrator"""
    
    def __init__(self, brave_api_key: str):
        self.brave = BraveSearchAPI(brave_api_key)
        self.state = None
        self.conversation = []
        self.chain_depth = 0
    
    def research(self, topic: str, max_depth: int = 3) -> Dict:
        """
        Execute deep research on topic
        max_depth: 1=quick, 2=moderate, 3=deep
        """
        
        print("\n" + "="*70)
        print(f"DEEP RESEARCH: {topic}")
        print("="*70)
        
        # Initialize state
        self.state = ResearchState(topic)
        self.executor = ResearchToolExecutor(self.brave, self.state)
        
        # Phase 1: Exploration
        print(f"\n📍 PHASE 1: EXPLORATION")
        print("-" * 70)
        self._phase_1_exploration(topic)
        
        # Phase 2: Deep Dives
        if max_depth >= 2:
            print(f"\n📍 PHASE 2: DEEP INVESTIGATION")
            print("-" * 70)
            self._phase_2_investigation()
        
        # Phase 3: Cross-validation
        if max_depth >= 3:
            print(f"\n📍 PHASE 3: CROSS-VALIDATION")
            print("-" * 70)
            self._phase_3_validation()
        
        # Phase 4: Synthesis
        print(f"\n📍 PHASE 4: SYNTHESIS")
        print("-" * 70)
        report = self._phase_4_synthesis()
        
        return {
            "topic": topic,
            "report": report,
            "sources": self.state.sources,
            "findings": self.state.findings,
            "searches_executed": len(self.state.searches_executed),
            "phases_completed": self.state.phases_completed
        }
    
    def _phase_1_exploration(self, topic: str):
        """Phase 1: Initial broad search"""
        
        self._reset_chain()
        
        prompt = f"""You are a research assistant. Your task is to explore the topic: "{topic}"

Step 1: Perform a broad web search to get an overview
Step 2: Extract key facts from the top results
Step 3: Identify 2-3 main angles to investigate further

Use the available tools to accomplish this."""
        
        self._execute_research_chain(prompt, max_steps=5)
        self.state.complete_phase()
    
    def _phase_2_investigation(self):
        """Phase 2: Deep dive on key angles"""
        
        if not self.state.key_angles:
            print("  ⚠️  No angles identified, skipping deep investigation")
            return
        
        # Investigate each angle (limit to 2 to stay within limits)
        for angle in self.state.key_angles[:2]:
            print(f"\n  🔍 Investigating angle: {angle}")
            
            self._reset_chain()
            
            prompt = f"""Continue researching "{self.state.topic}".

Previous context: We've identified "{angle}" as a key angle to investigate.

Task: Perform targeted searches on this angle and extract detailed findings.
Use cross_reference to verify important facts."""
            
            self._execute_research_chain(prompt, max_steps=5)
        
        self.state.complete_phase()
    
    def _phase_3_validation(self):
        """Phase 3: Cross-validate findings"""
        
        self._reset_chain()
        
        # Get top facts to validate
        top_facts = []
        for finding in self.state.findings[:3]:
            if 'facts' in finding:
                top_facts.extend(finding['facts'][:2])
        
        if not top_facts:
            print("  ⚠️  No facts to validate")
            return
        
        prompt = f"""Validate key findings for research on "{self.state.topic}".

Previous research has identified these facts:
{json.dumps(top_facts[:3], indent=2)}

Task: Use cross_reference to verify these facts appear in multiple sources.
If verification fails, perform additional searches to find better sources."""
        
        self._execute_research_chain(prompt, max_steps=5)
        self.state.complete_phase()
    
    def _phase_4_synthesis(self) -> str:
        """Phase 4: Synthesize final report"""
        
        self._reset_chain()
        
        # Create comprehensive context
        context = {
            "topic": self.state.topic,
            "total_sources": len(self.state.sources),
            "findings_summary": self.state.findings[:10],  # Top 10
            "key_angles": self.state.key_angles,
            "searches_executed": len(self.state.searches_executed)
        }
        
        prompt = f"""Synthesize a final research report on "{self.state.topic}".

Research Summary:
{json.dumps(context, indent=2)}

Available sources: {len(self.state.sources)} web sources

Task: Create a comprehensive summary that:
1. Provides an overview of the topic
2. Discusses key findings
3. Mentions important angles/subtopics
4. Cites specific sources where relevant

Format as a clear, informative report."""
        
        self.conversation.append({"role": "user", "content": prompt})
        
        # Get synthesis without tools
        response = call_llm(self.conversation, temperature=0.3)
        
        self.state.complete_phase()
        
        return response
    
    def _execute_research_chain(self, prompt: str, max_steps: int = 5):
        """Execute a research chain with tool calls"""
        
        self.conversation.append({"role": "user", "content": prompt})
        
        for step in range(max_steps):
            self.chain_depth += 1
            
            if self.chain_depth > MAX_CHAIN_DEPTH:
                print(f"  ⚠️  Chain depth limit reached ({MAX_CHAIN_DEPTH})")
                break
            
            print(f"\n  Step {step + 1}/{max_steps} (chain depth: {self.chain_depth})")
            
            # Get LLM response
            response = call_llm(self.conversation, RESEARCH_TOOLS)
            
            # Check for tool calls
            tool_calls = extract_tool_calls(response)
            
            if not tool_calls:
                # No more tools needed
                print(f"  ✓ Phase complete (no more tool calls)")
                self.conversation.append({"role": "assistant", "content": response})
                break
            
            # Execute tools
            self.conversation.append({"role": "assistant", "content": response})
            
            for tool_call in tool_calls:
                result = self.executor.execute(tool_call)
                self.conversation.append({"role": "tool", "content": result})
                
                # Store findings
                try:
                    result_data = json.loads(result)
                    self.state.add_finding(result_data)
                except:
                    pass
    
    def _reset_chain(self):
        """Reset conversation chain"""
        self.conversation = []
        self.chain_depth = 0
        print(f"  🔄 Chain reset")


# ============================================================
# MAIN INTERFACE
# ============================================================
def main():
    """Interactive research interface"""
    
    print("\n" + "="*70)
    print("DEEP RESEARCHER - Brave Search API")
    print("="*70)
    print("Commands:")
    print("  research <topic>        - Start research (quick)")
    print("  deep-research <topic>   - Deep research (thorough)")
    print("  quit                    - Exit")
    print("="*70)
    
    orchestrator = DeepResearchOrchestrator(BRAVE_API_KEY)
    
    while True:
        try:
            user_input = input("\n> ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == 'quit':
                print("Goodbye!")
                break
            
            # Parse command
            if user_input.lower().startswith('research '):
                topic = user_input[9:].strip()
                depth = 2
            elif user_input.lower().startswith('deep-research '):
                topic = user_input[14:].strip()
                depth = 3
            else:
                topic = user_input
                depth = 2
            
            if not topic:
                print("Please provide a research topic")
                continue
            
            # Execute research
            start_time = time.time()
            result = orchestrator.research(topic, max_depth=depth)
            elapsed = time.time() - start_time
            
            # Display results
            print("\n" + "="*70)
            print("RESEARCH RESULTS")
            print("="*70)
            print(f"\nTopic: {result['topic']}")
            print(f"Searches executed: {result['searches_executed']}")
            print(f"Sources found: {len(result['sources'])}")
            print(f"Time elapsed: {elapsed:.1f}s")
            print(f"\n{'-'*70}")
            print("REPORT:")
            print(f"{'-'*70}")
            print(result['report'])
            print(f"\n{'-'*70}")
            print(f"Top {min(5, len(result['sources']))} Sources:")
            print(f"{'-'*70}")
            for i, source in enumerate(result['sources'][:5], 1):
                print(f"\n{i}. {source['title']}")
                print(f"   {source['url']}")
            
            # Save to file
            filename = f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\n✓ Full results saved to: {filename}")
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Type 'quit' to exit.")
        except Exception as e:
            print(f"\n✗ Error: {str(e)}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()