"""
Component 3: LLM Layer for FPL Graph-RAG
Combines KG results and generates responses using different LLMs
"""

import requests
import json
from typing import Dict, List, Any
import time
from huggingface_hub import InferenceClient


class FPLLLMLayer:
    """
    Handles LLM interaction for generating final responses
    """
    
    def __init__(self, openai_key: str = None, hf_key: str = None):
        """
        Initialize LLM layer with API keys
        
        Args:
            openai_key: OpenAI API key (optional)
            hf_key: HuggingFace API key (optional)
        """
        self.openai_key = openai_key
        self.hf_key = hf_key
        
        # Initialize HuggingFace client if token provided
        self.hf_client = None
        if hf_key:
            try:
                self.hf_client = InferenceClient(token=hf_key)
            except Exception as e:
                print(f"Warning: Could not initialize HuggingFace client: {e}")
        
        # Available models
        self.models = {
            "gpt-3.5-turbo": self._call_openai,
            "mistral-7b": self._call_huggingface_inference,
            "gemma-2b": self._call_huggingface_inference,
            "llama-2-7b": self._call_huggingface_inference,
        }
    
    def format_context(self, baseline_results: Dict, embedding_results: Dict = None) -> str:
        """
        Format KG retrieval results into context for LLM
        
        Args:
            baseline_results: Results from Cypher queries
            embedding_results: Results from embedding retrieval (optional)
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        # Add baseline results
        if baseline_results and baseline_results.get("data"):
            context_parts.append("=== Cypher Query Results ===")
            
            for item in baseline_results["data"][:5]:  # Limit to top 5
                item_str = ", ".join([f"{k}: {v}" for k, v in item.items()])
                context_parts.append(f"- {item_str}")
        
        # Add embedding results
        if embedding_results and embedding_results.get("data"):
            context_parts.append("\n=== Similar Players (Semantic Search) ===")
            
            for item in embedding_results["data"][:5]:
                item_copy = item.copy()
                similarity = item_copy.pop("similarity", 0)
                item_str = ", ".join([f"{k}: {v}" for k, v in item_copy.items()])
                context_parts.append(f"- {item_str} (similarity: {similarity:.3f})")
        
        return "\n".join(context_parts)
    
    def create_prompt(self, query: str, context: str, persona: str = None) -> str:
        """
        Create structured prompt with context, persona, and task
        
        Args:
            query: User query
            context: Formatted KG context
            persona: LLM persona (optional)
            
        Returns:
            Complete prompt
        """
        if persona is None:
            persona = """You are an expert Fantasy Premier League (FPL) assistant. 
You provide accurate, data-driven advice based on player statistics and performance.
You only use information from the provided context and never make up statistics."""
        
        prompt = f"""### PERSONA
{persona}

### CONTEXT (FPL Knowledge Graph Data)
{context}

### TASK
Answer the following user question using ONLY the information provided in the context above.
If the context doesn't contain enough information to answer the question, say so.
Be concise and specific, citing relevant statistics from the context.

### USER QUESTION
{query}

### ANSWER
"""
        return prompt
    
    def _call_openai(self, prompt: str, model: str = "gpt-3.5-turbo") -> Dict[str, Any]:
        """
        Call OpenAI API
        
        Args:
            prompt: Complete prompt
            model: Model name
            
        Returns:
            Response dictionary with answer and metadata
        """
        if not self.openai_key:
            return {
                "answer": "OpenAI API key not configured",
                "model": model,
                "error": True
            }
        
        try:
            import openai
            openai.api_key = self.openai_key
            
            start_time = time.time()
            
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an FPL expert assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            end_time = time.time()
            
            return {
                "answer": response.choices[0].message.content,
                "model": model,
                "response_time": end_time - start_time,
                "tokens": response.usage.total_tokens,
                "error": False
            }
        
        except Exception as e:
            return {
                "answer": f"Error calling OpenAI: {str(e)}",
                "model": model,
                "error": True
            }
    
    def _call_huggingface(self, prompt: str, model: str = "mistralai/Mistral-7B-Instruct-v0.1") -> Dict[str, Any]:
        """
        Call HuggingFace Inference API
        
        Args:
            prompt: Complete prompt
            model: Model name
            
        Returns:
            Response dictionary with answer and metadata
        """
        if not self.hf_key:
            return {
                "answer": "HuggingFace API key not configured",
                "model": model,
                "error": True
            }
        
        try:
            api_url = f"https://api-inference.huggingface.co/models/{model}"
            headers = {"Authorization": f"Bearer {self.hf_key}"}
            
            start_time = time.time()
            
            response = requests.post(
                api_url,
                headers=headers,
                json={"inputs": prompt, "parameters": {"max_new_tokens": 500, "temperature": 0.3}}
            )
            
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                answer = result[0]["generated_text"] if isinstance(result, list) else result.get("generated_text", "")
                
                # Extract answer after the prompt
                if "### ANSWER" in answer:
                    answer = answer.split("### ANSWER")[-1].strip()
                
                return {
                    "answer": answer,
                    "model": model,
                    "response_time": end_time - start_time,
                    "error": False
                }
            else:
                return {
                    "answer": f"HuggingFace API error: {response.status_code}",
                    "model": model,
                    "error": True
                }
        
        except Exception as e:
            return {
                "answer": f"Error calling HuggingFace: {str(e)}",
                "model": model,
                "error": True
            }
    
    def _call_huggingface_inference(self, prompt: str, model: str = "mistralai/Mistral-7B-Instruct-v0.2") -> Dict[str, Any]:
        """
        Call HuggingFace Inference API using InferenceClient
        
        Args:
            prompt: Complete prompt
            model: Model name (mistral-7b, gemma-2b, or llama-2-7b)
            
        Returns:
            Response dictionary with answer and metadata
        """
        if not self.hf_client:
            return {
                "answer": "HuggingFace client not initialized. Please provide HF_TOKEN.",
                "model": model,
                "error": True
            }
        
        # Map simple names to full model paths
        model_map = {
            "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.2",
            "gemma-2b": "google/gemma-2-2b-it",
            "llama-2-7b": "meta-llama/Llama-2-7b-chat-hf"
        }
        
        full_model = model_map.get(model, model)
        
        try:
            start_time = time.time()
            
            # Use chat completion for instruction-tuned models
            response = self.hf_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                model=full_model,
                max_tokens=500,
                temperature=0.3,
            )
            
            end_time = time.time()
            
            answer = response.choices[0].message.content
            
            # Extract answer after the ### ANSWER marker if present
            if "### ANSWER" in answer:
                answer = answer.split("### ANSWER")[-1].strip()
            
            return {
                "answer": answer,
                "model": full_model,
                "response_time": end_time - start_time,
                "error": False
            }
        
        except Exception as e:
            return {
                "answer": f"Error calling HuggingFace Inference API: {str(e)}",
                "model": full_model,
                "error": True
            }
    
    def _call_local_model(self, prompt: str) -> Dict[str, Any]:
        """
        Fallback: Simple rule-based response (for demo purposes)
        
        Args:
            prompt: Complete prompt
            
        Returns:
            Response dictionary
        """
        # Extract context and question
        if "### CONTEXT" in prompt and "### USER QUESTION" in prompt:
            context_part = prompt.split("### CONTEXT")[1].split("### TASK")[0]
            
            # Simple extraction of first few results
            lines = [line.strip() for line in context_part.split('\n') if line.strip().startswith('-')]
            
            answer = "Based on the FPL data:\n\n"
            for line in lines[:3]:
                answer += line + "\n"
            
            return {
                "answer": answer,
                "model": "rule-based-fallback",
                "response_time": 0.1,
                "error": False
            }
        
        return {
            "answer": "Unable to process query with local model",
            "model": "rule-based-fallback",
            "error": True
        }
    
    def generate_response(self, 
                         query: str, 
                         baseline_results: Dict, 
                         embedding_results: Dict = None,
                         model_name: str = "gpt-3.5-turbo",
                         persona: str = None) -> Dict[str, Any]:
        """
        Generate final response using specified LLM
        
        Args:
            query: User query
            baseline_results: Cypher query results
            embedding_results: Embedding retrieval results
            model_name: Name of LLM to use
            persona: Optional custom persona
            
        Returns:
            Complete response with answer and metadata
        """
        # Format context
        context = self.format_context(baseline_results, embedding_results)
        
        # Create prompt
        prompt = self.create_prompt(query, context, persona)
        
        # Call appropriate LLM
        if model_name == "gpt-3.5-turbo" and self.openai_key:
            response = self._call_openai(prompt, model_name)
        elif model_name in ["mistral-7b", "gemma-2b", "llama-2-7b"] and self.hf_client:
            response = self._call_huggingface_inference(prompt, model_name)
        else:
            # Fallback to simple rule-based
            response = self._call_local_model(prompt)
        
        # Add query and context to response
        response["query"] = query
        response["context"] = context
        response["prompt"] = prompt
        
        return response
    
    def compare_models(self, 
                      query: str, 
                      baseline_results: Dict,
                      embedding_results: Dict = None,
                      models: List[str] = None) -> Dict[str, Dict]:
        """
        Compare responses from multiple LLMs
        
        Args:
            query: User query
            baseline_results: Cypher results
            embedding_results: Embedding results
            models: List of model names to compare
            
        Returns:
            Dictionary mapping model names to responses
        """
        if models is None:
            models = ["gpt-3.5-turbo", "mistral-7b", "rule-based-fallback"]
        
        responses = {}
        
        for model in models:
            print(f"Calling {model}...")
            response = self.generate_response(
                query, baseline_results, embedding_results, model
            )
            responses[model] = response
            
            # Add small delay to avoid rate limits
            time.sleep(1)
        
        return responses


# Example usage
if __name__ == "__main__":
    llm_layer = FPLLLMLayer()
    
    # Mock data for testing
    mock_baseline = {
        "method": "baseline",
        "data": [
            {"name": "Erling Haaland", "position": "FWD", "team": "Man City", "points": 196, "price": 14.0},
            {"name": "Harry Kane", "position": "FWD", "team": "Tottenham", "points": 178, "price": 11.5}
        ]
    }
    
    mock_embedding = {
        "method": "embedding",
        "data": [
            {"name": "Mohamed Salah", "position": "MID", "team": "Liverpool", "points": 188, "similarity": 0.85}
        ]
    }
    
    # Test prompt creation
    context = llm_layer.format_context(mock_baseline, mock_embedding)
    prompt = llm_layer.create_prompt("Who are the top forwards?", context)
    
    print("Generated Prompt:")
    print("=" * 60)
    print(prompt)
