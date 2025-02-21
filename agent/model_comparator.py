"""
Module for comparing responses from different Ollama models.
Implements concurrent model execution and response analysis.
"""
import asyncio
import aiohttp
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from .code_executor import execute_code_async, extract_code_blocks
from .config import LLM_API_URL
from .db import ConversationDB

logger = logging.getLogger(__name__)

async def get_available_models(session: aiohttp.ClientSession) -> List[str]:
    """
    Fetches the list of available models from the Ollama API.
    Uses exponential backoff for retries.
    """
    max_retries = 3
    base_wait = 1
    
    for attempt in range(max_retries):
        try:
            # Use the correct Ollama API endpoint for listing models
            api_url = LLM_API_URL.rstrip("/api/chat")  # Remove /api/chat if present
            async with session.get(f"{api_url}/api/tags") as resp:
                if resp.status == 404:
                    # Try alternative endpoint if first one fails
                    async with session.get(f"{api_url}/api/models") as alt_resp:
                        if alt_resp.status == 404:
                            raise Exception("Neither /api/tags nor /api/models endpoints available")
                        alt_resp.raise_for_status()
                        data = await alt_resp.json()
                        # Extract model names from the response
                        models = [model.get('name', model) for model in data.get('models', [])]
                else:
                    resp.raise_for_status()
                    data = await resp.json()
                    # Extract model names from the response
                    models = [model.get('name', model) for model in data.get('models', [])]
                
                if not models:
                    raise Exception("No models found in response")
                
                logger.info(f"Found {len(models)} available models: {models}")
                return models
        except Exception as e:
            wait_time = base_wait * (2 ** attempt)
            logger.error(f"Failed to fetch models (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
            else:
                logger.error("Failed to fetch models after all retries")
                return []

async def run_prompt_on_model(
    model: str, 
    prompt: str, 
    session: aiohttp.ClientSession,
    timeout: int = 300
) -> Dict:
    """
    Runs a prompt on a specific model and returns detailed results including timing and errors.
    Always generates fresh responses, ignoring any caching.
    """
    start_time = datetime.now()
    generation_start = None
    generation_end = None
    execution_start = None
    execution_end = None
    
    try:
        # Get the system prompt to ensure consistent context
        from .system_prompt import get_system_prompt
        system_prompt = await get_system_prompt()
        
        # Build the messages array with system prompt and user query
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        # Use the correct Ollama endpoint
        api_url = LLM_API_URL.rstrip("/api/chat")  # Remove /api/chat if present
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "raw": True  # Request raw output to bypass any caching
        }
        
        generation_start = datetime.now()
        print(f"\nStarting generation for {model}...")
        
        async with session.post(
            f"{api_url}/api/chat",  # Correct endpoint for chat completion
            json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout)
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
            response_text = data.get("message", {}).get("content", "") or data.get("response", "")
            generation_end = datetime.now()
            
            print(f"{model}: Generation completed in {(generation_end - generation_start).total_seconds():.2f}s")
            
            # Extract and execute any code blocks
            code_results = []
            code_blocks = extract_code_blocks(response_text)
            
            if code_blocks:
                execution_start = datetime.now()
                print(f"{model}: Executing {len(code_blocks)} code blocks...")
                for lang, code in code_blocks:
                    try:
                        ret_code, output = await execute_code_async(lang, code)
                        code_results.append({
                            "language": lang,
                            "code": code,
                            "return_code": ret_code,
                            "output": output,
                            "success": ret_code == 0
                        })
                    except Exception as e:
                        code_results.append({
                            "language": lang,
                            "code": code,
                            "return_code": -1,
                            "output": f"Execution error: {str(e)}",
                            "success": False
                        })
                execution_end = datetime.now()
                print(f"{model}: Code execution completed in {(execution_end - execution_start).total_seconds():.2f}s")
            elif not code_blocks and not response_text.strip():
                # If no response or code blocks, consider it an error
                return {
                    "model": model,
                    "response": "",
                    "timing": {
                        "total_time": (datetime.now() - start_time).total_seconds(),
                        "generation_time": (generation_end - generation_start).total_seconds() if generation_end else None,
                        "execution_time": None,
                        "generation_tokens_per_second": None
                    },
                    "code_results": [],
                    "error": "Model did not generate any response or code blocks",
                    "token_count": None
                }

            # Calculate timing metrics
            end_time = datetime.now()
            timing = {
                "total_time": (end_time - start_time).total_seconds(),
                "generation_time": (generation_end - generation_start).total_seconds() if generation_end else None,
                "execution_time": (execution_end - execution_start).total_seconds() if execution_end else None,
                "generation_tokens_per_second": (data.get("usage", {}).get("total_tokens", 0) / 
                    (generation_end - generation_start).total_seconds()) if generation_end and data.get("usage") else None
            }

            return {
                "model": model,
                "response": response_text,
                "timing": timing,
                "code_results": code_results,
                "error": None,
                "token_count": data.get("usage", {}).get("total_tokens")
            }
            
    except asyncio.TimeoutError:
        print(f"\n{model}: Request timed out after {timeout} seconds")
        end_time = datetime.now()
        timing = {
            "total_time": (end_time - start_time).total_seconds(),
            "generation_time": None,
            "execution_time": None,
            "generation_tokens_per_second": None
        }
        return {
            "model": model,
            "response": "",
            "timing": timing,
            "code_results": [],
            "error": f"Request timed out after {timeout} seconds",
            "token_count": None
        }
    except Exception as e:
        print(f"\n{model}: Error occurred: {str(e)}")
        end_time = datetime.now()
        timing = {
            "total_time": (end_time - start_time).total_seconds(),
            "generation_time": (generation_end - generation_start).total_seconds() if generation_end else None,
            "execution_time": None,
            "generation_tokens_per_second": None
        }
        return {
            "model": model,
            "response": "",
            "timing": timing,
            "code_results": [],
            "error": str(e),
            "token_count": None
        }

async def compare_models(
    prompt: str, 
    session: aiohttp.ClientSession, 
    db: ConversationDB,
    specific_models: Optional[List[str]] = None
) -> List[Dict]:
    """
    Runs the prompt on all available models (or specified models) and returns comparison results.
    Shows real-time progress updates as each model completes.
    
    Args:
        prompt: The prompt to test
        session: aiohttp ClientSession
        db: Database connection for storing results
        specific_models: Optional list of specific models to test (if None, tests all available)
    
    Returns:
        List of result dictionaries from each model
    """
    try:
        # Get available models
        models = specific_models or await get_available_models(session)
        if not models:
            logger.error("No models available for comparison")
            return []

        # Initialize progress tracking
        total_models = len(models)
        completed_models = set()
        print(f"\nComparing prompt across {total_models} models:")
        for model in models:
            print(f"  • {model}: Pending...")
        print("\nProgress:")
        
        async def run_model_with_progress(model: str) -> Dict:
            try:
                result = await run_prompt_on_model(model, prompt, session)
                completed_models.add(model)
                # Clear previous line and show updated progress
                print(f"\033[1A\033[K  • {model}: Completed ✓")
                print(f"Progress: {len(completed_models)}/{total_models} models completed")
                return result
            except Exception as e:
                completed_models.add(model)
                print(f"\033[1A\033[K  • {model}: Failed ✗ ({str(e)})")
                print(f"Progress: {len(completed_models)}/{total_models} models completed")
                return e
        
        # Run prompts concurrently with progress tracking
        tasks = [run_model_with_progress(model) for model in models]
        results = await asyncio.gather(*tasks)
        
        # Process results and handle any exceptions
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Error during model comparison: {result}")
                continue
            processed_results.append(result)
            
            # Store results in database
            try:
                db.add_comparison_result(prompt, result)
            except Exception as e:
                logger.error(f"Failed to store comparison result: {e}")

        print(f"\nCompleted {len(processed_results)}/{total_models} model comparisons successfully.\n")
        return processed_results
        
    except Exception as e:
        logger.error(f"Error during model comparison: {e}")
        return []

def analyze_results(results: List[Dict]) -> str:
    """
    Analyzes the comparison results and returns a formatted analysis string.
    
    Args:
        results: List of result dictionaries from compare_models
    
    Returns:
        Formatted string containing the analysis
    """
    if not results:
        return "No results to analyze."
    
    analysis = []
    
    # First, show the performance comparison table
    analysis.append("\n=== Performance Comparison ===\n")
    
    # Summary table with enhanced timing metrics
    headers = [
        ("Model", 20),
        ("Total (s)", 10),
        ("Gen (s)", 10),
        ("Exec (s)", 10),
        ("Tok/s", 10),
        ("Tokens", 10),
        ("Success", 10),
        ("Code", 8)
    ]
    
    # Print headers
    header_line = ""
    for header, width in headers:
        header_line += f"{header:<{width}}"
    analysis.append(header_line + "\n")
    analysis.append("-" * (sum(width for _, width in headers)) + "\n")
    
    # Sort results by total time
    sorted_results = sorted(results, key=lambda x: x['timing']['total_time'])
    
    for result in sorted_results:
        timing = result['timing']
        model_name = result['model']
        error = result.get('error')
        token_count = result.get('token_count', 'N/A')
        code_blocks = len(result.get('code_results', []))
        success = "✅" if not error else "❌"
        
        # Format timing values with appropriate precision
        total_time = f"{timing['total_time']:.2f}" if timing['total_time'] is not None else "N/A"
        gen_time = f"{timing['generation_time']:.2f}" if timing['generation_time'] is not None else "N/A"
        exec_time = f"{timing['execution_time']:.2f}" if timing['execution_time'] is not None else "N/A"
        tokens_per_sec = f"{timing['generation_tokens_per_second']:.1f}" if timing['generation_tokens_per_second'] is not None else "N/A"
        
        # Build the line with consistent spacing
        line = (
            f"{model_name:<20}"
            f"{total_time:<10}"
            f"{gen_time:<10}"
            f"{exec_time:<10}"
            f"{tokens_per_sec:<10}"
            f"{str(token_count):<10}"
            f"{success:<10}"
            f"{code_blocks:<8}"
        )
        analysis.append(line + "\n")
    
    # Add timing metrics explanation
    analysis.append("\nTiming Metrics:\n")
    analysis.append("- Total (s)  : Total time including all operations\n")
    analysis.append("- Gen (s)    : Time spent on response generation\n")
    analysis.append("- Exec (s)   : Time spent on code execution\n")
    analysis.append("- Tok/s      : Tokens generated per second\n")
    
    # Then, show side-by-side response comparison
    analysis.append("\n=== Response Comparison ===\n")
    
    # First, show just the responses side by side (if possible)
    max_width = 40  # Width for each response column
    model_responses = []
    
    for result in sorted_results:
        model_name = result['model']
        response = result.get('response', '').strip()
        model_responses.append((model_name, response))
    
    # Show responses in columns if we have 2 or 3 models
    if 2 <= len(model_responses) <= 3:
        # Split responses into lines
        split_responses = [(name, resp.split('\n')) for name, resp in model_responses]
        max_lines = max(len(lines) for _, lines in split_responses)
        
        # Print headers
        for name, _ in split_responses:
            analysis.append(f"{name:<{max_width}}")
        analysis.append("\n")
        analysis.append("=" * (max_width * len(split_responses)) + "\n")
        
        # Print responses line by line
        for i in range(max_lines):
            for _, lines in split_responses:
                line = lines[i] if i < len(lines) else ""
                analysis.append(f"{line:<{max_width}}")
            analysis.append("\n")
        
        analysis.append("=" * (max_width * len(split_responses)) + "\n\n")
    
    # Detailed analysis of each model
    analysis.append("\n=== Detailed Analysis ===\n")
    
    for result in sorted_results:
        analysis.append("\n" + "=" * 80 + "\n")
        analysis.append(f"Model: {result['model']}\n")
        analysis.append("=" * 80 + "\n")
        
        # Show timing information
        timing = result['timing']
        analysis.append(f"Timing:\n")
        analysis.append(f"  - Total Time: {timing['total_time']:.2f}s\n")
        if timing['generation_time']:
            analysis.append(f"  - Generation Time: {timing['generation_time']:.2f}s\n")
        if timing['execution_time']:
            analysis.append(f"  - Code Execution Time: {timing['execution_time']:.2f}s\n")
        if timing['generation_tokens_per_second']:
            analysis.append(f"  - Generation Speed: {timing['generation_tokens_per_second']:.1f} tokens/s\n")
        
        # Show full response
        if result['error']:
            analysis.append(f"\nError: {result['error']}\n")
        else:
            analysis.append("\nFull Response:\n")
            analysis.append("-" * 80 + "\n")
            analysis.append(result['response'].strip() + "\n")
            analysis.append("-" * 80 + "\n")
            
            # Show code execution results
            if result['code_results']:
                analysis.append("\nCode Execution Results:\n")
                for i, code_result in enumerate(result['code_results'], 1):
                    analysis.append(f"\nCode Block #{i} ({code_result['language']}):\n")
                    analysis.append("-" * 40 + "\n")
                    analysis.append(f"Code:\n{code_result['code']}\n")
                    analysis.append(f"Success: {'✅' if code_result['success'] else '❌'}\n")
                    if not code_result['success']:
                        analysis.append(f"Error: {code_result['output']}\n")
                    else:
                        analysis.append(f"Output:\n{code_result['output']}\n")
                    analysis.append("-" * 40 + "\n")
    
    return "".join(analysis) 