# agent/main.py
import asyncio
import argparse
from .agent import MinimalAIAgent
from .config import LLM_MODEL, RESPONSE_TIMEOUT, MAX_CONCURRENT_LLM_CALLS, MAX_CONCURRENT_CODE_EXECUTIONS

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='MacBot AI Agent - A command-line AI assistant for macOS')
    
    # Model configuration
    parser.add_argument('--model', type=str, default=LLM_MODEL, 
                        help=f'LLM model to use (default: {LLM_MODEL})')
    
    # Performance settings
    parser.add_argument('--timeout', type=int, default=RESPONSE_TIMEOUT, 
                        help=f'Response timeout in seconds (default: {RESPONSE_TIMEOUT})')
    parser.add_argument('--max-llm-calls', type=int, default=MAX_CONCURRENT_LLM_CALLS, 
                        help=f'Maximum concurrent LLM calls (default: {MAX_CONCURRENT_LLM_CALLS})')
    parser.add_argument('--max-code-execs', type=int, default=MAX_CONCURRENT_CODE_EXECUTIONS, 
                        help=f'Maximum concurrent code executions (default: {MAX_CONCURRENT_CODE_EXECUTIONS})')
    parser.add_argument('--threads', type=int, default=None,
                        help='Number of CPU threads for Ollama (default: auto-detected)')
    parser.add_argument('--gpu-layers', type=int, default=None,
                        help='Number of GPU layers to use (default: 0, CPU-only mode)')
    
    # Debug options
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with detailed logging')
    parser.add_argument('--save-code', action='store_true', help='Save executed code blocks')
    
    return parser.parse_args()

async def main():
    """Main entry point for the MacBot AI Agent."""
    args = parse_arguments()
    
    # Create agent with command line arguments
    agent = MinimalAIAgent(
        model=args.model,
        timeout=args.timeout,
        max_llm_calls=args.max_llm_calls,
        max_code_execs=args.max_code_execs,
        debug_mode=args.debug,
        save_code=args.save_code
    )
    
    # Set thread count and GPU layers if specified
    if args.threads is not None:
        agent.ollama_config["num_thread"] = args.threads
        agent.db.set_setting("ollama_num_thread", str(args.threads))
        
    if args.gpu_layers is not None:
        agent.ollama_config["num_gpu"] = args.gpu_layers
        agent.db.set_setting("ollama_num_gpu", str(args.gpu_layers))
    
    await agent.run()

if __name__ == "__main__":
    asyncio.run(main())
