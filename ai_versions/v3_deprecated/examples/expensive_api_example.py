# Example of expensive API usage with high latency
import openai
import time
import sys

def analyze_code_with_gpt35(code: str, language: str = "python") -> dict:
    """
    Analyze code using GPT-3.5 Turbo
    This demonstrates why v3 was deprecated:
    - High latency (12+ seconds)
    - Expensive ($0.05 per review)
    - Rate limited
    """
    start_time = time.time()
    
    try:
        # Expensive API call
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a code reviewer."},
                {"role": "user", "content": f"Review this {language} code:\n\n{code}"}
            ],
            max_tokens=1000,
            temperature=0.3
        )
        
        processing_time = time.time() - start_time
        cost = 0.05  # Approximate cost per review
        
        return {
            "review": response.choices[0].message.content,
            "processing_time": processing_time,
            "cost": cost,
            "tokens_used": response.usage.total_tokens,
            "model": "gpt-3.5-turbo"
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "processing_time": time.time() - start_time,
            "cost": 0.05,  # Still charged for failed attempts
            "model": "gpt-3.5-turbo"
        }

def batch_analyze(code_snippets: list) -> list:
    """
    Batch analysis - demonstrates rate limiting issues
    """
    results = []
    
    for i, code in enumerate(code_snippets):
        print(f"Analyzing snippet {i+1}/{len(code_snippets)}...")
        
        # Rate limiting - need to wait between requests
        if i > 0:
            time.sleep(2)  # Avoid rate limits
        
        result = analyze_code_with_gpt35(code)
        results.append(result)
        
        # Show why this is problematic
        print(f"  Time: {result.get('processing_time', 0):.2f}s")
        print(f"  Cost: ${result.get('cost', 0):.4f}")
        print(f"  Tokens: {result.get('tokens_used', 0)}")
    
    return results

# Example usage that shows the problems
if __name__ == "__main__":
    sample_code = """
def process_user_data(user_input):
    # Process user input without validation
    data = eval(user_input)  # Security issue!
    return data
"""
    
    print("=== GPT-3.5 Turbo Analysis (v3 - Deprecated) ===")
    result = analyze_code_with_gpt35(sample_code)
    
    print(f"Processing Time: {result.get('processing_time', 0):.2f} seconds")
    print(f"Cost: ${result.get('cost', 0):.4f}")
    print(f"Tokens Used: {result.get('tokens_used', 0)}")
    
    if 'error' in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Review: {result['review'][:200]}...")
    
    print("\n=== Issues with this approach ===")
    print("1. High latency - 12+ seconds per review")
    print("2. Expensive - $0.05 per review")
    print("3. Rate limited - ~60 requests per minute")
    print("4. External dependency - API downtime affects service")
    print("5. Privacy concerns - code sent to external service")