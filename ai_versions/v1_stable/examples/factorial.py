def calculate_factorial(n):
    """Calculate factorial of a number"""
    if n < 0:
        return None
    result = 1
    for i in range(1, n + 1):
        result = result * i
    return result

def fibonacci(n):
    """Generate Fibonacci sequence"""
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]
    
    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[i-1] + fib[i-2])
    return fib

# Example usage
if __name__ == "__main__":
    print(calculate_factorial(5))
    print(fibonacci(10))