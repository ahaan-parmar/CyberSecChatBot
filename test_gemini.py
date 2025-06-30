import google.generativeai as genai

# Configure the API
genai.configure(api_key='AIzaSyDnAn7O9hWTBD3axmXPPeMEnV9886pAlxY')

print("Available Gemini models:")
try:
    for model in genai.list_models():
        if 'generateContent' in model.supported_generation_methods:
            print(f"Model: {model.name}")
            print(f"Display Name: {model.display_name}")
            print(f"Supports: {model.supported_generation_methods}")
            print("---")
except Exception as e:
    print(f"Error listing models: {e}")

# Test with a specific model
try:
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content('Hello, this is a test message')
    print(f"Test successful with gemini-1.5-flash: {response.text[:100]}...")
except Exception as e:
    print(f"Error with gemini-1.5-flash: {e}")
    
    # Try another model
    try:
        model = genai.GenerativeModel('gemini-1.5-pro')
        response = model.generate_content('Hello, this is a test message')
        print(f"Test successful with gemini-1.5-pro: {response.text[:100]}...")
    except Exception as e:
        print(f"Error with gemini-1.5-pro: {e}")
