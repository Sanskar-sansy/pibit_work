from app.llm.ollama_client import OllamaClient

client = OllamaClient()

print("Ollama available:", client.is_available())

response = client.generate(
    model="qwen2.5-coder:7b",
    prompt="Explain JSON extraction in 2 lines."
)

print("\nRESPONSE:\n")
print(response["response"])

print("\nSTATS:\n")
print(client.stats())