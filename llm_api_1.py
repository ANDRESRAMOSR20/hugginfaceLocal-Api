from huggingface_hub import InferenceClient

client = InferenceClient(api_key="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")

messages = [
	{
		"role": "user",
		"content": "Which is the lightest element of the metals?"
	}
]

completion = client.chat.completions.create(
    model="mistralai/Mistral-7B-Instruct-v0.3",
	messages=messages,
	max_tokens=500
)

print(completion.choices[0].message)