import requests

url = "http://localhost:23333/v1/completions"
headers = {
    "Content-Type": "application/json"
}
payload = {
    "model": "facebook/opt-125m",
    "prompt": "happy happy"
}

response = requests.post(url, json=payload, headers=headers)

print("Status Code:", response.status_code)
print("Response JSON:", response.json())