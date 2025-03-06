from google import genai

client = genai.Client(api_key="AIzaSyD5Kom_ek39c882kX5ELrZZT3UpBB8H46I")
response = client.models.generate_content(
    model="gemini-2.0-flash", contents="say hello in vietnamese"
)
print(response.text)