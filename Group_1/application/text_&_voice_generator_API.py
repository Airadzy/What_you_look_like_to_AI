import json
import requests


def text_generator():
    # key API
    headers = {
        "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiZTZhOGQxZmMtZjc1Ny00OTVlLTgzNzgtMDJkZjExNjhhNTg3IiwidHlwZSI6ImFwaV90b2tlbiJ9.t6D9SSDFgLrHy1GmywPam2MHTrN2qcEIZWsYO3KZEbk"}

    url = "https://api.edenai.run/v2/text/generation"

    # create the prompt attributes should take from the prediction
    intro = " create a text that describe a personn with the following attribute in two line : "
    attributes = "brwon hair, glasses, blue eyes"
    prompt = intro + attributes

    # create the question
    payload = {"providers": "openai,cohere",
               "text": prompt,
               "temperature": 0.2,
               "max_tokens": 250
               }
    response = requests.post(url, json=payload, headers=headers)

    # print the answer
    result = json.loads(response.text)
    description = result['openai']['generated_text']
    return description


# creating the voice
def voice_generator(description):
    CHUNK_SIZE = 1024
    url = "https://api.elevenlabs.io/v1/text-to-speech/pNInz6obpgDQGcFmaJgB"

    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": "642a30a506ea5fc233f1997618b4cb05"
    }

    data = {
        "text": description,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5
        }
    }

    response = requests.post(url, json=data, headers=headers)
    with open('output.mp3', 'wb') as f:
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            if chunk:
                f.write(chunk)


def main():
    result = text_generator()
    voice_generator(result)


if __name__ == "__main__":
    main()
