from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os
import re
import time

app = Flask(__name__)
CORS(app)

GROQ_KEY = os.environ.get("GROQ_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODELO = "llama-3.1-8b-instant"

def extraer_fragmentos_relevantes(pregunta, contenido, max_chars=4000):
    palabras = re.findall(r'\w+', pregunta.lower())
    palabras_clave = [p for p in palabras if len(p) > 3]
    parrafos = [p.strip() for p in re.split(r'\n+', contenido) if len(p.strip()) > 50]
    puntuados = []
    for parrafo in parrafos:
        parrafo_lower = parrafo.lower()
        puntaje = sum(1 for palabra in palabras_clave if palabra in parrafo_lower)
        if puntaje > 0:
            puntuados.append((puntaje, parrafo))
    puntuados.sort(reverse=True)
    fragmentos = []
    total_chars = 0
    for _, parrafo in puntuados:
        if total_chars + len(parrafo) > max_chars:
            break
        fragmentos.append(parrafo)
        total_chars += len(parrafo)
    if not fragmentos:
        return contenido[:2000]
    return "\n\n".join(fragmentos)

def consultar_groq(pregunta, historial, doc_nombre, doc_contenido):
    fragmento = extraer_fragmentos_relevantes(pregunta, doc_contenido)
    system = f"""Eres el Asistente de Interconsultas Odontológicas de CORSABER, Chile.
Responde ÚNICAMENTE basándote en el siguiente fragmento del documento: {doc_nombre}.
Si la información no está en el fragmento, responde exactamente: "NO_ENCONTRADO".
Jamás inventes información. Responde en español, de forma clara y estructurada.

FRAGMENTO DEL DOCUMENTO:
{fragmento}"""
    messages = [{"role": "system", "content": system}]
    for m in historial[:-1]:
        messages.append({
            "role": "assistant" if m["role"] == "model" else "user",
            "content": m["parts"][0]["text"]
        })
    messages.append({"role": "user", "content": pregunta})
    body = {
        "model": MODELO,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 800
    }
    res = requests.post(GROQ_URL, json=body, headers={
        "Authorization": f"Bearer {GROQ_KEY}",
        "Content-Type": "application/json"
    })
    data = res.json()
    if "choices" in data:
        resultado = data["choices"][0]["message"]["content"]
        print(f"DOC: {doc_nombre} | TOKENS: {data.get('usage', {}).get('total_tokens', '?')} | RESPUESTA: {resultado[:100]}")
        return resultado
    print(f"DOC: {doc_nombre} | ERROR: {data}")
    return "NO_ENCONTRADO"

def sintetizar_respuestas(pregunta, respuestas):
    contenido = "\n\n".join([
        f"Documento '{doc}':\n{resp[:2000]}"
        for doc, resp in respuestas.items()
        if resp != "NO_ENCONTRADO"
    ])
    if not contenido:
        return "No encontré información sobre tu consulta en los documentos disponibles. Te sugiero consultar directamente con el Odontólogo Interconsultor Comunal."
    system = """Eres el Asistente de Interconsultas Odontológicas de CORSABER.
Sintetiza la información de los documentos en UNA respuesta clara y sin repeticiones.
Integra información complementaria si la hay.
Al final indica en qué documento(s) encontraste la información.
Jamás inventes. Responde en español."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Pregunta: {pregunta}\n\nRespuestas por documento:\n{contenido}"}
    ]
    body = {
        "model": MODELO,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 800
    }
    res = requests.post(GROQ_URL, json=body, headers={
        "Authorization": f"Bearer {GROQ_KEY}",
        "Content-Type": "application/json"
    })
    data = res.json()
    print(f"SÍNTESIS | TOKENS: {data.get('usage', {}).get('total_tokens', '?')}")
    if "choices" in data:
        return data["choices"][0]["message"]["content"]
    for doc, resp in respuestas.items():
        if resp != "NO_ENCONTRADO":
            return f"**{doc}:**\n{resp}"
    return "No encontré información en los documentos disponibles."

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    pregunta = data.get("pregunta", "")
    historial = data.get("contents", [])
    documentos = data.get("documentos", [])
    if not documentos:
        return jsonify({"respuesta": "No hay documentos disponibles."}), 200
    respuestas = {}
    for doc in documentos:
        respuestas[doc["name"]] = consultar_groq(pregunta, historial, doc["name"], doc["content"])
        time.sleep(3)
    respuesta_final = sintetizar_respuestas(pregunta, respuestas)
    return jsonify({"respuesta": respuesta_final}), 200

if __name__ == "__main__":
    app.run(debug=False)