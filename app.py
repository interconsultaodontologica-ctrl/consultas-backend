from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os
import concurrent.futures

app = Flask(__name__)
CORS(app)

GROQ_KEY = os.environ.get("GROQ_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

def consultar_groq(pregunta, historial, doc_nombre, doc_contenido):
    system = f"""Eres el Asistente de Interconsultas Odontológicas de la Corporación de Salud de San Bernardo (CORSABER), Chile.
Responde ÚNICAMENTE basándote en el siguiente documento: {doc_nombre}.
Si la información no está en el documento, responde exactamente: "NO_ENCONTRADO".
Jamás inventes información. Responde en español, de forma clara y estructurada.

DOCUMENTO:
{doc_contenido[:28000]}"""

    messages = [{"role": "system", "content": system}]
    for m in historial[:-1]:
        messages.append({
            "role": "assistant" if m["role"] == "model" else "user",
            "content": m["parts"][0]["text"]
        })
    messages.append({"role": "user", "content": pregunta})

    body = {
        "model": "llama-3.3-70b-versatile",
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 1000
    }

    res = requests.post(GROQ_URL, json=body, headers={
        "Authorization": f"Bearer {GROQ_KEY}",
        "Content-Type": "application/json"
    })
    data = res.json()
    if "choices" in data:
        return data["choices"][0]["message"]["content"]
    return "NO_ENCONTRADO"

def sintetizar_respuestas(pregunta, respuestas):
    contenido = "\n\n".join([f"Documento '{doc}': {resp[:3000]}" 
                              for doc, resp in respuestas.items() 
                              if resp != "NO_ENCONTRADO"])
    
    if not contenido:
        return "No encontré información sobre tu consulta en los documentos disponibles. Te sugiero consultar directamente con el Odontólogo Interconsultor Comunal."

    system = """Eres el Asistente de Interconsultas Odontológicas de CORSABER.
Recibirás respuestas parciales de varios documentos sobre una misma pregunta.
Tu tarea es sintetizar toda esa información en UNA sola respuesta clara, estructurada y sin repeticiones.
Si hay información complementaria entre documentos, intégrala.
Al final indica en qué documento(s) encontraste la información.
Jamás inventes información. Responde en español."""

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Pregunta original: {pregunta}\n\nRespuestas por documento:\n{contenido}"}
    ]

    body = {
        "model": "llama-3.3-70b-versatile",
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 1500
    }

    res = requests.post(GROQ_URL, json=body, headers={
        "Authorization": f"Bearer {GROQ_KEY}",
        "Content-Type": "application/json"
    })
    data = res.json()
    if "choices" in data:
        return data["choices"][0]["message"]["content"]
    # Si falla síntesis, devolver la primera respuesta válida directamente
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

    # Consultar todos los documentos en paralelo
    respuestas = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futuros = {
            executor.submit(consultar_groq, pregunta, historial, doc["name"], doc["content"]): doc["name"]
            for doc in documentos
        }
        for futuro in concurrent.futures.as_completed(futuros):
            nombre = futuros[futuro]
            respuestas[nombre] = futuro.result()

    # Sintetizar todas las respuestas en una sola
    respuesta_final = sintetizar_respuestas(pregunta, respuestas)
    return jsonify({"respuesta": respuesta_final}), 200

if __name__ == "__main__":
    app.run(debug=False)