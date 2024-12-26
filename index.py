import os
from dotenv import load_dotenv  # Importa load_dotenv
from google.cloud import vision
import requests
import json
from PIL import Image
import io
import fitz

# Carga las variables de entorno desde el archivo .env
load_dotenv()

# Configuración (IMPORTANTE: Configura tu clave de API como variable de entorno)
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("Error: La variable de entorno GEMINI_API_KEY no está definida. Consulta las instrucciones para configurarla.")

GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"

def procesar_pdf_con_vision_y_gemini(pdf_path, prompt_gemini="Extrae los ativos totais do seguinte texto em português:\n\n"):
    """
    Procesa un PDF con la API de Vision para OCR y luego con la API de Gemini para análisis.

    Args:
        pdf_path: Ruta al archivo PDF.
        prompt_gemini: Prompt para enviar a Gemini (por defecto, extrae "ativos totais").

    Returns:
        El resultado del análisis de Gemini o un mensaje de error detallado.
    """
    try:
        # 1. Abre el PDF con PyMuPDF (fitz)
        try:
            doc = fitz.open(pdf_path)
        except fitz.fitz.FileNotFoundError:
            return f"Error: No se encontró el archivo PDF en la ruta: {pdf_path}"
        except Exception as e:
            return f"Error al abrir el PDF: {e}"

        textos_por_pagina = []

        # 2. Itera sobre las páginas del PDF
        for page_num in range(doc.page_count):
            try:
                page = doc[page_num]
                pix = page.get_pixmap(matrix=fitz.Matrix(3, 3)) # Mejora la resolución para OCR
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                image_bytes = io.BytesIO()
                img.save(image_bytes, format="PNG")
                image_content = image_bytes.getvalue()

                # 3. OCR con Google Cloud Vision API
                client = vision.ImageAnnotatorClient()
                image = vision.Image(content=image_content)
                response_vision = client.document_text_detection(image=image)

                if response_vision.error.message:
                    print(f"Advertencia: Error en OCR de la página {page_num + 1}: {response_vision.error.message}")
                    continue  # Continúa con la siguiente página si hay un error en OCR

                text = response_vision.full_text_annotation.text if response_vision.full_text_annotation else None

                if text:
                    textos_por_pagina.append(text)
                else:
                    print(f"Advertencia: No se pudo extraer texto de la página {page_num + 1}.")

            except Exception as e:
                print(f"Error al procesar la página {page_num + 1}: {e}")

        doc.close() # Cierra el documento PDF

        texto_completo = "\n\n".join(textos_por_pagina)

        if not texto_completo:
            return "Advertencia: No se pudo extraer texto de ninguna página del PDF."

        # 4. Envía el texto a la API de Gemini
        headers = {
            "Content-Type": "application/json"
        }

        data = {
            "contents": [{
            "parts":[{"text": f"{prompt_gemini}{texto_completo}"}]
            }]
        }

        try:
            response_gemini = requests.post(GEMINI_API_URL, headers=headers, json=data, params={"key": API_KEY})
            response_gemini.raise_for_status()  # Lanza una excepción para códigos de estado HTTP erróneos
            response_json = response_gemini.json()
        except requests.exceptions.RequestException as e:
            if response_gemini is not None:
                try:
                    error_data = response_gemini.json()
                    return f"Error en la solicitud a la API de Gemini: {e}\nDetalles del error: {error_data}"
                except json.JSONDecodeError:
                    return f"Error en la solicitud a la API de Gemini: {e}\nCódigo de estado: {response_gemini.status_code}\nRespuesta (no JSON): {response_gemini.text}"
            return f"Error en la solicitud a la API de Gemini: {e}"

        # 5. Procesa la respuesta de Gemini (manejo robusto de errores)
        try:
            candidates = response_json.get('candidates', [])
            if candidates:
                parts = candidates[0]['content'].get('parts', [])
                if parts:
                    return parts[0].get('text', "Respuesta de Gemini sin texto.")
                else:
                    return "Error: Respuesta de Gemini sin 'parts'."
            else:
                return "Error: Respuesta de Gemini sin 'candidates'."
        except (KeyError, IndexError, TypeError) as e:
            print(f"Error al procesar la respuesta de Gemini (formato inesperado): {e}")
            return f"Respuesta completa de Gemini (para depuración):\n{response_json}"

    except Exception as e:
        return f"Error general al procesar el PDF: {e}"

# Ejemplo de uso
pdf_path = "orestes.pdf"  # Reemplaza con la ruta a tu PDF
prompt_personalizado = "Extrae el activo o los activos totales con su fecha el texto esta en portugues genera un json con esa informacion:\n\n"
resultado = procesar_pdf_con_vision_y_gemini(pdf_path, prompt_personalizado)
print(resultado)